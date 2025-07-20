#!/usr/bin/env python3
"""
TinyLlama LoRA Training Script
CPU-friendly training with minimal configuration
"""
import os
import json
import time
import torch
import argparse
from pathlib import Path
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling
)
from transformers import TrainingArguments, Trainer
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
from datasets import Dataset


class TinyLlamaLoRATrainer:
    """Simple LoRA trainer for TinyLlama"""
    
    def __init__(self, model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.model_id = model_id
        self.tokenizer = None
        self.model = None
        self.output_dir = Path("./lora_output")
        self.output_dir.mkdir(exist_ok=True)
        
    def load_model(self):
        """Load TinyLlama model and tokenizer"""
        print(f"🔄 Loading {self.model_id}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        # Load model (CPU-friendly settings)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32,  # CPU prefers float32
            low_cpu_mem_usage=True
        )
        
        print("✅ Model loaded successfully!")
        print(f"   Model size: {sum(p.numel() for p in self.model.parameters()) / 1e9:.2f}B parameters")
        
    def prepare_lora(self, r=4, alpha=8, dropout=0.05):
        """Apply LoRA configuration"""
        print(f"\n🔧 Applying LoRA (r={r}, alpha={alpha}, dropout={dropout})...")
        
        # LoRA configuration for CPU
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"],
            bias="none"
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
    def load_dataset(self, train_file, val_file=None, max_length=256):
        """Load and prepare dataset"""
        print(f"\n📚 Loading dataset from {train_file}...")
        
        # Load JSON data
        with open(train_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        # Handle different data formats
        if isinstance(raw_data, dict) and all(isinstance(v, list) for v in raw_data.values()):
            # Format like dialogue_dataset.json with themes
            train_data = []
            for theme, items in raw_data.items():
                train_data.extend(items)
            print(f"   Loaded {len(train_data)} examples from {len(raw_data)} themes")
        elif isinstance(raw_data, list):
            # Direct list format
            train_data = raw_data
        else:
            raise ValueError(f"Unsupported data format in {train_file}")
            
        # Convert to format suitable for training
        def format_instruction(example):
            if "instruction" in example:
                # Alpaca format
                text = f"### Instruction:\n{example['instruction']}\n\n"
                if example.get("input", ""):
                    text += f"### Input:\n{example['input']}\n\n"
                text += f"### Response:\n{example['output']}"
            elif "question" in example and "answer" in example:
                # Q&A format
                text = f"### Instruction:\n{example['question']}\n\n"
                text += f"### Response:\n{example['answer']}"
            else:
                # Simple text format
                text = example.get("text", "")
            
            return {"text": text}
        
        # Create dataset
        train_dataset = Dataset.from_list([format_instruction(ex) for ex in train_data])
        
        # Tokenize
        def tokenize_function(examples):
            # Handle both single examples and batches
            if isinstance(examples["text"], list):
                texts = examples["text"]
            else:
                texts = [examples["text"]]
            
            # Tokenize with proper settings
            result = self.tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors=None  # Important: don't return tensors yet
            )
            
            # Add labels (same as input_ids for language modeling)
            result["labels"] = result["input_ids"].copy()
            
            return result
        
        train_dataset = train_dataset.map(
            tokenize_function, 
            batched=True,
            remove_columns=["text"]  # Remove the text column after tokenization
        )
        
        # Validation dataset if provided
        val_dataset = None
        if val_file and Path(val_file).exists():
            with open(val_file, 'r', encoding='utf-8') as f:
                val_data = json.load(f)
            val_dataset = Dataset.from_list([format_instruction(ex) for ex in val_data])
            val_dataset = val_dataset.map(
                tokenize_function, 
                batched=True,
                remove_columns=["text"]
            )
            
        print(f"✅ Dataset loaded: {len(train_dataset)} training examples")
        if val_dataset:
            print(f"   Validation: {len(val_dataset)} examples")
            
        return train_dataset, val_dataset
    
    def train(self, train_dataset, val_dataset=None, num_epochs=3, batch_size=1, 
              gradient_accumulation_steps=4, learning_rate=5e-4, warmup_steps=10):
        """Run training"""
        print(f"\n🚀 Starting training for {num_epochs} epochs...")
        
        # Training arguments optimized for CPU
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            overwrite_output_dir=True,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            weight_decay=0.01,
            logging_steps=5,
            save_steps=50,
            save_strategy="steps",
            report_to=[],  # No external reporting
            push_to_hub=False,
            fp16=False,  # CPU doesn't support fp16
            dataloader_num_workers=0,  # Better for CPU
            remove_unused_columns=False,
            learning_rate=learning_rate,
            dataloader_pin_memory=False,  # Disable for MPS/CPU
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
        
        # Start training
        start_time = time.time()
        trainer.train()
        training_time = time.time() - start_time
        
        print(f"\n✅ Training completed in {training_time/60:.2f} minutes!")
        
        # Save the final model
        final_model_path = self.output_dir / "final_model"
        trainer.save_model(str(final_model_path))
        self.tokenizer.save_pretrained(str(final_model_path))
        
        return trainer
    
    def test_model(self, prompt="Hello, how are you?"):
        """Test the trained model with before/after comparison"""
        print(f"\n🧪 Testing with prompt: '{prompt}'")
        
        # Get model device
        device = next(self.model.parameters()).device
        
        # Tokenize inputs
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 1. Test with base model (disable LoRA)
        self.model.disable_adapter_layers()
        with torch.no_grad():
            base_outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
            )
        base_response = self.tokenizer.decode(base_outputs[0], skip_special_tokens=True)
        
        # 2. Test with LoRA model (enable LoRA)
        self.model.enable_adapter_layers()
        with torch.no_grad():
            lora_outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
            )
        lora_response = self.tokenizer.decode(lora_outputs[0], skip_special_tokens=True)
        
        # 3. Display comparison
        print(f"\n📊 Comparison:")
        print(f"├─ Base Model: {base_response}")
        print(f"└─ LoRA Model: {lora_response}")
        
        # 4. Check if different
        if base_response == lora_response:
            print(f"   ⚠️  No difference detected (might need more training)")
        else:
            print(f"   ✅ Different outputs! LoRA is having an effect")
        
        return lora_response


def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description="🎯 TinyLlama LoRA Training Script")
    
    # Model arguments
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        help="Model ID from HuggingFace (default: TinyLlama/TinyLlama-1.1B-Chat-v1.0)")
    
    # Data arguments
    parser.add_argument("--train-data", type=str, default="data/dialogue_dataset.json",
                        help="Path to training data JSON file")
    parser.add_argument("--val-data", type=str, default=None,
                        help="Path to validation data JSON file (optional)")
    
    # LoRA arguments
    parser.add_argument("--lora-r", type=int, default=8,
                        help="LoRA rank (default: 8, recommended: 8-16)")
    parser.add_argument("--lora-alpha", type=int, default=16,
                        help="LoRA alpha (default: 16, typically 2x rank)")
    parser.add_argument("--lora-dropout", type=float, default=0.05,
                        help="LoRA dropout (default: 0.05)")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs (default: 10, recommended: 10+ for meaningful results)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size per device (default: 1)")
    parser.add_argument("--gradient-accumulation", type=int, default=4,
                        help="Gradient accumulation steps (default: 4)")
    parser.add_argument("--learning-rate", type=float, default=5e-4,
                        help="Learning rate (default: 5e-4)")
    parser.add_argument("--warmup-steps", type=int, default=10,
                        help="Warmup steps (default: 10)")
    
    # Output arguments
    parser.add_argument("--output-dir", type=str, default="./lora_output",
                        help="Output directory for trained model (default: ./lora_output)")
    
    # Other arguments
    parser.add_argument("--max-length", type=int, default=256,
                        help="Maximum sequence length (default: 256)")
    parser.add_argument("--test", action="store_true",
                        help="Run test after training")
    parser.add_argument("--test-prompts", nargs="+", 
                        default=["What is LoRA fine-tuning?", "What are some best practices for error handling in Rust?", "How do memory mechanisms enhance AI agents?"],
                        help="Test prompts to use after training")
    
    args = parser.parse_args()
    
    print("🎯 TinyLlama LoRA Training Script\n")
    print("📝 Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Training data: {args.train_data}")
    print(f"  LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Output: {args.output_dir}\n")
    
    # Initialize trainer
    trainer = TinyLlamaLoRATrainer(model_id=args.model)
    trainer.output_dir = Path(args.output_dir)
    trainer.output_dir.mkdir(exist_ok=True)
    
    # Load model
    trainer.load_model()
    
    # Apply LoRA
    trainer.prepare_lora(r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
    
    # Load dataset
    train_dataset, val_dataset = trainer.load_dataset(
        train_file=args.train_data,
        val_file=args.val_data,
        max_length=args.max_length
    )
    
    # Train
    trainer.train(
        train_dataset, 
        val_dataset, 
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps
    )
    
    # Test the model if requested
    if args.test:
        print("\n" + "="*50)
        print("🧪 Testing trained model...")
        print("="*50 + "\n")
        
        for prompt in args.test_prompts:
            trainer.test_model(prompt)
            print()
    
    print(f"\n✨ Training complete! Model saved to {args.output_dir}/")


if __name__ == "__main__":
    # Check dependencies
    try:
        import peft
        import datasets
    except ImportError:
        print("❌ Missing dependencies. Please install:")
        print("   pip install peft datasets")
        exit(1)
    
    main()