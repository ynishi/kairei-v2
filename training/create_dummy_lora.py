#!/usr/bin/env python3
"""
Create a small dummy LoRA adapter for testing Kairei CLI
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
import os
from pathlib import Path

def create_dummy_lora(
    model_name: str = "stories15M",  # This would normally be a HF model ID
    output_dir: str = "./dummy_lora",
    lora_rank: int = 8,
    lora_alpha: int = 16,
):
    """Create a minimal LoRA adapter for testing"""
    
    print(f"Creating dummy LoRA adapter...")
    
    # For dummy purposes, we'll create a small model structure
    # In real usage, you'd load an actual model like:
    # model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a dummy LoRA configuration
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],  # Common targets for LLaMA models
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Save the config
    lora_config.save_pretrained(output_dir)
    
    # Create dummy weights for the adapter
    # In a real scenario, these would be trained weights
    dummy_weights = {}
    
    # Create dummy LoRA A and B matrices
    # Format: base_model.model.layers.{layer_num}.{module}.lora_{A/B}.weight
    for layer in range(2):  # Just 2 layers for dummy
        for module in ["q_proj", "v_proj"]:
            # LoRA decomposes weight updates into low-rank matrices A and B
            # W' = W + BA where B is (out_features x r) and A is (r x in_features)
            dummy_weights[f"base_model.model.layers.{layer}.{module}.lora_A.weight"] = torch.randn(lora_rank, 768)  # 768 is typical hidden size
            dummy_weights[f"base_model.model.layers.{layer}.{module}.lora_B.weight"] = torch.randn(768, lora_rank)
    
    # Save as safetensors
    from safetensors.torch import save_file
    
    adapter_path = os.path.join(output_dir, "adapter_model.safetensors")
    save_file(dummy_weights, adapter_path)
    
    print(f"✅ Created dummy LoRA adapter at: {output_dir}")
    print(f"   - adapter_model.safetensors: {os.path.getsize(adapter_path) / 1024:.2f} KB")
    print(f"   - adapter_config.json")
    
    return output_dir

if __name__ == "__main__":
    # Create a small dummy LoRA for testing
    output_dir = create_dummy_lora(
        model_name="stories15M",
        output_dir="../lora_datasets/dummy_lora_stories15M",
        lora_rank=4,  # Even smaller for testing
        lora_alpha=8,
    )