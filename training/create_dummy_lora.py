#!/usr/bin/env python3
"""
Create a small dummy LoRA adapter for testing Kairei CLI
"""

import torch
from peft import LoraConfig, TaskType
import os


def create_dummy_lora(
    model_name: str = "stories15M",  # This would normally be a HF model ID
    output_dir: str = "./dummy_lora",
    lora_rank: int = 8,
    lora_alpha: int = 16,
):
    """Create a minimal LoRA adapter for testing"""

    print("Creating dummy LoRA adapter for Llama2c...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create a LoRA configuration for Llama2c
    # Target all the important layers
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
            "gate_proj", "up_proj", "down_proj",     # FFN layers
        ],
        lora_dropout=0.0,  # No dropout for testing
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Save the config
    lora_config.save_pretrained(output_dir)

    # Create dummy weights for the adapter
    # Llama2c stories15M parameters: dim=288, n_layers=6
    dim = 288
    hidden_dim = 768  # Usually 4*dim but stories15M uses 768
    n_layers = 6
    dummy_weights = {}

    # Create LoRA A and B matrices for all layers
    # Format needs to match the model structure
    for layer in range(n_layers):
        # Attention layers (self_attn)
        for module in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            # LoRA decomposes weight updates into low-rank matrices A and B
            # W' = W + (B @ A) * (alpha/r)
            key_a = f"base_model.model.layers.{layer}.self_attn.{module}.lora_A.weight"
            key_b = f"base_model.model.layers.{layer}.self_attn.{module}.lora_B.weight"
            
            dummy_weights[key_a] = torch.randn(lora_rank, dim) * 0.01
            dummy_weights[key_b] = torch.zeros(dim, lora_rank)  # Initialize to zero
            
        # FFN layers (mlp)
        # gate_proj, up_proj: dim -> hidden_dim
        # down_proj: hidden_dim -> dim
        for module in ["gate_proj", "up_proj"]:
            key_a = f"base_model.model.layers.{layer}.mlp.{module}.lora_A.weight"
            key_b = f"base_model.model.layers.{layer}.mlp.{module}.lora_B.weight"
            
            dummy_weights[key_a] = torch.randn(lora_rank, dim) * 0.01
            dummy_weights[key_b] = torch.zeros(hidden_dim, lora_rank)
            
        # down_proj is opposite direction
        key_a = f"base_model.model.layers.{layer}.mlp.down_proj.lora_A.weight"
        key_b = f"base_model.model.layers.{layer}.mlp.down_proj.lora_B.weight"
        
        dummy_weights[key_a] = torch.randn(lora_rank, hidden_dim) * 0.01
        dummy_weights[key_b] = torch.zeros(dim, lora_rank)

    # Save as safetensors
    from safetensors.torch import save_file

    adapter_path = os.path.join(output_dir, "adapter_model.safetensors")
    save_file(dummy_weights, adapter_path)

    print(f"✅ Created dummy LoRA adapter at: {output_dir}")
    print(
        f"   - adapter_model.safetensors: {os.path.getsize(adapter_path) / 1024:.2f} KB"
    )
    print("   - adapter_config.json")

    return output_dir


if __name__ == "__main__":
    # Create a small dummy LoRA for testing
    output_dir = create_dummy_lora(
        model_name="stories15M",
        output_dir="../lora_datasets/llama2c_lora_test",
        lora_rank=4,  # Small rank for testing
        lora_alpha=8,
    )
