#!/usr/bin/env python3
"""
Simple test script to verify Manual LoRA processor
"""

import json
import sys
import subprocess
import os

def test_manual_lora():
    """Test the manual LoRA processor"""
    
    # Configuration for test
    config = {
        "model_path": "path/to/tinyllama.safetensors",  # Update this
        "tokenizer_path": "path/to/tokenizer.json",     # Update this
        "lora_path": None,  # Optional: path to LoRA weights
        "processor": "manual_llama2_lora",
        "config": {
            "dim": 768,
            "hidden_dim": 2048,
            "n_layers": 12,
            "n_heads": 12,
            "n_kv_heads": 12,
            "vocab_size": 32000,
            "seq_len": 2048,
            "norm_eps": 1e-5
        }
    }
    
    print("🧪 Testing Manual LoRA Processor...")
    print(f"   Config: {json.dumps(config['config'], indent=2)}")
    
    # For now, just print what we would do
    print("\n📝 To use this processor in Rust:")
    print("   1. Update model_path and tokenizer_path")
    print("   2. Create ManualLlama2LoraProcessor with the config")
    print("   3. Call process() with a test message")
    
    print("\n✅ Test configuration ready!")

if __name__ == "__main__":
    test_manual_lora()