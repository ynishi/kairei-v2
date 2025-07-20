#!/usr/bin/env python3
"""Download TinyLlama model from HuggingFace"""

from huggingface_hub import snapshot_download
import os

# Create models directory if it doesn't exist
os.makedirs("models/tinyllama", exist_ok=True)

print("🚀 Downloading TinyLlama-1.1B-Chat-v1.0...")
print("This may take a while depending on your internet connection...")

# Download the model
snapshot_download(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    local_dir="models/tinyllama",
    allow_patterns=[
        "*.safetensors",
        "*.json",
        "tokenizer.model",
        "config.json",
    ],
    ignore_patterns=[
        "*.bin",
        "*.h5",
        "*.msgpack",
    ]
)

print("✅ Download complete!")
print("📁 Model saved to: models/tinyllama/")