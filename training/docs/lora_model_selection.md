# LoRA Learning Verification - Base Model Selection Guide

## Recommended Base Models for LoRA Training

### 1. **Llama 2 7B** (Most Recommended for Learning)
- **Size**: 7B parameters
- **Memory**: ~13GB (fp16)
- **Pros**: 
  - Excellent balance of performance and resource efficiency
  - Well-documented LoRA support
  - Active community and many examples
- **Use Case**: General purpose, instruction following

```bash
model_id = "meta-llama/Llama-2-7b-hf"
```

### 2. **Mistral 7B**
- **Size**: 7B parameters  
- **Memory**: ~13GB (fp16)
- **Pros**:
  - State-of-the-art performance for its size
  - Efficient attention mechanism
  - Good for code and reasoning tasks
- **Use Case**: Code generation, reasoning

```bash
model_id = "mistralai/Mistral-7B-v0.1"
```

### 3. **Japanese Models (If Japanese Focus)**

#### **rinna/japanese-gpt-neox-3.6b**
- **Size**: 3.6B parameters
- **Memory**: ~7GB (fp16)
- **Pros**: 
  - Smaller size, faster training
  - Japanese specialized
  - Good baseline for Japanese tasks

```bash
model_id = "rinna/japanese-gpt-neox-3.6b"
```

#### **cyberagent/open-calm-7b**
- **Size**: 7B parameters
- **Memory**: ~13GB (fp16)
- **Pros**:
  - Better Japanese performance
  - Commercial use friendly license

```bash
model_id = "cyberagent/open-calm-7b"
```

### 4. **For Limited Resources**

#### **Microsoft Phi-2**
- **Size**: 2.7B parameters
- **Memory**: ~5GB (fp16)
- **Pros**:
  - Very efficient
  - Good performance despite small size
  - Fast training

```bash
model_id = "microsoft/phi-2"
```

## LoRA Configuration Recommendations

### Standard Configuration
```python
lora_config = {
    "r": 16,                    # Rank
    "lora_alpha": 32,          # Scaling parameter
    "target_modules": ["q_proj", "v_proj"],  # Target layers
    "lora_dropout": 0.1,       # Dropout
    "bias": "none",            # Bias handling
}
```

### Memory-Efficient Configuration
```python
lora_config = {
    "r": 8,                    # Lower rank for less memory
    "lora_alpha": 16,
    "target_modules": ["q_proj", "v_proj"],
    "lora_dropout": 0.05,
    "bias": "none",
}
```

## Hardware Requirements

### Minimum Requirements
- **GPU**: RTX 3060 (12GB) or better
- **RAM**: 32GB system memory
- **Storage**: 50GB free space

### Recommended Setup
- **GPU**: RTX 3090/4090 (24GB) or A100
- **RAM**: 64GB system memory
- **Storage**: 100GB+ SSD

## Quick Start Example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# Load base model
model_id = "meta-llama/Llama-2-7b-hf"  # or your chosen model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_8bit=True,  # Use 8-bit quantization to save memory
    device_map="auto"
)

# Configure LoRA
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
```

## Recommendation Summary

For your first LoRA training verification, I recommend:

1. **Llama 2 7B** - Best overall choice for learning
2. **Mistral 7B** - If focusing on code/reasoning tasks  
3. **Phi-2** - If GPU memory is limited (<12GB)
4. **rinna/japanese-gpt-neox-3.6b** - For Japanese language tasks

Start with the standard LoRA configuration (r=16) and adjust based on your GPU memory and training results.