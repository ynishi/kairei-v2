# CPU-Friendly Models for LoRA Training

## Best Models for CPU Verification

### 1. **TinyLlama-1.1B** (Most Recommended)
- **Size**: 1.1B parameters
- **Memory**: ~2GB
- **Speed**: 10-15 tokens/sec on modern CPU
```python
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

### 2. **Microsoft Phi-1.5**
- **Size**: 1.3B parameters  
- **Memory**: ~2.5GB
- **Speed**: 8-12 tokens/sec
```python
model_id = "microsoft/phi-1_5"
```

### 3. **OPT-125M** (Ultra Light)
- **Size**: 125M parameters
- **Memory**: ~250MB
- **Speed**: 50+ tokens/sec
```python
model_id = "facebook/opt-125m"
```

### 4. **BLOOM-560M**
- **Size**: 560M parameters
- **Memory**: ~1GB
- **Speed**: 20-30 tokens/sec
```python
model_id = "bigscience/bloom-560m"
```

### 5. **Japanese Small Models**

#### **rinna/japanese-gpt2-small**
- **Size**: 110M parameters
- **Memory**: ~220MB
- **Speed**: Very fast on CPU
```python
model_id = "rinna/japanese-gpt2-small"
```

## CPU Optimization Tips

### 1. Use INT8 Quantization
```python
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.float32,  # CPU prefers float32
    low_cpu_mem_usage=True
)
```

### 2. LoRA Config for CPU
```python
from peft import LoraConfig

cpu_lora_config = LoraConfig(
    r=4,                    # Very low rank for CPU
    lora_alpha=8,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none"
)
```

### 3. Training Configuration
```python
training_args = {
    "per_device_train_batch_size": 1,  # Small batch for CPU
    "gradient_accumulation_steps": 4,   # Simulate larger batch
    "num_train_epochs": 3,
    "learning_rate": 5e-4,
    "fp16": False,                      # CPU doesn't support fp16
    "logging_steps": 10,
    "save_strategy": "epoch",
    "dataloader_num_workers": 2,        # Limited for CPU
}
```

## Quick CPU Test Script

```python
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# Load small model
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print(f"Loading {model_id}...")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
)

# Apply LoRA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=4,
    lora_alpha=8,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05
)

model = get_peft_model(model, peft_config)
print(f"Trainable params: {model.print_trainable_parameters()}")

# Test inference speed
text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors="pt")

start = time.time()
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50)
end = time.time()

generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\nGenerated: {generated}")
print(f"Time: {end-start:.2f}s")
```

## Recommended Setup for CPU Testing

1. **First Choice**: TinyLlama-1.1B
   - Best balance of capability and speed
   - Can handle real tasks despite small size

2. **Fastest Option**: OPT-125M or GPT2-small
   - For quick iteration and testing
   - Very limited capabilities

3. **Japanese**: rinna/japanese-gpt2-small
   - Fast and Japanese-optimized

## Expected Training Times (CPU)

| Model | 1000 samples | 10000 samples |
|-------|--------------|---------------|
| OPT-125M | ~30 min | ~5 hours |
| TinyLlama-1.1B | ~2 hours | ~20 hours |
| Phi-1.5 | ~3 hours | ~30 hours |

Note: Use small datasets (<1000 samples) for CPU verification!