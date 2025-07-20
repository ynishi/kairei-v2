# TinyLlama Training Data Format Guide

## TinyLlama Overview
TinyLlama uses the Llama 2 chat template format, which makes it compatible with standard instruction-following datasets.

## Supported Data Formats

### 1. **Alpaca Format** (Recommended for Instruction-Following)
```json
[
  {
    "instruction": "What is the capital of France?",
    "input": "",
    "output": "The capital of France is Paris."
  },
  {
    "instruction": "Translate the following text to Japanese",
    "input": "Hello, how are you?",
    "output": "こんにちは、元気ですか？"
  }
]
```

### 2. **Chat Format** (For Conversational Tasks)
```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "Can you help me write a Python function?"
      },
      {
        "from": "assistant", 
        "value": "Of course! What kind of function would you like to create?"
      },
      {
        "from": "human",
        "value": "A function to calculate fibonacci numbers"
      },
      {
        "from": "assistant",
        "value": "Here's a Python function to calculate Fibonacci numbers:\n\n```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n```"
      }
    ]
  }
]
```

### 3. **Simple Text Format** (For Completion Tasks)
```json
[
  {
    "text": "The quick brown fox jumps over the lazy dog. This sentence contains all letters of the alphabet."
  },
  {
    "text": "Machine learning is a subset of artificial intelligence that focuses on learning from data."
  }
]
```

## TinyLlama Chat Template

TinyLlama uses this specific template for chat:
```
<|system|>
You are a helpful assistant.
<|user|>
{user_message}
<|assistant|>
{assistant_response}
```

## Creating Your Dataset

### Step 1: Choose Your Task Type

1. **Instruction-Following**: Use Alpaca format
2. **Conversational**: Use Chat format
3. **Text Completion**: Use Simple text format

### Step 2: Prepare Your Data

Create a JSON file with your data. Here's a complete example:

```python
import json

# Example: Creating a coding assistant dataset
dataset = [
    {
        "instruction": "Write a Python function to reverse a string",
        "input": "",
        "output": "def reverse_string(s):\n    return s[::-1]"
    },
    {
        "instruction": "Explain what this code does",
        "input": "x = [i**2 for i in range(10)]",
        "output": "This code creates a list comprehension that generates a list of squares. It calculates the square of each number from 0 to 9 and stores them in a list called 'x'. The result will be: [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]"
    },
    {
        "instruction": "Fix this Python code",
        "input": "def add(a, b)\n    return a + b",
        "output": "The code is missing a colon after the function definition. Here's the corrected version:\n\ndef add(a, b):\n    return a + b"
    }
]

# Save to file
with open('coding_assistant_dataset.json', 'w', encoding='utf-8') as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)
```

### Step 3: Dataset Size Recommendations

For CPU training with TinyLlama:
- **Minimum**: 50-100 examples (for testing)
- **Recommended**: 500-1000 examples
- **Maximum**: 5000 examples (longer training time)

### Step 4: Data Quality Tips

1. **Diversity**: Include various types of instructions
2. **Length**: Keep responses concise (50-200 tokens)
3. **Consistency**: Maintain consistent formatting
4. **Quality**: Ensure correct and helpful responses

## Example Dataset Creation Script

```python
from pathlib import Path
import json
import random

def create_tinyllama_dataset(output_path="./data/tinyllama_train.json"):
    """Create a sample dataset for TinyLlama fine-tuning"""
    
    # Different categories of examples
    coding_examples = [
        {
            "instruction": "Write a Python function to check if a number is prime",
            "input": "",
            "output": "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True"
        },
        {
            "instruction": "What is a list comprehension in Python?",
            "input": "",
            "output": "A list comprehension is a concise way to create lists in Python. It consists of brackets containing an expression followed by a for clause. Example: [x**2 for x in range(10)] creates a list of squares from 0 to 81."
        }
    ]
    
    general_knowledge = [
        {
            "instruction": "Explain photosynthesis in simple terms",
            "input": "",
            "output": "Photosynthesis is the process plants use to make food from sunlight. They take in carbon dioxide from the air and water from the soil, then use sunlight to convert these into glucose (sugar) for energy and oxygen as a byproduct."
        }
    ]
    
    translation_examples = [
        {
            "instruction": "Translate to Japanese",
            "input": "Good morning",
            "output": "おはようございます (Ohayou gozaimasu)"
        }
    ]
    
    # Combine all examples
    all_examples = coding_examples + general_knowledge + translation_examples
    
    # Shuffle for variety
    random.shuffle(all_examples)
    
    # Save dataset
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_examples, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Created dataset with {len(all_examples)} examples")
    print(f"📁 Saved to: {output_path}")
    
    return all_examples

# Create the dataset
if __name__ == "__main__":
    create_tinyllama_dataset()
```

## Validation Data

Always create a separate validation set (10-20% of your data):

```python
# Split into train/validation
full_dataset = create_tinyllama_dataset()
split_idx = int(len(full_dataset) * 0.8)

train_data = full_dataset[:split_idx]
val_data = full_dataset[split_idx:]

# Save separately
with open('./data/train.json', 'w') as f:
    json.dump(train_data, f, indent=2)
    
with open('./data/val.json', 'w') as f:
    json.dump(val_data, f, indent=2)
```

## Data Format Validation

Use this script to validate your dataset format:

```python
def validate_dataset(file_path):
    """Validate if dataset is in correct format"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        return False, "Dataset must be a list"
    
    for i, item in enumerate(data):
        if 'instruction' in item:
            # Alpaca format
            if 'output' not in item:
                return False, f"Item {i} missing 'output' field"
        elif 'conversations' in item:
            # Chat format
            if not isinstance(item['conversations'], list):
                return False, f"Item {i} conversations must be a list"
        elif 'text' in item:
            # Text format
            if not isinstance(item['text'], str):
                return False, f"Item {i} text must be a string"
        else:
            return False, f"Item {i} has unknown format"
    
    return True, f"✅ Valid dataset with {len(data)} examples"

# Validate your dataset
valid, message = validate_dataset('./data/train.json')
print(message)
```

## Next Steps

1. Create your dataset using one of the formats above
2. Validate the format using the validation script
3. Split into train/validation sets
4. Ready for LoRA fine-tuning!

Remember: For CPU training, keep datasets small (100-1000 examples) for reasonable training times.