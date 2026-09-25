import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from peft import PeftModel


# ==========================================
# Configuration
# ==========================================

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

ADAPTER_PATH = "./models/django-lora"


# ==========================================
# Device
# ==========================================

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Using device:", device)


# ==========================================
# Load tokenizer
# ==========================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL
)


# ==========================================
# Load base model
# ==========================================

print("Loading base model...")

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
)


# ==========================================
# Load your LoRA adapter
# ==========================================

print("Loading trained LoRA adapter...")

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
)

model.to(device)

model.eval()


# ==========================================
# Ask question
# ==========================================

question = "What is select_related?"


prompt = f"""### Instruction:
{question}

### Response:
"""


inputs = tokenizer(
    prompt,
    return_tensors="pt"
)

inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}


# ==========================================
# Generate response
# ==========================================

with torch.no_grad():

    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7,
        do_sample=True,
    )


# ==========================================
# Decode
# ==========================================

response = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True,
)


print("\n==============================")
print("QUESTION")
print("==============================")

print(question)


print("\n==============================")
print("MODEL RESPONSE")
print("==============================")

print(response)