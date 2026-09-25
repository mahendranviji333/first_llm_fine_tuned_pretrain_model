import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = "./models/django-lora"

# --------------------------------------------------
# Device
# --------------------------------------------------

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Using device:", device)

# --------------------------------------------------
# Load tokenizer
# --------------------------------------------------

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# --------------------------------------------------
# Load BASE model
# --------------------------------------------------

print("Loading base model...")

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
)

base_model.to(device)
base_model.eval()

# --------------------------------------------------
# Load FINE-TUNED model
# --------------------------------------------------

print("Loading fine-tuned model...")

finetuned_base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
)

finetuned_base.to(device)

finetuned_model = PeftModel.from_pretrained(
    finetuned_base,
    ADAPTER_PATH,
)

finetuned_model.to(device)
finetuned_model.eval()

# --------------------------------------------------
# Function to generate answer
# --------------------------------------------------

def generate_answer(model, question):

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

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
        )

    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )

    return response


# --------------------------------------------------
# Questions to test
# --------------------------------------------------

questions = [
    "What is Django ORM?",
    "What is select_related in Django?",
    "What is prefetch_related in Django?",
    "What is Django middleware?",
    "What is a DRF serializer?",
    "What is an N+1 query problem?",
]


# --------------------------------------------------
# Compare
# --------------------------------------------------

for i, question in enumerate(questions, start=1):

    print("\n")
    print("=" * 80)
    print(f"QUESTION {i}")
    print("=" * 80)

    print(question)

    # BASE MODEL
    base_answer = generate_answer(
        base_model,
        question
    )

    print("\n--- BASE MODEL ---")
    print(base_answer)

    # FINE-TUNED MODEL
    finetuned_answer = generate_answer(
        finetuned_model,
        question
    )

    print("\n--- FINE-TUNED MODEL ---")
    print(finetuned_answer)

    print("\n")