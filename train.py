from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer


# -----------------------------------
# 1. Model
# -----------------------------------

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)


# -----------------------------------
# 2. Dataset
# -----------------------------------

dataset = load_dataset(
    "json",
    data_files="data/train.jsonl"
)

print(dataset)


# -----------------------------------
# 3. Format dataset
# -----------------------------------

def format_example(example):
    return (
        f"### Instruction:\n"
        f"{example['instruction']}\n\n"
        f"### Response:\n"
        f"{example['response']}"
    )


# -----------------------------------
# 4. LoRA configuration
# -----------------------------------

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
    task_type="CAUSAL_LM",
)


# -----------------------------------
# 5. Training configuration
# -----------------------------------

training_args = TrainingArguments(
    output_dir="./models/django-lora",

    num_train_epochs=3,

    per_device_train_batch_size=1,

    gradient_accumulation_steps=4,

    learning_rate=2e-4,

    logging_steps=10,

    save_steps=100,

    save_total_limit=2,

    report_to="none",
)


# -----------------------------------
# 6. Trainer
# -----------------------------------

trainer = SFTTrainer(
    model=model,

    train_dataset=dataset["train"],

    peft_config=lora_config,

    formatting_func=format_example,

    args=training_args,
)


# -----------------------------------
# 7. Start training
# -----------------------------------

print("Starting training...")

trainer.train()


# -----------------------------------
# 8. Save model
# -----------------------------------

trainer.save_model("./models/django-lora")

tokenizer.save_pretrained("./models/django-lora")

print("Training completed!")