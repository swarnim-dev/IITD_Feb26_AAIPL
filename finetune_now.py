"""
Fine-tune Answer Agent from questions file
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import sys
import os

print("="*60)
print("FINE-TUNING ANSWER AGENT")
print("="*60)

# Load questions from outputs directory
questions_file = "outputs/filtered_questions.json"

if not os.path.exists(questions_file):
    print(f"\n✗ File not found: {questions_file}")
    sys.exit(1)

print(f"\nLoading {questions_file}...")
with open(questions_file, 'r') as f:
    questions = json.load(f)

print(f"✓ Loaded {len(questions)} questions")

if len(questions) < 3:
    print("✗ Need at least 3 questions")
    sys.exit(1)

# Create training data
print("\nPreparing training data...")
training_data = []

for q in questions:
    choices_text = "\n".join(q['choices'])
    
    prompt = f"""Question: {q['question']}

Choices:
{choices_text}

Instructions:
1. Analyze each choice carefully
2. Use logical reasoning
3. State your answer clearly

Provide your answer in this format:
Answer: [A/B/C/D]
Reasoning: [Brief explanation]"""
    
    completion = f"Answer: {q['answer']}\nReasoning: {q['explanation']}"
    
    training_data.append({'prompt': prompt, 'completion': completion})

# Augment for better learning
original_count = len(training_data)
training_data = training_data * 3  # Repeat 3x
print(f"✓ {original_count} examples → {len(training_data)} (augmented 3x)")

# Load model
print("\nLoading Qwen2.5-14B model...")
model_path = "hf_models/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

print(f"✓ Model loaded on {model.device}")

# LoRA configuration
print("\nApplying LoRA (efficient fine-tuning)...")
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=32,
    lora_alpha=64,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
)

model = get_peft_model(model, lora_config)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"✓ Trainable: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")

# Tokenize
def tokenize_function(examples):
    full_texts = []
    for prompt, completion in zip(examples['prompt'], examples['completion']):
        messages = [
            {"role": "system", "content": "You are an expert at logical reasoning."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        full_texts.append(text)
    
    tokenized = tokenizer(full_texts, truncation=True, max_length=512, padding="max_length")
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

train_dataset = Dataset.from_dict({
    'prompt': [d['prompt'] for d in training_data],
    'completion': [d['completion'] for d in training_data]
})

print("\nTokenizing dataset...")
train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=['prompt', 'completion'])
print(f"✓ Tokenized {len(train_dataset)} samples")

# Training configuration
print("\nConfiguring training...")
print("Epochs: 3")
print("Estimated time: 30-45 minutes")

training_args = TrainingArguments(
    output_dir="./answer_agent_finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_steps=20,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    bf16=True,
    optim="adamw_torch",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Start training
print("\n" + "="*60)
print("STARTING FINE-TUNING...")
print("="*60 + "\n")

trainer.train()

# Save
print("\nSaving fine-tuned model...")
model.save_pretrained("./answer_agent_finetuned/final")
tokenizer.save_pretrained("./answer_agent_finetuned/final")

print("\n" + "="*60)
print("✅ FINE-TUNING COMPLETE!")
print("="*60)
print(f"✓ Model saved to: ./answer_agent_finetuned/final")
print(f"✓ Trained on {len(questions)} questions")
print("\nTo use the fine-tuned model:")
print("  Update model_path in agents/answer_model.py to:")
print('  "./answer_agent_finetuned/final"')
print("="*60)

