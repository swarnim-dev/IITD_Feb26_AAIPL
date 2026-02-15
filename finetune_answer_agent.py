"""
Fine-tune Answer Agent on Question-Answer Pairs
Uses LoRA for efficient fine-tuning
"""

import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import sys

print("="*60)
print("ANSWER AGENT FINE-TUNING")
print("="*60)

# Load question-answer pairs
print("\n1. Loading training data...")
try:
    with open('outputs/all_answers.json', 'r') as f:
        qa_pairs = json.load(f)
    
    # Filter to only correct answers
    qa_pairs = [qa for qa in qa_pairs if qa.get('is_correct', False)]
    print(f"✓ Loaded {len(qa_pairs)} correct Q&A pairs for training")
    
    if len(qa_pairs) < 10:
        print("✗ Need at least 10 correct answers for fine-tuning")
        print("Run: python answer_all_questions.py first")
        sys.exit(1)
        
except FileNotFoundError:
    print("✗ Error: outputs/all_answers.json not found")
    print("Run: python answer_all_questions.py first")
    sys.exit(1)

# Prepare training data
print("\n2. Preparing training dataset...")

def format_training_example(qa):
    """Format Q&A pair for training."""
    choices_text = "\n".join(qa['choices'])
    
    prompt = f"""Question: {qa['question']}

Choices:
{choices_text}

Instructions:
1. Analyze each choice carefully
2. Use logical reasoning
3. State your answer clearly

Provide your answer in this format:
Answer: [A/B/C/D]
Reasoning: [Brief explanation]"""
    
    # Target answer
    answer = qa['generated_answer']
    
    return {
        'prompt': prompt,
        'completion': answer
    }

# Format all examples
training_data = [format_training_example(qa) for qa in qa_pairs]

# Split into train/val (90/10)
split_idx = int(len(training_data) * 0.9)
train_data = training_data[:split_idx]
val_data = training_data[split_idx:]

print(f"✓ Training samples: {len(train_data)}")
print(f"✓ Validation samples: {len(val_data)}")

# Load base model
print("\n3. Loading base model...")
model_path = "hf_models/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

print(f"✓ Model loaded on device: {model.device}")

# Configure LoRA
print("\n4. Configuring LoRA for efficient fine-tuning...")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                    # LoRA rank
    lora_alpha=32,           # LoRA scaling
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Attention layers
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Prepare datasets
def tokenize_function(examples):
    """Tokenize training examples."""
    # Format as chat
    full_texts = []
    for prompt, completion in zip(examples['prompt'], examples['completion']):
        messages = [
            {"role": "system", "content": "You are an expert at logical reasoning."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        full_texts.append(text)
    
    # Tokenize
    tokenized = tokenizer(
        full_texts,
        truncation=True,
        max_length=512,
        padding="max_length",
    )
    
    # Labels = input_ids for causal LM
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

# Create HF datasets
train_dataset = Dataset.from_dict({
    'prompt': [d['prompt'] for d in train_data],
    'completion': [d['completion'] for d in train_data]
})

val_dataset = Dataset.from_dict({
    'prompt': [d['prompt'] for d in val_data],
    'completion': [d['completion'] for d in val_data]
})

# Tokenize
print("\n5. Tokenizing datasets...")
train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=['prompt', 'completion'])
val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=['prompt', 'completion'])

print(f"✓ Training samples tokenized: {len(train_dataset)}")
print(f"✓ Validation samples tokenized: {len(val_dataset)}")

# Training arguments
print("\n6. Configuring training...")

training_args = TrainingArguments(
    output_dir="./answer_agent_finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_steps=10,
    logging_steps=5,
    eval_strategy="steps",
    eval_steps=10,
    save_steps=20,
    save_total_limit=2,
    fp16=False,
    bf16=True,
    optim="adamw_torch",
    report_to="none",
    load_best_model_at_end=True,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train!
print("\n7. Starting fine-tuning...")
print("="*60)
trainer.train()

# Save
print("\n8. Saving fine-tuned model...")
model.save_pretrained("./answer_agent_finetuned/final")
tokenizer.save_pretrained("./answer_agent_finetuned/final")

print("\n" + "="*60)
print("FINE-TUNING COMPLETE!")
print("="*60)
print(f"Model saved to: ./answer_agent_finetuned/final")
print("\nTo use the fine-tuned model, update model_path in answer_model.py")
print("="*60)

