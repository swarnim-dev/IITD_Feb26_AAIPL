"""
Answer Agent Tuning Interface
Test different configurations to optimize accuracy vs speed
"""

import sys
import json
sys.path.insert(0, 'agents')

# Configuration presets
CONFIGS = {
    "1_fastest": {
        "name": "Fastest (Current)",
        "max_new_tokens": 40,
        "do_sample": False,
        "temperature": None,
        "prompt_style": "brief"
    },
    "2_balanced": {
        "name": "Balanced Speed/Accuracy",
        "max_new_tokens": 80,
        "do_sample": True,
        "temperature": 0.5,
        "top_p": 0.85,
        "top_k": 30,
        "prompt_style": "detailed"
    },
    "3_accurate": {
        "name": "Maximum Accuracy",
        "max_new_tokens": 150,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "prompt_style": "detailed"
    },
    "4_smart": {
        "name": "Smart Greedy (Fast + Better Prompt)",
        "max_new_tokens": 100,
        "do_sample": False,
        "temperature": None,
        "prompt_style": "reasoning"
    }
}

def build_prompt(question, choices, style):
    """Build prompt based on style."""
    choices_text = "\n".join(choices)
    
    if style == "brief":
        return f"""Question: {question}

Choices:
{choices_text}

Format: Answer: [A/B/C/D] | Reason: [5 words max]"""
    
    elif style == "detailed":
        return f"""Question: {question}

Choices:
{choices_text}

Instructions:
1. Analyze each choice carefully
2. Use logical reasoning
3. State your answer clearly

Provide your answer in this format:
Answer: [A/B/C/D]
Reasoning: [Brief explanation]"""
    
    elif style == "reasoning":
        return f"""You must solve this logical reasoning question step by step.

Question: {question}

Choices:
{choices_text}

Think through this systematically:
1. What is being asked?
2. What logical rules apply?
3. Which choice follows from the rules?

Answer: [A/B/C/D]
Reasoning: [One clear sentence explaining why]"""

def test_config(config_name, config, questions):
    """Test a configuration on sample questions."""
    
    print(f"\n{'='*60}")
    print(f"Testing: {config['name']}")
    print(f"{'='*60}")
    
    # Import here to avoid loading model multiple times
    from answer_model import AAgent
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import time
    
    # Load model
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
    model.eval()
    
    results = []
    total_time = 0
    
    for i, q in enumerate(questions, 1):
        # Build prompt based on style
        prompt = build_prompt(q['question'], q['choices'], config['prompt_style'])
        
        # Format messages
        messages = [
            {"role": "system", "content": "You are an expert at logical reasoning."},
            {"role": "user", "content": prompt}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        # Build generation config
        gen_config = {
            "max_new_tokens": config['max_new_tokens'],
            "min_new_tokens": 8,
            "do_sample": config['do_sample'],
            "repetition_penalty": 1.2,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        
        if config['do_sample']:
            gen_config['temperature'] = config['temperature']
            gen_config['top_p'] = config['top_p']
            gen_config['top_k'] = config['top_k']
        
        # Generate
        start = time.time()
        with torch.no_grad():
            generated_ids = model.generate(**model_inputs, **gen_config)
        gen_time = time.time() - start
        total_time += gen_time
        
        # Decode
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Check correctness
        is_correct = q['answer'] in answer.upper()
        
        results.append({
            'correct': is_correct,
            'time': gen_time,
            'answer': answer
        })
        
        status = "✓" if is_correct else "✗"
        time_status = "✓" if gen_time <= 9.0 else "⚠"
        print(f"  Q{i}: {status} | Time: {gen_time:5.2f}s {time_status}")
    
    # Calculate metrics
    correct = sum(1 for r in results if r['correct'])
    within_time = sum(1 for r in results if r['time'] <= 9.0)
    avg_time = total_time / len(results)
    
    print(f"\nResults:")
    print(f"  Accuracy:     {correct}/{len(results)} ({correct/len(results)*100:.1f}%)")
    print(f"  Avg Time:     {avg_time:.2f}s")
    print(f"  Within 9s:    {within_time}/{len(results)} ({within_time/len(results)*100:.1f}%)")
    
    # Clean up
    del model
    del tokenizer
    torch.cuda.empty_cache()
    
    return {
        'config_name': config_name,
        'name': config['name'],
        'accuracy': correct/len(results)*100,
        'avg_time': avg_time,
        'within_time_pct': within_time/len(results)*100
    }

# Main
print("="*60)
print("ANSWER AGENT TUNING")
print("="*60)

# Load test questions
with open('outputs/filtered_questions.json', 'r') as f:
    all_questions = json.load(f)

# Use first 10 questions for tuning
test_questions = all_questions[:10]
print(f"\nUsing {len(test_questions)} questions for tuning")

# Test each configuration
summary = []

for config_name, config in CONFIGS.items():
    result = test_config(config_name, config, test_questions)
    summary.append(result)

# Print comparison
print(f"\n{'='*60}")
print("CONFIGURATION COMPARISON")
print(f"{'='*60}")
print(f"{'Config':<30} {'Accuracy':>10} {'Avg Time':>10} {'Within 9s':>12}")
print("-"*60)

for s in summary:
    print(f"{s['name']:<30} {s['accuracy']:>9.1f}% {s['avg_time']:>9.2f}s {s['within_time_pct']:>11.1f}%")

# Find best
best_acc = max(summary, key=lambda x: x['accuracy'])
best_speed = min(summary, key=lambda x: x['avg_time'])

print(f"\n{'='*60}")
print("RECOMMENDATIONS")
print(f"{'='*60}")
print(f"Best Accuracy:  {best_acc['name']} ({best_acc['accuracy']:.1f}%)")
print(f"Fastest:        {best_speed['name']} ({best_speed['avg_time']:.2f}s)")

# Save results
with open('outputs/tuning_results.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\n✓ Tuning results saved to outputs/tuning_results.json")

