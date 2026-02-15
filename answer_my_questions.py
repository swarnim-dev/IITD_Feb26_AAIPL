import sys
import json
sys.path.insert(0, 'agents')
from answer_model import AAgent

# Load YOUR generated questions
print("Loading questions...")
with open('outputs/filtered_questions.json', 'r') as f:
    questions = json.load(f)

print(f"Found {len(questions)} questions")

# Initialize agent
model_path = "hf_models/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
agent = AAgent(model_path)

# Answer first 5 questions
results = []
for i, q in enumerate(questions[:5], 1):
    print(f"\n{'='*60}")
    print(f"Question {i}/{5}")
    print(f"Topic: {q['topic']}")
    print(f"Question: {q['question'][:100]}...")
    print(f"Expected Answer: {q['answer']}")
    
    # Build prompt
    choices_text = "\n".join(q['choices'])
    prompt = f"""Question: {q['question']}

Choices:
{choices_text}

Answer:"""
    
    # Generate answer
    answer, gen_time, _ = agent.generate_response(
        prompt, 
        "You are an expert at logical reasoning."
    )
    
    # Check if correct
    answer_upper = answer.upper()
    is_correct = q['answer'] in answer_upper
    
    print(f"\nGenerated Answer: {answer}")
    print(f"Time: {gen_time:.2f}s {'✅' if gen_time <= 9.0 else '❌'}")
    print(f"Correct: {'✅' if is_correct else '❌'}")
    
    results.append({
        "question": q['question'],
        "expected": q['answer'],
        "generated": answer,
        "time": gen_time,
        "correct": is_correct
    })

# Summary
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
correct_count = sum(1 for r in results if r['correct'])
within_time = sum(1 for r in results if r['time'] <= 9.0)
avg_time = sum(r['time'] for r in results) / len(results)

print(f"Questions answered: {len(results)}")
print(f"Correct: {correct_count}/{len(results)} ({correct_count/len(results)*100:.1f}%)")
print(f"Within time limit: {within_time}/{len(results)} ({within_time/len(results)*100:.1f}%)")
print(f"Average time: {avg_time:.2f}s")

# Save results
with open('outputs/answer_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to outputs/answer_results.json")

agent.print_statistics()
