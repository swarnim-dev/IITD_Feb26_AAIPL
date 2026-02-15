import sys
import json
import time
sys.path.insert(0, 'agents')
from answer_model import AAgent

# Load YOUR generated questions
print("="*60)
print("ANSWERING ALL GENERATED QUESTIONS")
print("="*60)

try:
    with open('outputs/filtered_questions.json', 'r') as f:
        questions = json.load(f)
    print(f"✓ Loaded {len(questions)} questions from filtered_questions.json")
except FileNotFoundError:
    print("✗ Error: outputs/filtered_questions.json not found")
    print("Please generate questions first using: python -m agents.question_agent")
    sys.exit(1)

# Initialize agent
model_path = "hf_models/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
print(f"\nInitializing Answer Agent...")
agent = AAgent(model_path)

# Answer ALL questions
results = []
start_time = time.time()

print(f"\n{'='*60}")
print(f"Processing {len(questions)} questions...")
print(f"{'='*60}\n")

for i, q in enumerate(questions, 1):
    # Progress indicator
    if i % 10 == 0 or i == 1:
        print(f"Progress: {i}/{len(questions)} ({i/len(questions)*100:.1f}%)")
    
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
    
    # Show individual result
    status = "✓" if is_correct else "✗"
    time_status = "✓" if gen_time <= 9.0 else "⚠"
    print(f"  Q{i:4d}: {status} {q['answer']} | Time: {gen_time:5.2f}s {time_status} | Topic: {q['topic'][:30]}")
    
    results.append({
        "question_id": i,
        "topic": q['topic'],
        "question": q['question'],
        "choices": q['choices'],
        "expected_answer": q['answer'],
        "generated_answer": answer,
        "generation_time": gen_time,
        "is_correct": is_correct,
        "within_time_limit": gen_time <= 9.0
    })

total_time = time.time() - start_time

# Calculate statistics
print(f"\n{'='*60}")
print("FINAL RESULTS")
print(f"{'='*60}")

correct_count = sum(1 for r in results if r['is_correct'])
within_time = sum(1 for r in results if r['within_time_limit'])
avg_time = sum(r['generation_time'] for r in results) / len(results)

print(f"Total questions:          {len(results)}")
print(f"Correct answers:          {correct_count} ({correct_count/len(results)*100:.1f}%)")
print(f"Within time limit (≤9s):  {within_time} ({within_time/len(results)*100:.1f}%)")
print(f"Average time per Q:       {avg_time:.2f}s")
print(f"Total processing time:    {total_time/60:.1f} minutes")

# Topic breakdown
print(f"\n{'='*60}")
print("ACCURACY BY TOPIC")
print(f"{'='*60}")

topics = {}
for r in results:
    topic = r['topic']
    if topic not in topics:
        topics[topic] = {'total': 0, 'correct': 0}
    topics[topic]['total'] += 1
    if r['is_correct']:
        topics[topic]['correct'] += 1

for topic, stats in sorted(topics.items()):
    acc = stats['correct'] / stats['total'] * 100
    print(f"{topic[:40]:40s}: {stats['correct']:3d}/{stats['total']:3d} ({acc:5.1f}%)")

# Save results
output_file = 'outputs/all_answers.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n✓ Detailed results saved to {output_file}")

# Save summary
summary = {
    "total_questions": len(results),
    "correct_answers": correct_count,
    "accuracy_percentage": round(correct_count/len(results)*100, 2),
    "within_time_limit": within_time,
    "time_compliance_percentage": round(within_time/len(results)*100, 2),
    "average_time_seconds": round(avg_time, 2),
    "total_processing_minutes": round(total_time/60, 2),
    "topic_breakdown": {
        topic: {
            "total": stats['total'],
            "correct": stats['correct'],
            "accuracy": round(stats['correct']/stats['total']*100, 2)
        }
        for topic, stats in topics.items()
    }
}

with open('outputs/answer_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"✓ Summary saved to outputs/answer_summary.json")

# Print agent statistics
agent.print_statistics()

print(f"\n{'='*60}")
print("DONE!")
print(f"{'='*60}")
