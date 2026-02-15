import sys, json, time
sys.path.insert(0, 'agents')
from answer_model import AAgent

INPUT_FILE  = "outputs/filtered_questions.json"
OUTPUT_FILE = "outputs/answers_800_1000.json"

with open(INPUT_FILE) as f:
    questions = json.load(f)
print(f"Loaded {len(questions)}")

MODEL_PATH = "hf_models/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
agent = AAgent(MODEL_PATH)

results = []
run_start = time.time()

for i, q in enumerate(questions, 1):
    choices_text = "\n".join(q['choices'])
    prompt = f"Question: {q['question']}\n\nChoices:\n{choices_text}\n\nAnswer:"
    
    answer, gen_time, _ = agent.generate_response(
        prompt, "You are an expert at logical reasoning."
    )
    
    is_correct   = q['answer'] in answer.upper()
    within_limit = gen_time <= 9.0
    
    print(f"  Q{0+i:4d}: {'✓' if is_correct else '✗'} (exp {q['answer']}) | "
          f"{gen_time:5.2f}s {'✓' if within_limit else '⚠ OVER'} | {q['topic'][:35]}")
    
    results.append({
        "question_number": 0+i,
        "topic":           q['topic'],
        "question":        q['question'],
        "choices":         q['choices'],
        "expected_answer": q['answer'],
        "generated_answer": answer,
        "generation_time": round(gen_time, 3),
        "is_correct":      is_correct,
        "within_time_limit": within_limit,
    })

total_time = time.time() - run_start
correct     = sum(1 for r in results if r['is_correct'])
within_time = sum(1 for r in results if r['within_time_limit'])
avg_time    = sum(r['generation_time'] for r in results) / len(results)

print(f"\n{'='*50}")
print(f"Total questions  : {len(results)}")
print(f"Correct          : {correct}/{len(results)} ({correct/len(results)*100:.1f}%)")
print(f"Within 9s limit  : {within_time}/{len(results)} ({within_time/len(results)*100:.1f}%)")
print(f"Avg time         : {avg_time:.2f}s")
print(f"Total runtime    : {total_time/60:.1f} min")
from collections import defaultdict
topic_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
for r in results:
    topic_stats[r['topic']]['total']   += 1
    topic_stats[r['topic']]['correct'] += int(r['is_correct'])

print(f"\n{'='*50}")
print("ACCURACY BY TOPIC")
print(f"{'='*50}")
for topic, s in sorted(topic_stats.items()):
    acc = s['correct'] / s['total'] * 100
    print(f"  {topic:<45} {s['correct']:3d}/{s['total']:3d} ({acc:5.1f}%)")

with open(OUTPUT_FILE, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n✓ Results saved to {OUTPUT_FILE}")
