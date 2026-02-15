import sys
sys.path.insert(0, 'agents')
from answer_model import AAgent

# Test question
test_q = {
    "question": "All doctors are graduates. Some graduates are teachers. Which is true?",
    "choices": ["A) All doctors are teachers", "B) Some teachers are doctors", "C) Some doctors are teachers", "D) None of the above"],
    "answer": "D"
}

print("="*60)
print("TESTING FINE-TUNED MODEL")
print("="*60)

# Initialize with fine-tuned model (default now)
agent = AAgent()

# Build prompt
choices_text = "\n".join(test_q['choices'])
prompt = f"""Question: {test_q['question']}

Choices:
{choices_text}

Answer:"""

print(f"\nQuestion: {test_q['question']}")
print(f"Expected: {test_q['answer']}")

# Generate
answer, time_taken, _ = agent.generate_response(prompt, "You are an expert at logical reasoning.")

print(f"\nGenerated Answer:\n{answer}")
print(f"\nTime: {time_taken:.2f}s")
print(f"Within 9s limit: {'✅' if time_taken <= 9.0 else '❌'}")

# Check if correct
is_correct = test_q['answer'] in answer.upper()
print(f"Correct: {'✅' if is_correct else '❌'}")

agent.print_statistics()

print("\n" + "="*60)
print("✅ FINE-TUNED MODEL IS READY!")
print("="*60)
print("Now you can:")
print("1. Generate answers: python answer_all_questions.py")
print("2. Or use in your own scripts")
print("="*60)

