import sys
sys.path.insert(0, 'agents')
from answer_model import AAgent

model_path = "hf_models/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
print("Loading model...")
agent = AAgent(model_path)

prompt = """Question: All doctors are graduates. Some graduates are teachers. Which is definitely true?

Choices:
A) All doctors are teachers
B) Some teachers are doctors
C) Some doctors are teachers  
D) None of the above

Answer:"""

print("Generating answer...")
answer, time_taken, _ = agent.generate_response(prompt, "You are an expert at logical reasoning.")

print(f"\nTime: {time_taken:.2f}s")
print(f"Answer:\n{answer}")
print(f"\n{'✅' if time_taken <= 9.0 else '❌'} Speed: {time_taken:.2f}s / 9.0s limit")
