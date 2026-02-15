"""
Production-Ready Answer Generation Model (AAgent)
FIXED: Warm-up pass eliminates slow first inference
"""

import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Tuple


class AAgent:
    
    def __init__(self, model_path: str = None, **kwargs):
        if model_path is None:
            model_path = kwargs.get('model_path',
                'hf_models/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8')

        print(f"Loading answer model from {model_path}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, padding_side='left'
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        self.model.eval()

        self.stats = {
            "total_attempts": 0, "successful": 0,
            "timeout_failures": 0, "token_limit_exceeded": 0,
            "validation_failed": 0, "total_time": 0.0, "avg_time": 0.0
        }

        # CRITICAL: Warm-up pass so first real question isn't slow
        self._warmup()

        print(f"✓ Answer model ready on device: {self.model.device}")

    def _warmup(self):
        """Run a dummy forward pass to warm up ROCm kernels."""
        print("Running warm-up pass (eliminates slow first inference)...")
        warmup_start = time.time()
        try:
            messages = [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Answer: A\nReason: test"}
            ]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            #print(f"✓ Warm-up done in {time.time() - warmup_start:.1f}s — first question will now be fast")
        except Exception as e:
            print(f"⚠ Warm-up failed (non-fatal): {e}")

    def generate_response(self, prompt: str, system_prompt: str, **gen_kwargs) -> Tuple[str, float, float]:
        
        self.stats["total_attempts"] += 1
        start_time = time.time()

        try:
            enhanced_system_prompt = "You are an expert at logical reasoning."
            enhanced_prompt = f"""{prompt}

Instructions:
1. Analyze each choice carefully
2. Use logical reasoning
3. State your answer clearly

Provide your answer in this format:
Answer: [A/B/C/D]
Reasoning: [Brief explanation]"""

            messages = [
                {"role": "system", "content": enhanced_system_prompt},
                {"role": "user", "content": enhanced_prompt}
            ]

            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

            input_length = model_inputs.input_ids.shape[1]
            max_new = min(80, 1024 - input_length)

            if max_new < 20:
                return "Error: Not enough token budget", 0.0, 0.0

            gen_start = time.time()
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=max_new,
                    min_new_tokens=15,
                    temperature=0.5,
                    top_p=0.85,
                    top_k=30,
                    do_sample=True,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            generation_time = time.time() - gen_start

            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            answer = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            total_time = time.time() - start_time
            self.stats["total_time"] += total_time
            self.stats["avg_time"] = self.stats["total_time"] / self.stats["total_attempts"]
            self.stats["successful"] += 1

            return answer, generation_time, 0.0

        except Exception as e:
            self.stats["validation_failed"] += 1
            error_time = time.time() - start_time
            print(f"⚠ Error: {e}")
            return f"Error: {str(e)}", error_time, 0.0

    def print_statistics(self):
        print("\n" + "="*60)
        print("ANSWER GENERATION STATISTICS")
        print("="*60)
        print(f"Total attempts:       {self.stats['total_attempts']}")
        print(f"Successful:           {self.stats['successful']}")
        print(f"Validation failures:  {self.stats['validation_failed']}")
        if self.stats['total_attempts'] > 0:
            success_rate = (self.stats['successful'] / self.stats['total_attempts']) * 100
            print(f"\nSuccess rate: {success_rate:.1f}%")
            print(f"Average time: {self.stats['avg_time']:.2f}s")
        print("="*60 + "\n")
