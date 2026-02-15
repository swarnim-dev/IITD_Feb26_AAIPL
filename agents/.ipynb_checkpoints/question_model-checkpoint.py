import os
import time
import json
import random
from typing import Optional, Union, List
from collections import deque
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class QAgent(object):
    def __init__(self, **kwargs):

        model_path = "hf_models/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side="left",
            local_files_only=True,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype="auto",
            device_map="auto",
            local_files_only=True,
            trust_remote_code=True
        )

        self.model.eval()

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        topics_path = os.path.join(base_dir, "assets", "topics.json")

        with open(topics_path, "r") as f:
            topics_data = json.load(f)

        # FIXED: Handle different topics.json formats
        self.topics = self._parse_topics(topics_data)
        
        # Diversity tracking mechanisms
        self.topic_history = deque(maxlen=15)  # Track last 15 topics used
        self.recent_questions = deque(maxlen=30)  # Track last 30 question texts
        self.topic_usage_count = {topic: 0 for topic in self.topics}  # Count usage per topic
        
        # Variation seeds for diversity
        self.context_variations = [
            "Use a corporate/business scenario with employees and departments",
            "Use a sports/games context with teams and players",
            "Use a family gathering with relatives of different generations",
            "Use a travel/geography theme with cities and countries",
            "Use a school/university setting with students and teachers",
            "Use abstract logical reasoning with symbols",
            "Use a shopping/market scenario with items and prices",
            "Use a transportation theme with vehicles and routes",
            "Use a technology/gadgets context",
            "Use a nature/animals setting",
            "Use a food/restaurant scenario",
            "Use a hospital/medical context",
            "Use a library/books theme",
            "Use a party/celebration setting",
            "Use a workplace meeting scenario"
        ]
        
        self.complexity_levels = [
            "Make it straightforward with 2-3 logical steps",
            "Add moderate complexity with 3-4 inference steps",
            "Create a challenging problem requiring careful analysis"
        ]


    def _parse_topics(self, topics_data):
        """
        Parse topics.json in different formats:
        
        Format 1: List of strings
        ["Syllogisms", "Seating Arrangements", ...]
        
        Format 2: Dict with string values
        {"topic1": "Syllogisms", "topic2": "Seating Arrangements", ...}
        
        Format 3: Dict with list values
        {"topic1": ["Syllogisms", "description"], "topic2": ["Seating", "desc"], ...}
        
        Format 4: List of lists/tuples
        [["Syllogisms", "desc"], ["Seating", "desc"], ...]
        
        Returns: List of topic strings
        """
        
        # Format 1: Already a list
        if isinstance(topics_data, list):
            # Check if list of strings or list of lists
            if topics_data and isinstance(topics_data[0], str):
                return topics_data
            elif topics_data and isinstance(topics_data[0], (list, tuple)):
                # Extract first element from each list/tuple
                return [item[0] if isinstance(item, (list, tuple)) else str(item) for item in topics_data]
            else:
                return [str(item) for item in topics_data]
        
        # Format 2 or 3: Dictionary
        elif isinstance(topics_data, dict):
            topics = []
            for key, value in topics_data.items():
                if isinstance(value, str):
                    # Format 2: {"topic1": "Syllogisms"}
                    topics.append(value)
                elif isinstance(value, (list, tuple)) and value:
                    # Format 3: {"topic1": ["Syllogisms", "description"]}
                    topics.append(value[0] if isinstance(value[0], str) else str(value[0]))
                else:
                    # Fallback: use the key as topic name
                    topics.append(key)
            return topics
        
        # Fallback: convert to string
        else:
            return [str(topics_data)]


    def select_diverse_topic(self):
        """
        Select a topic that ensures diversity:
        1. Avoid recently used topics
        2. Balance usage across all topics
        3. Add randomness for variety
        """
        # Get topics not used in recent history
        recent_topics_set = set(self.topic_history)
        available_topics = [t for t in self.topics if t not in recent_topics_set]
        
        # If all topics were recently used, use least recently used
        if not available_topics:
            # Get topics from the front of the deque (oldest)
            if len(self.topic_history) > len(self.topics) / 2:
                oldest_topics = list(self.topic_history)[:len(self.topics) // 2]
                available_topics = [t for t in self.topics if t not in oldest_topics]
            else:
                available_topics = self.topics.copy()
        
        # Among available topics, prefer those used less frequently
        if len(available_topics) > 1:
            # Sort by usage count (ascending) and take bottom 50%
            sorted_topics = sorted(available_topics, key=lambda t: self.topic_usage_count.get(t, 0))
            half_point = max(1, len(sorted_topics) // 2)
            available_topics = sorted_topics[:half_point]
        
        # Select randomly from available topics
        selected_topic = random.choice(available_topics)
        
        # Update tracking
        self.topic_history.append(selected_topic)
        self.topic_usage_count[selected_topic] = self.topic_usage_count.get(selected_topic, 0) + 1
        
        return selected_topic


    def is_unique_enough(self, new_question, threshold=0.5):
        """
        Check if the new question is sufficiently different from recent ones.
        Uses word overlap as a simple similarity metric.
        """
        if not self.recent_questions:
            return True
        
        # Normalize and tokenize
        new_words = set(new_question.lower().replace('\n', ' ').split())
        
        # Remove common words that don't indicate similarity
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                      'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
                      'statement', 'conclusion', 'if', 'some', 'all', 'no', 'question'}
        new_words = new_words - stop_words
        
        # Check against recent questions
        for old_q in self.recent_questions:
            old_words = set(old_q.lower().replace('\n', ' ').split()) - stop_words
            
            if not new_words or not old_words:
                continue
                
            # Calculate Jaccard similarity
            intersection = len(new_words & old_words)
            union = len(new_words | old_words)
            similarity = intersection / union if union > 0 else 0
            
            if similarity > threshold:
                return False
        
        return True


    def count_tokens_for_fields(self, parsed_json):
        """
        Count tokens for topic, question, choices, and answer fields.
        According to the constraint: these should total <= 150 tokens.
        """
        total_tokens = 0
        
        # Count topic tokens
        if 'topic' in parsed_json:
            topic_tokens = self.tokenizer.encode(parsed_json['topic'], add_special_tokens=False)
            total_tokens += len(topic_tokens)
        
        # Count question tokens
        if 'question' in parsed_json:
            question_tokens = self.tokenizer.encode(parsed_json['question'], add_special_tokens=False)
            total_tokens += len(question_tokens)
        
        # Count choices tokens
        if 'choices' in parsed_json and isinstance(parsed_json['choices'], list):
            for choice in parsed_json['choices']:
                choice_tokens = self.tokenizer.encode(choice, add_special_tokens=False)
                total_tokens += len(choice_tokens)
        
        # Count answer tokens
        if 'answer' in parsed_json:
            answer_tokens = self.tokenizer.encode(parsed_json['answer'], add_special_tokens=False)
            total_tokens += len(answer_tokens)
        
        return total_tokens


    def truncate_to_token_limit(self, parsed_json, max_tokens=150):
        """
        Ensure topic + question + choices + answer <= 150 tokens.
        Intelligently truncate while maintaining meaning.
        """
        current_tokens = self.count_tokens_for_fields(parsed_json)
        
        if current_tokens <= max_tokens:
            return parsed_json
        
        # Strategy: Truncate in order of priority (lowest to highest)
        # 1. Explanation (not counted in the 150 limit, but we still want it concise)
        # 2. Question (can be shortened)
        # 3. Choices (minimize words)
        # 4. Topic and Answer (keep as is - usually short)
        
        # First, aggressively truncate question if it's too long
        if 'question' in parsed_json:
            q_tokens = self.tokenizer.encode(parsed_json['question'], add_special_tokens=False)
            if len(q_tokens) > 40:  # Max ~20-25 words for question
                parsed_json['question'] = self.tokenizer.decode(q_tokens[:40], skip_special_tokens=True).strip()
        
        # Truncate each choice to max 10 tokens (~5-6 words)
        if 'choices' in parsed_json and isinstance(parsed_json['choices'], list):
            truncated_choices = []
            for choice in parsed_json['choices']:
                c_tokens = self.tokenizer.encode(choice, add_special_tokens=False)
                if len(c_tokens) > 10:
                    choice = self.tokenizer.decode(c_tokens[:10], skip_special_tokens=True).strip()
                truncated_choices.append(choice)
            parsed_json['choices'] = truncated_choices
        
        # Recount after truncation
        current_tokens = self.count_tokens_for_fields(parsed_json)
        
        # If still over limit, further reduce question
        if current_tokens > max_tokens:
            if 'question' in parsed_json:
                q_tokens = self.tokenizer.encode(parsed_json['question'], add_special_tokens=False)
                # Calculate how many tokens to remove
                excess = current_tokens - max_tokens
                target_q_tokens = max(20, len(q_tokens) - excess)
                parsed_json['question'] = self.tokenizer.decode(q_tokens[:target_q_tokens], skip_special_tokens=True).strip()
        
        return parsed_json


    def generate_response(
        self,
        message: Optional[Union[str, List[str]]] = None,
        system_prompt: Optional[str] = None,
        topic: Optional[str] = None,
        max_retries: int = 2,
        **kwargs
    ):
        """
        Generate questions with diversity and token constraints.
        
        Constraints:
        - topic + question + choices + answer <= 150 tokens
        - explanation can use remaining tokens (up to 874 for total 1024)
        - Each question should be generated under 13 seconds
        - Questions should be diverse and non-repetitive
        """
        
        # Select diverse topic if not provided
        if topic is None:
            selected_topic = self.select_diverse_topic()
        else:
            selected_topic = topic
            self.topic_history.append(selected_topic)
            self.topic_usage_count[selected_topic] = self.topic_usage_count.get(selected_topic, 0) + 1
        
        # Add variation seeds for diversity
        context_hint = random.choice(self.context_variations)
        complexity_hint = random.choice(self.complexity_levels)
        
        # Enhanced system prompt with strict constraints and diversity requirements
        enhanced_system_prompt = f"""{system_prompt}

TOPIC: {selected_topic}

DIVERSITY REQUIREMENTS (CRITICAL):
- {context_hint}
- {complexity_hint}
- Use UNIQUE names, numbers, and scenarios - never repeat from previous questions
- Vary the structure and approach significantly from typical patterns
- Make the question fresh and interesting

STRICT TOKEN LIMITS (CRITICAL):
- Topic + Question + Choices + Answer: MAXIMUM 150 tokens combined
- Question: Maximum 40 tokens (~20-25 words)
- Each choice: Maximum 10 tokens (~5-6 words)
- Answer: 1-2 tokens (just the letter)
- Explanation: Maximum 60 tokens (~30-35 words)
- Use concise, clear, direct language
- Avoid verbose problem statements
- No unnecessary words or repetition

FORMAT REQUIREMENTS:
- Return ONLY valid JSON with keys: topic, question, choices, answer, explanation
- Choices must be array of 4 options labeled A), B), C), D)
- Answer must be single letter: A, B, C, or D
- Keep all text concise and direct"""

        # Handle batch or single message
        if isinstance(message, list):
            texts = []
            for msg in message:
                messages = [
                    {"role": "system", "content": enhanced_system_prompt},
                    {"role": "user", "content": msg},
                ]
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                texts.append(text)

            model_inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.model.device)

        else:
            messages = [
                {"role": "system", "content": enhanced_system_prompt},
                {"role": "user", "content": message},
            ]

            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            model_inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True
            ).to(self.model.device)

        start_time = time.time()

        # Optimized generation parameters for speed and conciseness
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=180,  # Reduced for speed (150 tokens + some buffer for JSON structure)
            temperature=0.3,      # Moderate temperature for some creativity but focused output
            top_p=0.85,           # Slightly reduced for more focused generation
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        generation_time = time.time() - start_time

        outputs = []
        total_tokens = 0

        for input_ids, generated_sequence in zip(model_inputs.input_ids, generated_ids):

            output_ids = generated_sequence[len(input_ids):]
            total_tokens += len(output_ids)

            content = self.tokenizer.decode(
                output_ids,
                skip_special_tokens=True
            ).strip()

            import re
            match = re.search(r'\{.*\}', content, re.DOTALL)

            if not match:
                outputs.append("{}")
                continue

            raw_json = match.group()

            # Escape newlines only inside quoted strings
            def escape_newlines_inside_strings(text):
                result = []
                in_string = False
                i = 0
                while i < len(text):
                    char = text[i]
                    if char == '"' and (i == 0 or text[i - 1] != '\\'):
                        in_string = not in_string
                    if char == '\n' and in_string:
                        result.append('\\n')
                    else:
                        result.append(char)
                    i += 1
                return ''.join(result)

            cleaned_json = escape_newlines_inside_strings(raw_json)

            # Remove duplicate option labels like "A) A)"
            cleaned_json = re.sub(r'([ABCD]\)\s*)\1+', r'\1', cleaned_json)

            # Parse, validate, and enforce token limits
            try:
                parsed = json.loads(cleaned_json)
                
                # Ensure topic is set
                if 'topic' not in parsed or not parsed['topic']:
                    parsed['topic'] = selected_topic
                
                # Truncate to meet 150-token limit for main fields
                parsed = self.truncate_to_token_limit(parsed, max_tokens=150)
                
                # Also ensure explanation is concise (not counted in 150, but should be reasonable)
                if 'explanation' in parsed:
                    e_tokens = self.tokenizer.encode(parsed['explanation'], add_special_tokens=False)
                    if len(e_tokens) > 60:  # ~30-35 words
                        parsed['explanation'] = self.tokenizer.decode(e_tokens[:60], skip_special_tokens=True).strip()
                
                # Check for uniqueness (only for single messages, not batches)
                if not isinstance(message, list):
                    question_text = parsed.get('question', '')
                    if question_text and self.is_unique_enough(question_text):
                        self.recent_questions.append(question_text)
                    # Note: We still include the question even if not unique, 
                    # as regeneration would take too much time
                
                cleaned_json = json.dumps(parsed)
                
            except json.JSONDecodeError as e:
                # If JSON parsing fails, return empty JSON
                print(f"JSON decode error: {e}")
                cleaned_json = "{}"

            outputs.append(cleaned_json)

        if isinstance(message, list):
            return outputs, total_tokens, generation_time
        else:
            return outputs[0], total_tokens, generation_time


    def reset_diversity_tracking(self):
        """
        Reset diversity tracking mechanisms.
        Useful when starting a new batch of questions.
        """
        self.topic_history.clear()
        self.recent_questions.clear()
        self.topic_usage_count = {topic: 0 for topic in self.topics}


    def get_diversity_stats(self):
        """
        Get statistics about topic usage and diversity.
        Useful for monitoring and debugging.
        """
        return {
            'topic_usage': dict(self.topic_usage_count),
            'recent_topics': list(self.topic_history),
            'unique_topics_used': len(set(self.topic_history)),
            'total_questions': sum(self.topic_usage_count.values())
        }