import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.prompts import (
    CLAUSE_EXTRACTION_PROMPT, 
    RED_TEAM_PROMPT, 
    FORMALIZATION_PROMPT, 
    PATCH_GENERATION_PROMPT,
    PATTERN_CLASSIFICATION_PROMPT,
    AMBIGUITY_SCORING_PROMPT
)
from src import z3_templates

class LegalLLM:
    def __init__(self):
        model_id = os.environ.get("MODEL_ID", "Qwen/Qwen1.5-1.8B-Chat")
        print(f"Loading {model_id} in 4-bit precision...")
        
        # Configure BitsAndBytes for 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto"
        )
        # Pad token fix
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print("Model loaded successfully.")

    def _generate(self, prompt: str, max_tokens=512) -> str:
        messages = [
            {"role": "system", "content": "You are a legal AI assistant specialized in formal logic and Z3."},
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=max_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Clean up any markdown blocks
        if "```python" in response:
            response = response.split("```python")[1].split("```")[0].strip()
        elif "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].strip()
            
        return response.strip()

    def extract_clauses(self, bill_text: str) -> list[str]:
        prompt = CLAUSE_EXTRACTION_PROMPT.format(bill_text=bill_text)
        response = self._generate(prompt)
        clauses = [line.strip("- ").strip("* ") for line in response.split('\n') if line.strip()]
        return [c for c in clauses if len(c) > 10]

    def red_team_clause(self, clause: str, context: str) -> list[str]:
        prompt = RED_TEAM_PROMPT.format(clause=clause, context=context)
        response = self._generate(prompt)
        hypotheses = [line.strip("- ").strip("* ").strip("123. ") for line in response.split('\n') if line.strip()]
        return [h for h in hypotheses if len(h) > 10][:3]

    def _classify_pattern(self, clause: str, hypothesis: str) -> dict:
        prompt = PATTERN_CLASSIFICATION_PROMPT.format(clause=clause, hypothesis=hypothesis)
        response = self._generate(prompt, max_tokens=256)
        try:
            # Find JSON in response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                data = json.loads(response[json_start:json_end])
                # Normalize keys
                if "type" in data and "pattern" not in data:
                    data["pattern"] = data["type"]
                if "parameters" in data and "params" not in data:
                    data["params"] = data["parameters"]
                return data
        except Exception:
            pass
        return {"pattern": "none"}

    def formalize_to_z3(self, clause: str, hypothesis: str, context: str) -> str:
        # Path 1: Template-based extraction
        classification = self._classify_pattern(clause, hypothesis)
        if classification.get("pattern") not in ("none", None):
            try:
                z3_code = z3_templates.render(classification['pattern'], classification.get('params', {}))
                if z3_code:
                    print(f"  [+] Using template: {classification['pattern']}")
                    return z3_code
            except (TypeError, KeyError) as e:
                print(f"  [~] Template render failed ({e}), falling back...")
        
        # Path 2: Raw few-shot generation
        print("  [~] Falling back to raw Z3 generation...")
        prompt = FORMALIZATION_PROMPT.format(clause=clause, hypothesis=hypothesis, context=context)
        return self._generate(prompt)

    def generate_patch(self, clause: str, hypothesis: str, z3_code: str) -> str:
        prompt = PATCH_GENERATION_PROMPT.format(clause=clause, hypothesis=hypothesis, z3_code=z3_code)
        return self._generate(prompt, max_tokens=256)

    def score_ambiguity(self, clause: str, context: str) -> float:
        prompt = AMBIGUITY_SCORING_PROMPT.format(clause=clause, context=context)
        response = self._generate(prompt, max_tokens=128)
        try:
            import re
            nums = re.findall(r'\d+\.?\d*', response)
            if nums:
                score = float(nums[-1])
                return min(max(score, 0.0), 1.0)
        except Exception:
            pass
        return 0.5

class LegalLLM_API:
    """API-based inference for vLLM-served model on MI300X."""
    def __init__(self):
        from openai import OpenAI
        api_url = os.environ.get("LLM_API_URL", "http://localhost:8000/v1")
        self.client = OpenAI(base_url=api_url, api_key="none")
        # Auto-detect served model from vLLM /v1/models endpoint
        try:
            models = self.client.models.list()
            self.model = models.data[0].id
            print(f"Auto-detected served model: {self.model}")
        except Exception:
            self.model = os.environ.get("MODEL_ID", "Qwen/Qwen3-32B")
        print(f"Using API inference: {api_url} with model {self.model}")

    def _generate(self, prompt: str, max_tokens=2048) -> str:
        import time
        last_error = None
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a legal AI assistant specialized in formal logic and Z3. Output only what is asked."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=0.1
                )
                response = resp.choices[0].message.content
                break
            except Exception as e:
                last_error = e
                wait = 2 ** attempt
                print(f"  [!] API call failed (attempt {attempt+1}/3): {e}. Retrying in {wait}s...")
                time.sleep(wait)
        else:
            print(f"  [!] All 3 API attempts failed: {last_error}")
            return ""
        # Strip thinking tags if present (DeepSeek-R1)
        if "<think>" in response:
            parts = response.split("</think>")
            response = parts[-1].strip() if len(parts) > 1 else response
        
        # Clean up any markdown blocks
        if "```python" in response:
            response = response.split("```python")[1].split("```")[0].strip()
        elif "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].strip()
            
        return response.strip()

    def extract_clauses(self, bill_text: str) -> list[str]:
        prompt = CLAUSE_EXTRACTION_PROMPT.format(bill_text=bill_text)
        response = self._generate(prompt)
        clauses = [line.strip("- ").strip("* ") for line in response.split('\n') if line.strip()]
        return [c for c in clauses if len(c) > 10]

    def red_team_clause(self, clause: str, context: str) -> list[str]:
        prompt = RED_TEAM_PROMPT.format(clause=clause, context=context)
        response = self._generate(prompt)
        hypotheses = [line.strip("- ").strip("* ").strip("123. ") for line in response.split('\n') if line.strip()]
        return [h for h in hypotheses if len(h) > 10][:3]

    def _classify_pattern(self, clause: str, hypothesis: str) -> dict:
        prompt = PATTERN_CLASSIFICATION_PROMPT.format(clause=clause, hypothesis=hypothesis)
        response = self._generate(prompt, max_tokens=256)
        try:
            # Find JSON in response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                data = json.loads(response[json_start:json_end])
                # Normalize keys
                if "type" in data and "pattern" not in data:
                    data["pattern"] = data["type"]
                if "parameters" in data and "params" not in data:
                    data["params"] = data["parameters"]
                return data
        except Exception:
            pass
        return {"pattern": "none"}

    def formalize_to_z3(self, clause: str, hypothesis: str, context: str) -> str:
        # Path 1: Template-based extraction
        classification = self._classify_pattern(clause, hypothesis)
        if classification.get("pattern") not in ("none", None):
            try:
                z3_code = z3_templates.render(classification['pattern'], classification.get('params', {}))
                if z3_code:
                    print(f"  [+] Using template: {classification['pattern']}")
                    return z3_code
            except (TypeError, KeyError) as e:
                print(f"  [~] Template render failed ({e}), falling back...")
        
        # Path 2: Raw generation
        print("  [~] Generating bespoke Z3 via LLM API...")
        prompt = FORMALIZATION_PROMPT.format(clause=clause, hypothesis=hypothesis, context=context)
        return self._generate(prompt, max_tokens=2048)

    def generate_patch(self, clause: str, hypothesis: str, z3_code: str) -> str:
        prompt = PATCH_GENERATION_PROMPT.format(clause=clause, hypothesis=hypothesis, z3_code=z3_code)
        return self._generate(prompt, max_tokens=512)

    def score_ambiguity(self, clause: str, context: str) -> float:
        prompt = AMBIGUITY_SCORING_PROMPT.format(clause=clause, context=context)
        response = self._generate(prompt, max_tokens=128)
        try:
            import re
            nums = re.findall(r'\d+\.?\d*', response)
            if nums:
                score = float(nums[-1])
                return min(max(score, 0.0), 1.0)
        except Exception:
            pass
        return 0.5
