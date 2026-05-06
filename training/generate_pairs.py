import os
import json
import re
import argparse
import time
import sys

# Force UTF-8 for stdout on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from src.z3_solver import LogicSolver
from openai import OpenAI

SYSTEM_PROMPT = """You are a legal formalization engine. Given a legal clause, you must:
1. Identify a potential loophole or adversarial interpretation
2. Write executable Python Z3 code that proves whether the loophole is satisfiable
3. The code must use `from z3 import *`, define variables, add constraints, call `s.check()`, and print the result

CRITICAL RULES:
- Output ONLY valid Python code. No markdown. No explanation. No thinking.
- Use ONLY ASCII characters. Use <= instead of ≤, >= instead of ≥, etc.
- Do NOT wrap code in ``` blocks.
- The code must be directly executable by Python."""

# Unicode math symbol replacements
UNICODE_REPLACEMENTS = {
    '\u2264': '<=',   # ≤
    '\u2265': '>=',   # ≥
    '\u2260': '!=',   # ≠
    '\u2192': '->',   # →
    '\u2190': '<-',   # ←
    '\u2227': 'And',  # ∧
    '\u2228': 'Or',   # ∨
    '\u00ac': 'Not',  # ¬
    '\u2203': '',      # ∃
    '\u2200': '',      # ∀
}

def sanitize_code(code: str) -> str:
    """Strip thinking blocks, markdown, and Unicode math from LLM output."""
    # Strip <think>...</think> blocks (reasoning models)
    code = re.sub(r'<think>.*?</think>', '', code, flags=re.DOTALL)
    
    # Strip markdown code fences
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0].strip()
    elif "```" in code:
        parts = code.split("```")
        if len(parts) >= 3:
            code = parts[1].strip()
        elif len(parts) >= 2:
            code = parts[1].strip()
    
    # Replace Unicode math symbols with ASCII equivalents
    for uni_char, replacement in UNICODE_REPLACEMENTS.items():
        code = code.replace(uni_char, replacement)
    
    # Strip any remaining non-ASCII characters that would break cp1252
    code = code.encode('ascii', 'ignore').decode('ascii')
    
    # Remove empty lines at the start
    lines = code.split('\n')
    while lines and not lines[0].strip():
        lines.pop(0)
    
    return '\n'.join(lines).strip()

def generate_z3_via_api(clause, client, model="gpt-4o"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Formalize this legal clause as Z3 Python code:\n\n{clause}"}
        ],
        temperature=0.7
    )
    raw_code = response.choices[0].message.content
    return sanitize_code(raw_code)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--provider", type=str, default="lmstudio", choices=["openai", "lmstudio", "mock"])
    parser.add_argument("--api_key", type=str, default="lm-studio")
    parser.add_argument("--base_url", type=str, default="http://127.0.0.1:1234/v1")
    parser.add_argument("--model", type=str, default="model-identifier")
    parser.add_argument("--retries", type=int, default=2, help="Number of retries per clause on failure")
    args = parser.parse_args()
    
    solver = LogicSolver()
    
    # Initialize client
    if args.provider == "mock":
        client = None
    else:
        client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    
    with open("training/legal_clauses.jsonl", "r", encoding='utf-8') as f:
        clauses = [json.loads(line) for line in f if line.strip()]
    
    dataset = []
    for i in range(min(args.count, len(clauses))):
        clause = clauses[i]['clause']
        print(f"Generating pair {i+1}/{args.count}: {clause[:60]}...")
        
        success = False
        for attempt in range(1, args.retries + 1):
            try:
                if args.provider == "mock":
                    # Hardcoded mock for CI
                    z3_code = """from z3 import *
s = Solver()
x = Int('x')
s.add(x > 0)
if s.check() == sat:
    print("sat")
else:
    print("unsat")
"""
                else:
                    z3_code = generate_z3_via_api(clause, client, model=args.model)
                
                # Skip if code looks too short or empty
                if len(z3_code) < 30:
                    print(f"  [attempt {attempt}] Too short, retrying...")
                    continue
                
                # VALIDATE: actually run the Z3 code
                result = solver.verify_code(z3_code)
                
                if result['result'] in ('sat', 'unsat'):
                    pair = {
                        "instruction": f"Formalize this clause:\n\n{clause}",
                        "response": z3_code,
                        "z3_result": result['result'],
                        "valid": True,
                        "timestamp": time.time()
                    }
                    dataset.append(pair)
                    print(f"  [+] Valid pair (Result: {pair['z3_result']})")
                    success = True
                    break
                else:
                    print(f"  [attempt {attempt}] Z3 error: {result['stderr'][:80]}")
                    
            except Exception as e:
                print(f"  [attempt {attempt}] Error: {str(e)[:100]}")
        
        if not success:
            print(f"  [-] Failed after {args.retries} attempts")
            
    # Append to existing dataset or create new
    existing_data = []
    if os.path.exists("training/dataset.jsonl"):
        with open("training/dataset.jsonl", "r", encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    existing_data.append(json.loads(line))
    
    # Combine and save
    with open("training/dataset.jsonl", "w", encoding='utf-8') as f:
        for entry in existing_data + dataset:
            f.write(json.dumps(entry, ensure_ascii=True) + "\n")
    
    print(f"\nTotal pairs in training/dataset.jsonl: {len(existing_data) + len(dataset)}")

if __name__ == "__main__":
    main()
