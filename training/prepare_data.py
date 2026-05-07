import json
import os
import random
from datasets import load_dataset

def prepare_legalbrain():
    print("Processing LegalBrain Indic Dataset...")
    try:
        # Load a subset to avoid memory issues and keep processing fast
        dataset = load_dataset("Prarabdha/indian-legal-supervised-fine-tuning-data", split="train", streaming=True)
        data = []
        count = 0
        for item in dataset:
            # Simple heuristic for quality: English only, has context and response
            if item.get('language') == 'en' or any(word in item.get('context', '').lower() for word in ['india', 'court', 'act', 'section']):
                data.append({
                    "instruction": item.get('question', 'Explain the legal context.'),
                    "input": item.get('context', ''),
                    "output": item.get('response', '')
                })
                count += 1
            if count >= 5000:
                break
        return data
    except Exception as e:
        print(f"Error loading LegalBrain: {e}")
        return []

def prepare_legalbench():
    print("Processing LegalBench Tasks...")
    tasks = [
        "definition_classification", 
        "unfair_tos", 
        "contract_nli_confidentiality_of_agreement",
        "rule_qa",
        "statutory_reasoning"
    ]
    all_data = []
    for task in tasks:
        try:
            dataset = load_dataset("nguha/legalbench", task, split="train")
            for item in dataset:
                all_data.append({
                    "instruction": f"Legal Reasoning Task ({task}): {item.get('question', 'Analyze the text.')}",
                    "input": item.get('text', ''),
                    "output": item.get('answer', '')
                })
        except Exception as e:
            print(f"Error loading LegalBench task {task}: {e}")
    
    # Shuffle and sample
    random.shuffle(all_data)
    return all_data[:3000]

def prepare_kaggle_mock():
    # Since I cannot download from Kaggle directly without credentials in this script, 
    # I will provide a structure for it. On the server, the user would provide the file.
    # For now, I'll return empty list as a placeholder if file doesn't exist.
    print("Checking for Kaggle Indian Legal Dataset...")
    path = "training/kaggle_indian_legal.json"
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
            return [{"instruction": d['question'], "input": "", "output": d['answer']} for d in data[:2000]]
    return []

def merge_and_save():
    legalbrain = prepare_legalbrain()
    legalbench = prepare_legalbench()
    kaggle = prepare_kaggle_mock()
    
    # Load our existing custom data
    custom_data = []
    if os.path.exists("training/dataset.jsonl"):
        with open("training/dataset.jsonl", 'r') as f:
            for line in f:
                d = json.loads(line)
                custom_data.append({
                    "instruction": d['instruction'],
                    "input": d.get('input', ''),
                    "output": d['response']
                })

    final_dataset = legalbrain + legalbench + kaggle + custom_data
    random.shuffle(final_dataset)
    
    print(f"Total merged dataset size: {len(final_dataset)}")
    
    with open("training/sft_dataset.jsonl", 'w') as f:
        for item in final_dataset:
            f.write(json.dumps(item) + "\n")
    
    print("Saved to training/sft_dataset.jsonl")

if __name__ == "__main__":
    merge_and_save()
