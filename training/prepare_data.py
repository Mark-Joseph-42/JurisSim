import json
import os
import random
from datasets import load_dataset

def prepare_legalbrain():
    print("Processing LegalBrain Indic Dataset...")
    try:
        # Using a subset for faster processing in hackathon context, but targetting 5000
        dataset = load_dataset("Prarabdha/indian-legal-supervised-fine-tuning-data", split="train", streaming=True)
        data = []
        count = 0
        for item in dataset:
            context = item.get('context', '') or ''
            response = item.get('response', '') or ''
            question = item.get('question', '') or ''
            
            # Skip empty entries or too short responses
            if not response or len(response) < 30:
                continue
            
            # Filter for English and legal relevance
            lang = item.get('language', '')
            is_english = lang == 'en' or lang == '' or any(
                word in context.lower() for word in ['section', 'act', 'court', 'india', 'shall', 'provision']
            )
            
            if is_english:
                data.append({
                    "instruction": question if question else "Analyze the following legal provision.",
                    "input": context,
                    "output": response
                })
                count += 1
            if count >= 5000:
                break
        print(f"  → Loaded {len(data)} LegalBrain samples.")
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
                output = item.get('answer', '')
                if not output or len(str(output)) == 0:
                    continue
                all_data.append({
                    "instruction": f"Legal Reasoning Task ({task}): {item.get('question', 'Analyze the text.')}",
                    "input": item.get('text', ''),
                    "output": str(output)
                })
        except Exception as e:
            print(f"Error loading LegalBench task {task}: {e}")
    
    # Shuffle and sample
    random.shuffle(all_data)
    return all_data[:3000]

def merge_and_save():
    legalbrain = prepare_legalbrain()
    legalbench = prepare_legalbench()
    
    # Load our existing custom data (normalize 'response' to 'output')
    custom_data = []
    if os.path.exists("training/dataset.jsonl"):
        with open("training/dataset.jsonl", 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                # Key normalization
                output = d.get('output', d.get('response', ''))
                if output and len(output) >= 20:
                    custom_data.append({
                        "instruction": d.get('instruction', ''),
                        "input": d.get('input', ''),
                        "output": output
                    })

    final_dataset = legalbrain + legalbench + custom_data
    random.shuffle(final_dataset)
    
    print(f"Total merged dataset size: {len(final_dataset)}")
    
    # Train/Val Split (90/10)
    split_idx = int(len(final_dataset) * 0.9)
    train_data = final_dataset[:split_idx]
    val_data = final_dataset[split_idx:]
    
    print(f"Saving {len(train_data)} train samples and {len(val_data)} val samples.")
    
    with open("training/sft_dataset_train.jsonl", 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")
            
    with open("training/sft_dataset_val.jsonl", 'w') as f:
        for item in val_data:
            f.write(json.dumps(item) + "\n")
    
    # Also save a unified one for backward compatibility if needed
    with open("training/sft_dataset.jsonl", 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")
    
    print("Done. Datasets saved to training/sft_dataset_train.jsonl and sft_dataset_val.jsonl")

if __name__ == "__main__":
    merge_and_save()
