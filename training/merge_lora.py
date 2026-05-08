import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge():
    base_model_id = os.environ.get("MODEL_ID", "Qwen/Qwen3-32B")
    adapter_path = "./jurissim-lora"
    merged_path = "./jurissim-merged"

    print(f"--- Merging LoRA Adapter ---")
    print(f"Base Model: {base_model_id}")
    print(f"Adapter: {adapter_path}")

    if not os.path.exists(adapter_path):
        print(f"Error: Adapter path {adapter_path} not found.")
        return

    # 1. Load base model (in bf16)
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    # 2. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

    # 3. Load adapter and merge
    print("Loading adapter and merging...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()

    # 4. Save merged model
    print(f"Saving merged model to {merged_path}...")
    model.save_pretrained(merged_path, safe_serialization=True)
    tokenizer.save_pretrained(merged_path)
    
    print(f"--- Merge Complete. Merged model saved to {merged_path} ---")

if __name__ == "__main__":
    merge()
