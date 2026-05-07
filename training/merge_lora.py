import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

def merge_lora():
    base_model_id = os.environ.get("MODEL_ID", "Qwen/Qwen3-32B")
    adapter_dir = "./jurissim-lora"
    output_dir = "./jurissim-merged"

    print(f"--- Merging LoRA Adapter from {adapter_dir} into {base_model_id} ---")

    # Load base model in bfloat16
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    # Load adapter
    model = PeftModel.from_pretrained(base_model, adapter_dir)

    # Merge and unload
    print("Merging weights...")
    merged_model = model.merge_and_unload()

    # Save merged model and tokenizer
    print(f"Saving merged model to {output_dir}...")
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("--- Merge Complete ---")

if __name__ == "__main__":
    merge_lora()
