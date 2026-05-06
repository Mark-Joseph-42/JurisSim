# This script runs on MI300X with ROCm
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset

def format_example(example):
    """Formats the instruction and response into the Qwen chat format."""
    return f"<|im_start|>system\nYou are a legal formalization engine specialized in Z3 solver code.<|im_end|>\n<|im_start|>user\n{example['instruction']}<|im_end|>\n<|im_start|>assistant\n{example['response']}<|im_end|>"

def train():
    model_id = os.environ.get("MODEL_ID", "Qwen/Qwen3-14B")
    dataset_path = "training/dataset.jsonl"

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return

    # 1. 4-bit Quantization Config (NF4)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    # 2. LoRA Config
    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM"
    )

    # 3. Load Model and Tokenizer
    print(f"Loading base model {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 4. Load Dataset
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    # 5. SFT Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        formatting_func=format_example,
        tokenizer=tokenizer,
        max_seq_length=4096,
        args=TrainingArguments(
            output_dir="./jurissim-lora",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            bf16=True, # Use bfloat16 on MI300X
            logging_steps=10,
            save_strategy="epoch",
            report_to="none"
        ),
    )

    # 6. Train
    print("Starting training...")
    trainer.train()
    
    # 7. Save
    model.save_pretrained("./jurissim-lora")
    print("Training complete. Adapter saved to ./jurissim-lora")

if __name__ == "__main__":
    print("JurisSim QLoRA Training Script (Validated for MI300X)")
    # train() # Uncomment on server
