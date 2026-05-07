import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

def train():
    # Configuration
    model_id = os.environ.get("MODEL_ID", "Qwen/Qwen3-32B")
    dataset_path = "training/sft_dataset.jsonl"
    output_dir = "./jurissim-lora"

    print(f"--- Starting QLoRA Training for {model_id} ---")

    # 1. Loading BitsAndBytes for 4-bit (NF4)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    # 2. Load Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)

    # 3. LoRA Configuration (Targeting all linear layers)
    peft_config = LoraConfig(
        r=32,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, peft_config)

    # 4. Load SFT Dataset
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    # 5. SFT Trainer Config
    sft_config = SFTConfig(
        output_dir=output_dir,
        max_seq_length=2048,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        bf16=True,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        dataset_text_field="text" # We'll map instruction/input/output to a text field
    )

    def formatting_func(example):
        text = f"<|im_start|>system\nYou are JurisSim-32B, a legal auditor specializing in adversarial loophole detection.<|im_end|>\n"
        text += f"<|im_start|>user\n{example['instruction']}\n\nContext: {example.get('input', '')}<|im_end|>\n"
        text += f"<|im_start|>assistant\n{example['output']}<|im_end|>"
        return {"text": text}

    dataset = dataset.map(formatting_func)

    # 6. Initialize Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=sft_config,
        peft_config=peft_config,
        tokenizer=tokenizer,
    )

    # 7. Start Training
    print("Training in progress...")
    trainer.train()
    
    # 8. Save Adapter
    trainer.save_model(output_dir)
    print(f"--- Training Complete. Adapter saved to {output_dir} ---")

if __name__ == "__main__":
    train()
