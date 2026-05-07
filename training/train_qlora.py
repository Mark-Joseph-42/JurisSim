import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

def train():
    # Configuration
    model_id = os.environ.get("MODEL_ID", "Qwen/Qwen3-32B")
    dataset_path = "training/sft_dataset.jsonl"
    output_dir = "./jurissim-lora"

    print(f"--- Starting QLoRA Training for {model_id} ---")

    # 1. QLoRA: 4-bit NF4 quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    # 2. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 3. Load Model (quantized)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # 4. LoRA config — passed to SFTTrainer, NOT manually wrapped
    peft_config = LoraConfig(
        r=32,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # 5. Load dataset
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    # 6. Formatting function — converts instruction/input/output to chat format
    def formatting_func(examples):
        texts = []
        for i in range(len(examples['instruction'])):
            instruction = examples['instruction'][i]
            inp = examples.get('input', [''] * len(examples['instruction']))[i] or ''
            output = examples['output'][i]
            
            user_msg = instruction
            if inp:
                user_msg += f"\n\nContext: {inp}"
            
            text = (
                f"<|im_start|>system\n"
                f"You are JurisSim-32B, a legal auditor specializing in adversarial loophole detection and Z3 formal verification.<|im_end|>\n"
                f"<|im_start|>user\n{user_msg}<|im_end|>\n"
                f"<|im_start|>assistant\n{output}<|im_end|>"
            )
            texts.append(text)
        return texts

    # 7. SFT Training config
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
    )

    # 8. SFTTrainer handles LoRA wrapping internally via peft_config
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=sft_config,
        peft_config=peft_config,
        formatting_func=formatting_func,
        tokenizer=tokenizer,
    )

    # 9. Train
    print(f"Dataset size: {len(dataset)} examples")
    print("Training in progress...")
    trainer.train()
    
    # 10. Save adapter
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"--- Training Complete. Adapter saved to {output_dir} ---")

if __name__ == "__main__":
    train()
