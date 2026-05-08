import os
import sys
import json
import math
import subprocess

# --- ROCm/MI300X environment --- 
# Must be set before any torch imports to ensure proper hardware allocation
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "9.4.2"
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Ensure bitsandbytes-rocm is installed or throw a clear error
try:
    import bitsandbytes as bnb
except ImportError:
    print("FATAL: bitsandbytes is missing. Please run the setup commands to install bitsandbytes-rocm.")
    sys.exit(1)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset


class MetricsCallback(TrainerCallback):
    """Flush metrics to JSONL and stdout. Halt on persistent NaN/zero loss."""

    def __init__(self, metrics_file="training_metrics.jsonl"):
        self.metrics_file = metrics_file
        self.nan_streak = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return

        # Serialise — NaN is not valid JSON
        clean = {}
        for k, v in logs.items():
            if isinstance(v, float) and math.isnan(v):
                clean[k] = "NaN"
            else:
                clean[k] = v

        with open(self.metrics_file, "a") as f:
            f.write(json.dumps({"step": state.global_step, **clean}) + "\n")

        loss = logs.get("loss", None)
        grad = logs.get("grad_norm", None)
        lr   = logs.get("learning_rate", "?")
        ep   = logs.get("epoch", "?")

        print(
            f"[STEP {state.global_step:>4d}]  loss={loss!s:<10}  "
            f"grad_norm={grad!s:<12}  lr={lr!s:<12}  epoch={ep}"
        )
        sys.stdout.flush()

        # Early-stop on NaN / zero loss streak
        if loss is not None and (loss == 0.0 or (isinstance(loss, float) and math.isnan(loss))):
            self.nan_streak += 1
            if self.nan_streak >= 3:
                print("\n‼ FATAL: 3 consecutive NaN/zero-loss steps — aborting training.")
                control.should_training_stop = True
        else:
            self.nan_streak = 0


def formatting_func(example):
    """Convert instruction/input/output → Qwen3 ChatML prompt."""
    instruction = example.get("instruction", "")
    inp = example.get("input", "") or ""
    output = example.get("output", "")

    user_msg = instruction
    if inp:
        user_msg += f"\n\nContext: {inp}"

    return (
        "<|im_start|>system\n"
        "You are JurisSim-32B, a legal auditor specializing in adversarial "
        "loophole detection and Z3 formal verification.<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n{output}<|im_end|>"
    )


def preflight(model, tokenizer, train_dataset):
    """Forward-only pre-flight on one sample to verify model health."""
    print("\n=== Pre-flight validation ===")

    sample = train_dataset[0]
    text = formatting_func(sample)
    tok = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    tok = {k: v.to(model.device) for k, v in tok.items()}
    tok["labels"] = tok["input_ids"].clone()

    model.eval()
    with torch.no_grad():
        out = model(**tok)

    fwd_loss = out.loss.item()
    print(f"  Forward loss : {fwd_loss:.4f}")
    assert fwd_loss > 0 and not math.isnan(fwd_loss), "Forward loss is zero or NaN!"
    print(f"  Logits dtype : {out.logits.dtype}")
    print(f"  Model dtype  : {next(model.parameters()).dtype}")
    print("  Pre-flight PASSED ✓")
    print("=" * 44 + "\n")
    return True


def train():
    model_id = os.environ.get("MODEL_ID", "Qwen/Qwen3-32B")
    train_path = "training/sft_dataset_train.jsonl"
    val_path   = "training/sft_dataset_val.jsonl"
    output_dir = "./jurissim-lora"

    # --- Banner ---
    print("--- JurisSim Fine-Tuning (ROCm / MI300X) ---")
    print(f"  Model : {model_id}")
    print(f"  Torch : {torch.__version__}")
    print(f"  ROCm  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU   : {props.name}")
        print(f"  VRAM  : {props.total_memory / 1024**3:.1f} GB")

    # Back up old metrics so we start clean
    for fn in ("training_metrics.jsonl",):
        if os.path.exists(fn):
            os.rename(fn, fn + ".prev")
            print(f"  Backed up {fn}")

    # ---- 1. Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ---- 2. Model (bf16, eager attn) ----
    print("Loading model (bf16, eager) …")
    
    # Bypass Transformers' hardcoded Ampere check for tf32=True while still getting the perf boost
    torch.backends.cuda.matmul.allow_tf32 = True
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map={"": 0},
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model.config.use_cache = False

    # ---- 3. LoRA ----
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    )

    # ---- 4. Datasets ----
    train_dataset = load_dataset("json", data_files=train_path, split="train")
    val_dataset   = load_dataset("json", data_files=val_path,   split="train")

    # ---- 5. Forward pre-flight ----
    preflight(model, tokenizer, train_dataset)

    # ---- 6. SFT config ----
    sft_config = SFTConfig(
        output_dir=output_dir,
        max_length=2048,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,

        # Conservative LR for 32B
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        weight_decay=0.01,

        # Precision — MI300X specific optimizations
        bf16=True,
        # tf32=True would cause a ValueError in HF Trainer since it assumes only NVIDIA Ampere supports it. 
        # We manually enabled it via torch.backends.cuda.matmul.allow_tf32 = True above instead.

        # Gradient stability
        max_grad_norm=0.3,

        # We cannot use packing without flash_attn, so standard padding is used.
        packing=False,

        # Gradient checkpointing ON to fit eager attention in 192GB VRAM
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # Logging
        logging_steps=5,
        logging_first_step=True,
        disable_tqdm=True,

        # Eval & save
        eval_strategy="steps",
        eval_steps=200,
        per_device_eval_batch_size=1,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        report_to="none",

        # Optimizer — paged_adamw_32bit is much more robust for LoRA
        optim="paged_adamw_32bit",
    )

    # ---- 7. Trainer ----
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=sft_config,
        peft_config=peft_config,
        formatting_func=formatting_func,
        processing_class=tokenizer,
        callbacks=[MetricsCallback()],
    )

    # ---- 8. Train ----
    n_train = len(train_dataset)
    eff_bs  = sft_config.per_device_train_batch_size * sft_config.gradient_accumulation_steps
    print(f"Train : {n_train} samples  |  Val : {len(val_dataset)} samples")
    print(f"Batch : {eff_bs} effective  |  LR  : {sft_config.learning_rate}")
    print(f"Warmup: {sft_config.warmup_steps} steps  |  Clip : {sft_config.max_grad_norm}")
    print(f"Optim : {sft_config.optim}")
    print(f"Grad ckpt: {sft_config.gradient_checkpointing} (use_reentrant=False)")
    print("Starting training …\n")

    trainer.train()

    # ---- 9. Save ----
    print(f"\nSaving adapter → {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("--- Training Complete ---")


if __name__ == "__main__":
    train()
