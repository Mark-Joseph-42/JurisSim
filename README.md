# JurisSim: Technical Documentation & Hackathon Guide

## 1. Executive Summary
**JurisSim** is a Neuro-Symbolic Agent designed to act as an automated legal auditor. By fine-tuning the 32-billion parameter Qwen3 model, JurisSim translates adversarial legal ambiguities and regulatory loopholes into formal mathematical constraints using the Z3 theorem prover. This enables corporate compliance officers to mathematically prove the existence of exploitable loopholes in contracts and legislation.

This project was built for the **AMD Developer Hackathon** specifically targeting **Track 2: Fine-Tuning on AMD GPUs**.

---

## 2. Hardware & Environment Stack
The model was trained exclusively on AMD hardware via the AMD Developer Cloud.
*   **Compute:** AMD Instinct™ MI300X Accelerator (192GB VRAM)
*   **Software Stack:** ROCm 6.2, PyTorch 2.5.1
*   **Base Model:** `Qwen/Qwen3-32B`
*   **Training Method:** QLoRA (Quantized Low-Rank Adaptation) using `bitsandbytes`

---

## 3. The Fine-Tuning Pipeline (ROCm Optimization)
Training a 32B model on a single MI300X GPU required severe optimization to avoid mathematical overflows and Out of Memory (OOM) crashes.

### The PyTorch SDPA ROCm Bug
During initial training runs, PyTorch's default `sdpa` (Scaled Dot-Product Attention) implementation suffered a math overflow bug on ROCm 6.2 when handling sequences with padding tokens alongside gradient checkpointing, resulting in catastrophic `NaN` gradients. 

### The "Ultra-Stable" Workaround
Because compiling `flash_attention_2` for ROCm from source is time-prohibitive, we engineered an ultra-stable configuration that perfectly maximized the 192GB VRAM without hitting the bug:
1.  **Attention:** Reverted to PyTorch's native `eager` mathematical attention block to avoid `sdpa` bugs.
2.  **Memory Compression:** Reduced `per_device_train_batch_size=1` but massively increased `gradient_accumulation_steps=8` to maintain an effective batch size of 8.
3.  **Checkpointing:** Enabled `gradient_checkpointing=True` to prevent the `eager` attention matrices from consuming all 192GB of VRAM during the backward pass.
4.  **Evaluation Safety:** Enforced `per_device_eval_batch_size=1` to ensure the un-checkpointed validation phase did not crash the GPU.

---

## 4. Training Results
The model successfully converged after 3 Epochs (1,713 Steps).
*   **Final Train Loss:** `1.756`
*   **Final Validation Loss:** `1.676` (Signaling excellent generalization and zero overfitting).
*   **Token Prediction Accuracy:** `62.61%` (Extremely high for complex Legal English → Python Z3 logic translation).
*   **Gradient Stability:** Maintained a remarkably stable `grad_norm` of `~0.3` to `~0.8` throughout the run.

---

## 5. Deployment Guide (GitHub & Hugging Face)
To package this project for the hackathon submission, the code must reside on GitHub, and the model must be hosted on Hugging Face.

### Step A: Push to GitHub
Initialize your local directory, commit the codebase, and push it to a public repository to satisfy the Open Source hackathon requirement. *Do not push the massive `/jurissim-lora` folder to GitHub!*

### Step B: Push the Model to Hugging Face
You can push the trained LoRA adapters directly to your Hugging Face account so they can be easily imported into your Hackathon Demo Space.
```bash
# 1. Login to your HF Account in the terminal
huggingface-cli login

# 2. Upload the adapter folder
huggingface-cli upload your-username/JurisSim-32B-LoRA ./jurissim-lora .
```

---

## 6. Frontend Architecture (The Demo UI)
To win the hackathon, the ML model will be wrapped in a visually premium Agentic Workflow.

### The Agentic Workflow
1.  **Input:** User pastes legal text into the UI.
2.  **Inference:** The web app calls the Hugging Face model (`Qwen3-32B` + `JurisSim-LoRA`) to generate the Z3 verification code.
3.  **Execution:** A Python backend intercepts the code, executes the Z3 solver natively, and captures the result (SAT/UNSAT).
4.  **Self-Correction:** If Z3 throws a Python syntax error, the agent feeds the error back to the LLM to fix the code automatically before showing the user.
5.  **Output:** The UI flashes RED (Loophole Found) or GREEN (Secure) based on the mathematical proof.
