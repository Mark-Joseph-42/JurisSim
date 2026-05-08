<div align="center">
  <img src="https://img.shields.io/badge/AMD-Developer_Hackathon-ED1C24?style=for-the-badge&logo=amd&logoColor=white" alt="AMD Hackathon Badge"/>
  <img src="https://img.shields.io/badge/Hardware-MI300X_Accelerator-00A3E0?style=for-the-badge" alt="MI300X Badge"/>
  <img src="https://img.shields.io/badge/Model-Qwen3--32B-blue?style=for-the-badge" alt="Qwen Model Badge"/>
  
  <h1>⚖️ JurisSim</h1>
  <p><b>A Neuro-Symbolic Legal Auditor & Formal Verification Agent</b></p>
</div>

---

## 📖 Overview
**JurisSim** is an advanced AI agent designed to bridge the gap between generative language and mathematical certainty. Built for the **AMD Developer Hackathon (Track 2: Fine-Tuning on AMD GPUs)**, JurisSim acts as an automated legal compliance officer. 

Rather than simply summarizing contracts or highlighting vague legal text, JurisSim translates adversarial legislative ambiguities into **formal mathematical constraints using the Z3 theorem prover**. It then executes these constraints to mathematically prove whether an exploitable loophole exists, saving organizations millions in regulatory risks and litigation.

## ✨ Core Features
*   **Neuro-Symbolic Translation:** Fine-tuned to translate complex "Legal English" into executable Z3 Python logic.
*   **Agentic Verification Loop:** Automatically intercepts the LLM's Z3 code, executes it in a secure sandbox, and parses the mathematical proof (`SAT`/`UNSAT`).
*   **Self-Correcting Reasoning:** If the Z3 solver throws a syntax error, the agent automatically feeds the error back to the LLM for self-correction before returning the final result to the user.
*   **100% AMD Powered:** Trained and optimized entirely on AMD Instinct MI300X hardware using ROCm.

---

## 📚 Technical Documentation & Architecture

### 1. Hardware & Environment Stack
*   **Compute:** AMD Instinct™ MI300X Accelerator (192GB VRAM)
*   **Software Stack:** ROCm 6.2, PyTorch 2.5.1
*   **Base Model:** `Qwen/Qwen3-32B`
*   **Training Method:** QLoRA (Quantized Low-Rank Adaptation) using `bitsandbytes`

### 2. The Fine-Tuning Pipeline & ROCm Optimizations
Training a massive 32-billion parameter model on a single MI300X GPU required severe optimization to avoid mathematical overflows and Out of Memory (OOM) crashes.

*   **The PyTorch SDPA ROCm Bug:** During initial training runs, PyTorch's default `sdpa` (Scaled Dot-Product Attention) implementation suffered a math overflow bug on ROCm 6.2 when handling sequences with padding tokens alongside gradient checkpointing, resulting in catastrophic `NaN` gradients.
*   **The "Ultra-Stable" Workaround:** Because compiling `flash_attention_2` for ROCm from source is time-prohibitive, we engineered an ultra-stable configuration that perfectly maximized the 192GB VRAM without hitting the bug:
    1.  **Attention:** Reverted to PyTorch's native `eager` mathematical attention block to avoid `sdpa` math corruption.
    2.  **Memory Compression:** Reduced `per_device_train_batch_size=1` but massively increased `gradient_accumulation_steps=8` to maintain an effective batch size of 8.
    3.  **Checkpointing:** Enabled `gradient_checkpointing=True` to prevent the `eager` attention matrices from consuming all 192GB of VRAM during the backward pass.
    4.  **Evaluation Safety:** Enforced `per_device_eval_batch_size=1` to ensure the un-checkpointed validation phase did not crash the GPU.

### 3. Training Results
The model successfully converged after 3 Epochs (1,713 Steps).
*   **Final Train Loss:** `1.756`
*   **Final Validation Loss:** `1.676` *(Signaling excellent generalization and zero overfitting).*
*   **Token Prediction Accuracy:** `62.61%` *(Extremely high for complex Legal English → Python Z3 logic translation).*
*   **Gradient Stability:** Maintained a remarkably stable `grad_norm` of `~0.3` to `~0.8` throughout the run.

---

## 🚀 Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Mark-Joseph-42/JurisSim.git
cd JurisSim
```

### 2. Install Dependencies
Ensure you are running on an AMD machine with ROCm installed.
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Ensure bitsandbytes is configured for ROCm
pip install https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/v0.44.1/bitsandbytes-0.44.1-py3-none-manylinux_2_24_x86_64.whl
```

### 3. Download the Fine-Tuned Model
The LoRA adapters are publicly available on Hugging Face:
```bash
huggingface-cli download markjoseph2003/JurisSim-32B-LoRA --local-dir ./jurissim-lora
```

### 4. Run the Agent (Coming Soon!)
*(The Agentic Execution loop and Frontend UI are currently being finalized. Instructions for launching the UI will be placed here shortly).*

---

## 🛠️ Hugging Face Space
The interactive demo of JurisSim will be deployed as a Hugging Face Space for the hackathon judging. The link will be provided upon completion of the UI.

<div align="center">
  <p><i>Built with ❤️ for the AMD Developer Hackathon</i></p>
</div>
