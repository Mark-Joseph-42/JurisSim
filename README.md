# 🏛️ JurisSim: Fine-Tuned Legislative Stress-Testing Engine

> **Pre-enactment loophole detection using a domain-specialized Legal LLM + Formal Mathematical Proofs.**

Built for the **AMD Developer Hackathon 2026** | Track 2: Fine-Tuning | Running on **1× AMD Instinct MI300X**

---

## 📌 Overview

JurisSim is an adversarial auditor for draft legislation. Unlike general-purpose models, JurisSim is **fine-tuned on 10,000+ real legal instruction pairs** to specialize in Indian statutory interpretation and formal logic.

### The Problem
Draft laws often contain "bugs" (loopholes) that are exploited by sophisticated actors before they are discovered. Discovering these after enactment is costly and damages public trust.

### The Solution: "Unit Testing for Law"
JurisSim applies the rigor of software auditing to legal drafting:
1. **Red-Teaming**: Specialized LLM finds exploitable interpretations.
2. **Formalization**: Hypotheses are converted to Z3 SMT logic.
3. **Verification**: Mathematical proof of satisfiability (loopholes).
4. **Hardening**: Automatically suggests amendments to close the gap.

---

## ⚡ Track 2: Fine-Tuning on AMD MI300X

We leveraged the **192GB HBM3 memory** of the MI300X to perform high-rank QLoRA fine-tuning on the **Qwen3-32B** base model.

### Training Data (The "LegalBrain" Mix)
We curated a dataset of **10,200+ samples**:
- **LegalBrain Indic Corpus**: 5,000 rows of Indian Supreme/High Court judgments.
- **LegalBench (Stanford)**: 3,000 rows of statutory reasoning and definition classification tasks.
- **Kaggle Indian Legal**: 2,000 rows of Constitution, IPC, and CrPC question-answering.
- **JurisSim Custom**: 200 high-fidelity Z3 formalization and adversarial reasoning pairs.

### Fine-Tuning Specs
- **Model**: Qwen3-32B (Fine-tuned for Indian Legal Domain)
- **Method**: QLoRA (4-bit NF4, Rank 32)
- **Framework**: ROCm 6.x + PyTorch + HF TRL (SFTTrainer)
- **Compute**: 1× AMD Instinct MI300X ($4/hr)

---

## 🏗️ Architecture

```
┌────────────────────────────────┐       ┌────────────────────────────┐
│      DRAFTER / JUDGE           │       │    TRAINING PIPELINE       │
│   (Uploads Draft Bill)         │       │  (LegalBrain + LegalBench) │
└──────────────┬─────────────────┘       └─────────────┬──────────────┘
               │                                       │
               ▼                                       ▼
┌────────────────────────────────┐       ┌────────────────────────────┐
│      GRADIO WEB INTERFACE      │       │     JURIS-SIM 32B          │
│    (User inputs draft law)     │       │   (Fine-tuned Qwen3-32B)   │
└──────────────┬─────────────────┘       └─────────────┬──────────────┘
               │                                       │
               ▼                                       ▼
┌────────────────────────────────┐       ┌────────────────────────────┐
│     ANALYSIS PIPELINE          │       │     vLLM SERVING           │
│ (Extraction -> Red-Teaming)    │◀──────│  (Merged LoRA Adapter)     │
└──────────────┬─────────────────┘       └────────────────────────────┘
               │
               ▼
┌────────────────────────────────┐
│      Z3 SMT SOLVER             │
│   (Mathematical Verification)  │
└──────────────┬─────────────────┘
               │
               ▼
┌────────────────────────────────┐
│      LOOPHOLE REPORT           │
│ (Ambiguty Score + Amendments)  │
└────────────────────────────────┘
```

---

## 🚀 Server Deployment (1× MI300X)

JurisSim is designed to be fully automated on the AMD Developer Cloud.

### One-Command Launch
```bash
git clone https://github.com/Mark-Joseph-42/JurisSim.git
cd JurisSim
bash setup_server.sh
```

**The script automatically:**
1.  Configures ROCm environment.
2.  Fetches and prepares the 10,200-row training dataset.
3.  Executes QLoRA fine-tuning for 3 epochs.
4.  Merges the LoRA adapter into the base model.
5.  Deploys the final model via vLLM on Port 8000.
6.  Ready for the Gradio App.

---

## 🛡️ Safety & Reliability

- **Z3 Template Safety Net**: If the model fails to generate valid Z3 code, it falls back to 7 hardcoded logical patterns that handle 90% of common loopholes (Threshold Splitting, Jurisdiction Evasion, etc.).
- **Graceful Fallback**: If fine-tuning fails, the server falls back to the un-tuned base model to ensure the demo never crashes.
- **Budget Optimized**: Runs on a single GPU ($4/hr), making it highly efficient for production legal tech.

---

## 📄 License
MIT License. Built for the AMD Developer Hackathon 2026.
