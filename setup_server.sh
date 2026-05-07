#!/bin/bash
# ============================================================
# JurisSim MI300X Server Setup & Training Pipeline
# ============================================================
# One-command to rule them all: Data -> Train -> Merge -> Serve
# ============================================================

set -e

echo "=========================================="
echo " ⚖️ JurisSim: Legal Auditor Training Pipeline"
echo " Target Hardware: 1x AMD Instinct MI300X"
echo "=========================================="

# ---- 1. Environment Setups ----
echo "[1/6] Configuring environment..."
export HSA_OVERRIDE_GFX_VERSION=9.4.2
export VLLM_ROCM_USE_AITER=1
grep -q HSA_OVERRIDE ~/.bashrc 2>/dev/null || {
    echo 'export HSA_OVERRIDE_GFX_VERSION=9.4.2' >> ~/.bashrc
    echo 'export VLLM_ROCM_USE_AITER=1' >> ~/.bashrc
}

# Install core dependencies
pip install vllm openai gradio \
    sentence-transformers qdrant-client z3-solver \
    python-dotenv trl peft accelerate datasets -q

# ---- 2. Prepare Data ----
echo "[2/6] Preparing training data (fetching real legal datasets)..."
# Expand JurisSim custom data first
python3 training/expand_data.py
# Download and merge external datasets (LegalBrain, LegalBench)
python3 training/prepare_data.py

# ---- 3. Configure .env ----
echo "[3/6] Configuring .env..."
cat > .env << ENVEOF
MODEL_ID=Qwen/Qwen3-32B
QDRANT_URL=:memory:
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
LLM_API_URL=http://localhost:8000/v1
USE_API=true
ENVEOF

# ---- 4. QLoRA Fine-Tuning ----
echo "[4/6] Starting QLoRA Fine-Tuning (est. 45-60 min)..."
python3 training/train_qlora.py || {
    echo "Training failed. Falling back to base model for inference."
    export JURIS_FALLBACK=true
}

# ---- 5. Merge Adapter ----
if [ "$JURIS_FALLBACK" != "true" ]; then
    echo "[5/6] Merging LoRA adapters into base model..."
    python3 training/merge_lora.py
    SERVE_MODEL="./jurissim-merged"
else
    echo "[5/6] Skipping merge, using base model."
    SERVE_MODEL="Qwen/Qwen3-32B"
fi

# ---- 6. Launch vLLM ----
echo "[6/6] Launching vLLM with fine-tuned model..."
tmux kill-session -t vllm 2>/dev/null || true
tmux new-session -d -s vllm "python3 -m vllm.entrypoints.openai.api_server \
    --model $SERVE_MODEL \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --dtype bfloat16 \
    --port 8000 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.85"

echo "------------------------------------------"
echo " SUCCESS: JurisSim is deploying!"
echo " Use 'tmux attach -t vllm' to watch logs."
echo " Once ready, run: python3 app.py"
echo "------------------------------------------"
