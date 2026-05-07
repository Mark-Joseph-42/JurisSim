#!/bin/bash
# ============================================================
# JurisSim MI300X Server Setup & Training Pipeline
# ============================================================
# One-command: Data -> Train -> Merge -> Serve
# Usage: bash setup_server.sh
# ============================================================

echo "=========================================="
echo " ⚖️ JurisSim: Legal Auditor Training Pipeline"
echo " Target Hardware: 1x AMD Instinct MI300X"
echo "=========================================="

JURIS_FALLBACK=false

# ---- 1. Environment ----
echo "[1/6] Configuring environment..."
export HSA_OVERRIDE_GFX_VERSION=9.4.2
export VLLM_ROCM_USE_AITER=1
grep -q HSA_OVERRIDE ~/.bashrc 2>/dev/null || {
    echo 'export HSA_OVERRIDE_GFX_VERSION=9.4.2' >> ~/.bashrc
    echo 'export VLLM_ROCM_USE_AITER=1' >> ~/.bashrc
}

pip install vllm openai gradio \
    sentence-transformers qdrant-client z3-solver \
    python-dotenv trl peft accelerate datasets bitsandbytes -q
echo "  ✓ Dependencies installed."

# ---- 2. Prepare Data ----
echo "[2/6] Preparing training data..."
python3 training/expand_data.py
python3 training/prepare_data.py
echo "  ✓ Training data ready."

# ---- 3. Configure .env ----
echo "[3/6] Configuring .env..."
cat > .env << 'ENVEOF'
MODEL_ID=Qwen/Qwen3-32B
QDRANT_URL=:memory:
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
LLM_API_URL=http://localhost:8000/v1
USE_API=true
ENVEOF

# ---- 4. QLoRA Fine-Tuning ----
echo "[4/6] Starting QLoRA Fine-Tuning (est. 45-60 min)..."
echo "       Monitor GPU usage: watch rocm-smi"
if python3 training/train_qlora.py; then
    echo "  ✓ Training succeeded."
else
    echo "  ✗ Training failed! Falling back to base model."
    JURIS_FALLBACK=true
fi

# ---- 5. Merge Adapter ----
if [ "$JURIS_FALLBACK" = "false" ]; then
    echo "[5/6] Merging LoRA adapters into base model..."
    if python3 training/merge_lora.py; then
        SERVE_MODEL="./jurissim-merged"
        echo "  ✓ Merged model saved."
    else
        echo "  ✗ Merge failed! Falling back to base model."
        SERVE_MODEL="Qwen/Qwen3-32B"
    fi
else
    echo "[5/6] Skipping merge — using base Qwen3-32B."
    SERVE_MODEL="Qwen/Qwen3-32B"
fi

# Update .env with the actual served model
sed -i "s|MODEL_ID=.*|MODEL_ID=$SERVE_MODEL|" .env

# ---- 6. Launch vLLM ----
echo "[6/6] Launching vLLM inference server..."
tmux kill-session -t vllm 2>/dev/null || true
tmux new-session -d -s vllm "python3 -m vllm.entrypoints.openai.api_server \
    --model $SERVE_MODEL \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --dtype bfloat16 \
    --port 8000 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 2>&1 | tee vllm_server.log"

echo ""
echo "=========================================="
echo " ✓ JurisSim setup complete!"
echo "=========================================="
echo ""
echo " Model: $SERVE_MODEL"
echo " vLLM:  tmux attach -t vllm  (wait for 'Uvicorn running')"
echo ""
echo " Once vLLM is ready, start the demo:"
echo "   python3 app.py"
echo ""
echo " The Gradio app will print a public URL for remote access."
echo "=========================================="
