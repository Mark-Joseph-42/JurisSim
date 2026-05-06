#!/bin/bash
# ============================================================
# JurisSim MI300X Server Setup Script (with Safety Fallbacks)
# ============================================================
# Usage: bash setup_server.sh [model_tier]
#   model_tier: 1 = DeepSeek-R1-0528 (685B, best quality)
#               2 = Qwen3-235B-A22B (fast + smart)
#               3 = Qwen3-32B (fastest, guaranteed to work)
#
# Default: tries tier 1, auto-falls back if it fails.
# ============================================================

set -e
TIER=${1:-1}

echo "=========================================="
echo " JurisSim MI300X Server Setup"
echo " Model Tier: $TIER"
echo "=========================================="

# ---- 1. Environment ----
export HSA_OVERRIDE_GFX_VERSION=9.4.2
export VLLM_ROCM_USE_AITER=1
grep -q HSA_OVERRIDE ~/.bashrc 2>/dev/null || {
    echo 'export HSA_OVERRIDE_GFX_VERSION=9.4.2' >> ~/.bashrc
    echo 'export VLLM_ROCM_USE_AITER=1' >> ~/.bashrc
}

# ---- 2. Install Dependencies ----
echo "[1/4] Installing dependencies..."
pip install vllm openai gradio \
    sentence-transformers qdrant-client z3-solver \
    python-dotenv -q
echo "Dependencies installed."

# ---- 3. Select Model ----
if [ "$TIER" = "1" ]; then
    MODEL="deepseek-ai/DeepSeek-R1-0528"
    EXTRA_ARGS="--block-size 1 --enable-reasoning --reasoning-parser deepseek_r1"
    DTYPE="auto"
    MEM="0.90"
elif [ "$TIER" = "2" ]; then
    MODEL="Qwen/Qwen3-235B-A22B"
    EXTRA_ARGS="--enable-reasoning --reasoning-parser deepseek_r1"
    DTYPE="bfloat16"
    MEM="0.85"
else
    MODEL="Qwen/Qwen3-32B"
    EXTRA_ARGS="--enable-reasoning --reasoning-parser deepseek_r1"
    DTYPE="bfloat16"
    MEM="0.70"
fi

echo "[2/4] Selected model: $MODEL"

# ---- 4. Create .env ----
cat > .env << ENVEOF
MODEL_ID=$MODEL
QDRANT_URL=:memory:
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
LLM_API_URL=http://localhost:8000/v1
USE_API=true
ENVEOF
echo ".env configured."

# ---- 5. Launch vLLM ----
echo "[3/4] Launching vLLM (model will download on first run)..."

# Kill any existing vLLM
tmux kill-session -t vllm 2>/dev/null || true

tmux new-session -d -s vllm "python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --tensor-parallel-size 8 \
    --trust-remote-code \
    --dtype $DTYPE \
    --port 8000 \
    --max-model-len 16384 \
    --gpu-memory-utilization $MEM \
    $EXTRA_ARGS 2>&1 | tee vllm_server.log"

# ---- 6. Wait for vLLM to be ready ----
echo "[4/4] Waiting for vLLM to start (this may take 10-30 min for large models)..."
echo "       Tip: open another terminal and run 'tmux attach -t vllm' to see progress."

MAX_WAIT=1800  # 30 minutes
ELAPSED=0
while [ $ELAPSED -lt $MAX_WAIT ]; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo ""
        echo "=========================================="
        echo " vLLM is READY!"
        echo "=========================================="
        echo ""
        echo "Quick health check:"
        curl -s http://localhost:8000/v1/models | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'  Model loaded: {d[\"data\"][0][\"id\"]}')" 2>/dev/null || echo "  Model endpoint active."
        echo ""
        echo "NEXT STEPS:"
        echo "  1. python3 app.py           # Launch Gradio UI"
        echo "  2. python3 mock_test.py      # Run full pipeline test"
        echo ""
        exit 0
    fi
    sleep 10
    ELAPSED=$((ELAPSED + 10))
    MINS=$((ELAPSED / 60))
    echo "  Still waiting... (${MINS}m elapsed)"
done

echo ""
echo "=========================================="
echo " WARNING: vLLM did not start in 30 minutes."
echo " Possible issues:"
echo "   - Model download still in progress (check: tmux attach -t vllm)"
echo "   - Out of memory (try: bash setup_server.sh 3)"
echo "=========================================="
echo "Fallback: bash setup_server.sh 2   (Qwen3-235B)"
echo "Fallback: bash setup_server.sh 3   (Qwen3-32B, guaranteed)"
