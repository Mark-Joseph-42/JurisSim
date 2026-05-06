#!/bin/bash
# JurisSim MI300X Server Setup Script
# Run this on your 8x MI300X instance

echo "--- JurisSim Server Setup Starting ---"

# 1. Environment Overrides for MI300X
export HSA_OVERRIDE_GFX_VERSION=9.4.2
export VLLM_ROCM_USE_AITER=1
echo "export HSA_OVERRIDE_GFX_VERSION=9.4.2" >> ~/.bashrc
echo "export VLLM_ROCM_USE_AITER=1" >> ~/.bashrc

# 2. Install Dependencies
echo "Installing python dependencies..."
pip install vllm openai gradio \
    sentence-transformers qdrant-client z3-solver \
    python-dotenv -q

# 3. Create .env file
echo "Configuring .env..."
cat > .env << 'EOF'
MODEL_ID=deepseek-ai/DeepSeek-R1-0528
QDRANT_URL=:memory:
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
LLM_API_URL=http://localhost:8000/v1
USE_API=true
EOF

# 4. Launch vLLM in background (using tmux)
echo "Launching vLLM with DeepSeek-R1-0528 on 8 GPUs..."
echo "This will take a while to download the model (~700GB)."
tmux new-session -d -s vllm "python3 -m vllm.entrypoints.openai.api_server \
    --model deepseek-ai/DeepSeek-R1-0528 \
    --tensor-parallel-size 8 \
    --block-size 1 \
    --trust-remote-code \
    --dtype auto \
    --port 8000 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.90 \
    --enable-reasoning \
    --reasoning-parser deepseek_r1"

echo "vLLM is starting in tmux session 'vllm'."
echo "Use 'tmux attach -t vllm' to see logs."
echo "Wait for 'Uvicorn running on http://0.0.0.0:8000' before running the app."

# 5. Launch Gradio App (once vLLM is ready)
echo "Once vLLM is ready, you can start the demo with: python3 app.py"
echo "--- Setup Script Complete ---"
