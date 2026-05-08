#!/bin/bash
# JurisSim — MI300X Server Deployment & Training Script

echo "--- JurisSim Server Initialization ---"

# 1. Environment Check
export PATH=$PATH:/opt/rocm/bin
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
export HSA_OVERRIDE_GFX_VERSION=9.4.2

if ! command -v rocm-smi &> /dev/null; then
    echo "ERROR: ROCm not found. Please run Phase 0 installation manually or ensure /opt/rocm/bin is in PATH."
    exit 1
fi

rocm-smi

# 2. Virtual Env
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# 3. Data Preparation
if [ ! -f "training/sft_dataset_train.jsonl" ]; then
    echo "Running data preparation pipeline..."
    python3 training/expand_data.py
    python3 training/prepare_data.py
fi

# 4. Training (Optional - trigger with --train flag)
if [[ "$*" == *"--train"* ]]; then
    echo "Starting fine-tuning process..."
    python3 training/train_qlora.py
    echo "Merging LoRA adapter..."
    python3 training/merge_lora.py
    export MODEL_ID="./jurissim-merged"
fi

# 5. vLLM Server Launch
if [[ "$*" == *"--serve"* ]]; then
    echo "Launching vLLM server..."
    python3 -m vllm.entrypoints.openai.api_server \
        --model ${MODEL_ID:-"Qwen/Qwen3-32B"} \
        --tensor-parallel-size 1 \
        --trust-remote-code \
        --dtype bfloat16 \
        --port 8000 \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.85 &
    
    echo "Waiting for vLLM to initialize..."
    sleep 30
fi

# 6. Launch App
if [[ "$*" == *"--app"* ]]; then
    echo "Starting JurisSim Gradio App..."
    python3 app.py
fi
