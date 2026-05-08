#!/bin/bash
# setup_rocm.sh - Prepares the AMD ROCm environment for JurisSim LLM Training

echo "=== JurisSim ROCm Environment Setup ==="

# 1. Activate your virtual environment
source venv/bin/activate

# 2. Cleanup any stale memory, processes, or lock files
echo "[*] Cleaning up stale memory and locks..."
fuser -k /dev/kfd 2>/dev/null
rm -f ./jurissim-lora/.lock
rm -f training_metrics.jsonl.prev

# 3. Ensure the standard bitsandbytes is fully removed
echo "[*] Removing standard bitsandbytes to prevent conflicts..."
pip uninstall -y bitsandbytes

# 4. Install bitsandbytes (modern versions natively support ROCm)
echo "[*] Installing bitsandbytes..."
pip install bitsandbytes

# 5. Install ROCm-native Flash Attention 2
echo "[*] Installing build dependencies for flash-attn..."
pip install wheel ninja packaging
echo "[*] Installing flash-attn for ROCm..."
# The standard pip install flash-attn compiles against CUDA. For ROCm, we need the AMD fork or a source build.
pip install --no-build-isolation flash-attn --extra-index-url https://download.pytorch.org/whl/rocm6.2

# 6. Export variables just to ensure bash-level enforcement
export HSA_OVERRIDE_GFX_VERSION="9.4.2"
export PYTORCH_HIP_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM="false"

echo "=== Setup Complete! You can now run train_qlora.py ==="
