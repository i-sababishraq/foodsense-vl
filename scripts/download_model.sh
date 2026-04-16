#!/bin/bash
# Download FoodSense-VL adapter weights from HuggingFace Hub
# Requires: pip install huggingface_hub
#
# Usage: bash scripts/download_model.sh

set -euo pipefail

REPO_ID="${HF_MODEL_REPO:-YOUR_USERNAME/foodsense-vl}"
CKPT_DIR="checkpoints"
TARGET_ADAPTER_DIR="${CKPT_DIR}/foodsense-vl_chkpt"

echo "Downloading FoodSense-VL model from: $REPO_ID"
echo "Target directory: $TARGET_ADAPTER_DIR"

# Check for huggingface_hub
if ! python -c "import huggingface_hub" 2>/dev/null; then
    echo "ERROR: huggingface_hub not installed. Run: pip install huggingface_hub"
    exit 1
fi

# Download model weights
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='${REPO_ID}',
    repo_type='model',
    local_dir='${TARGET_ADAPTER_DIR}',
    local_dir_use_symlinks=False,
)
print('Download complete!')
"

echo ""
echo "Model downloaded to: $TARGET_ADAPTER_DIR"
echo "Use with: python inference/foodsensevl.py --adapter_dir $TARGET_ADAPTER_DIR"
