#!/bin/bash
# Download FoodSense dataset from HuggingFace Hub
# Requires: pip install huggingface_hub
#
# Usage: bash scripts/download_data.sh

set -euo pipefail

REPO_ID="${HF_DATASET_REPO:-sababishraq/foodsense-dataset}"
DATA_DIR="data"

echo "Downloading FoodSense dataset from: $REPO_ID"
echo "Target directory: $DATA_DIR"

# Check for huggingface_hub
if ! python -c "import huggingface_hub" 2>/dev/null; then
    echo "ERROR: huggingface_hub not installed. Run: pip install huggingface_hub"
    exit 1
fi

# Download dataset files. The HF repo stores the JPEGs at its root; put them
# in data/Images/ (the default --image_dir) and the annotations in data/.
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='${REPO_ID}',
    repo_type='dataset',
    local_dir='${DATA_DIR}',
    allow_patterns=['metadata.csv', 'README.md'],
)
snapshot_download(
    repo_id='${REPO_ID}',
    repo_type='dataset',
    local_dir='${DATA_DIR}/Images',
    allow_patterns=['*.jpg'],
)
print('Download complete!')
"

echo ""
echo "Dataset downloaded to: $DATA_DIR/"
echo "You should now have:"
echo "  - data/metadata.csv  (66,842 annotations, 2,987 images)"
echo "  - data/Images/       (food images)"
echo "The paper's train/val/test image lists are in splits/ (see splits/README.md)."
