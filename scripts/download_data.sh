#!/bin/bash
# Download FoodSense dataset from HuggingFace Hub
# Requires: pip install huggingface_hub
#
# Usage: bash scripts/download_data.sh

set -euo pipefail

REPO_ID="${HF_DATASET_REPO:-YOUR_USERNAME/foodsense-dataset}"
DATA_DIR="data"

echo "Downloading FoodSense dataset from: $REPO_ID"
echo "Target directory: $DATA_DIR"

# Check for huggingface_hub
if ! python -c "import huggingface_hub" 2>/dev/null; then
    echo "ERROR: huggingface_hub not installed. Run: pip install huggingface_hub"
    exit 1
fi

# Download dataset files
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='${REPO_ID}',
    repo_type='dataset',
    local_dir='${DATA_DIR}',
    local_dir_use_symlinks=False,
)
print('Download complete!')
"

echo ""
echo "Dataset downloaded to: $DATA_DIR/"
echo "You should now have:"
echo "  - data/FINAL_DATASET_COMPLETE_with_rescaling.csv"
echo "  - data/Images/"
echo "  - data/Apify_Yelp_photos/ (if included)"
