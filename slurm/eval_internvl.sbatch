#!/bin/bash
#SBATCH --job-name=reeval_internvl
#SBATCH --partition=ai             # UPDATE: set your SLURM partition
#SBATCH --account=soc250046-ai    # UPDATE: set your SLURM account
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=80G
#SBATCH --output=logs/reeval_internvl_%j.out
#SBATCH --error=logs/reeval_internvl_%j.err

# InternVL re-evaluation runner (portable defaults).
#
# Usage:
#   sbatch eval_sensory_internvl_reeval.sbatch
#
# Optional overrides:
#   SPLIT=test|val
#   MAX_IMAGES=50
#   IMAGE_IDS='img1.jpg,img2.jpg'
#   MAX_NEW_TOKENS=1024
#   OUTPUT_DIR=eval_outputs/internvl_reeval_custom
#   HUMAN_CSV=data/FINAL_DATASET_COMPLETE_with_rescaling.csv
#   IMAGE_DIR=data/Images
#   PROMPT='...'

set -euo pipefail

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ_DIR"

mkdir -p logs

if [ -f .env ]; then
  source .env
  export HF_TOKEN
  export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
  echo "Loaded .env"
else
  echo "ERROR: .env not found in project root: $PROJ_DIR"
  exit 1
fi

# Canonical data defaults requested for re-eval portability
HUMAN_CSV="${HUMAN_CSV:-data/FINAL_DATASET_COMPLETE_with_rescaling.csv}"
IMAGE_DIR="${IMAGE_DIR:-data/Images}"

SPLIT="${SPLIT:-test}"
MAX_IMAGES="${MAX_IMAGES:-}"
IMAGE_IDS="${IMAGE_IDS:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
PROMPT="${PROMPT:-Analyze this food image and estimate the likely sensory experience. Plain text only (no markdown; no bullets; no asterisks). Provide a Sensory Assessment with 4 sections, one each for Taste, Smell, Texture, and Sound, formatted as: Sense (X.X/5.0): 3-4 sentences of detailed visual justification.}"
OUTPUT_DIR="${OUTPUT_DIR:-eval_outputs/internvl_reeval_${SLURM_JOB_ID}}"

if [ ! -f "$HUMAN_CSV" ]; then
  echo "ERROR: HUMAN_CSV not found: $HUMAN_CSV"
  exit 1
fi
if [ ! -d "$IMAGE_DIR" ]; then
  echo "ERROR: IMAGE_DIR not found: $IMAGE_DIR"
  exit 1
fi
if [ -e "$OUTPUT_DIR" ]; then
  echo "ERROR: OUTPUT_DIR already exists (non-overwrite guard): $OUTPUT_DIR"
  echo "Set OUTPUT_DIR to a new path, e.g. OUTPUT_DIR=eval_outputs/internvl_reeval_${SLURM_JOB_ID}_v2"
  exit 1
fi

echo "============================================"
echo "Sensory Re-Eval: InternVL"
echo "============================================"
echo "Job ID        : $SLURM_JOB_ID"
echo "Node          : ${SLURMD_NODENAME:-unknown}"
echo "Start Time    : $(date)"
echo "Split         : $SPLIT"
echo "Human CSV     : $HUMAN_CSV"
echo "Image Dir     : $IMAGE_DIR"
echo "Output Dir    : $OUTPUT_DIR"
echo "Max New Tokens: $MAX_NEW_TOKENS"
if [ -n "$MAX_IMAGES" ]; then
  echo "Max Images    : $MAX_IMAGES"
fi
if [ -n "$IMAGE_IDS" ]; then
  echo "Image IDs     : $IMAGE_IDS"
fi
echo "============================================"

export HF_HOME="$PWD/cache/huggingface"
export PIP_CACHE_DIR="$PWD/cache/pip"
export XDG_CACHE_HOME="$PWD/cache"
export TMPDIR="$PWD/cache/tmp"
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$TMPDIR"

export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

LOCAL_STAGE="${LOCAL:-/local}/${SLURM_JOB_ID}_eval"
MODEL_CACHE_NAME="models--OpenGVLab--InternVL2_5-26B"
SRC_CACHE="$PWD/cache/huggingface/hub/$MODEL_CACHE_NAME"

cleanup_local() {
  echo "Cleaning up NVMe staging: $LOCAL_STAGE"
  rm -rf "$LOCAL_STAGE" 2>/dev/null || true
}
trap cleanup_local EXIT

if [ -d "$SRC_CACHE" ]; then
  echo "Staging InternVL cache to NVMe: $LOCAL_STAGE"
  mkdir -p "$LOCAL_STAGE/huggingface/hub"
  for attempt in 1 2 3 4 5; do
    rsync -a "$SRC_CACHE" "$LOCAL_STAGE/huggingface/hub/" && break
    echo "  rsync attempt $attempt failed, retrying in 10s..."
    sleep 10
  done
  export HF_HOME="$LOCAL_STAGE/huggingface"
  echo "NVMe staging complete"
else
  echo "WARNING: InternVL cache not found at $SRC_CACHE"
  echo "         continuing with shared HF cache"
fi

IMAGE="${CONTAINER_SIF:-$PROJ_DIR/containers/flash_attn.sif}"
VENV_DIR="/workspace/.venvs/sensory_internvl_compat"
TRANSFORMERS_PIN="4.48.3"

SING_EXEC=(apptainer exec --nv --writable-tmpfs --cleanenv --home /tmp
  --bind "$PWD:/workspace"
  --bind "$LOCAL_STAGE:/local_stage"
  --pwd /workspace
  --env HF_TOKEN="$HF_TOKEN"
  --env HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
  --env HF_HOME="/local_stage/huggingface"
  --env PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF"
  --env TOKENIZERS_PARALLELISM="$TOKENIZERS_PARALLELISM"
  --env PYTHONUNBUFFERED="1"
  --env SAFETENSORS_FAST_GPU="1"
  "$IMAGE")

VENV_SETUP='
set -euo pipefail
if [ -e /usr/local/cuda/compat/lib/libcuda.so.1 ] && [ ! -e /usr/local/cuda/compat/lib/libcuda.so ]; then
  ln -s /usr/local/cuda/compat/lib/libcuda.so.1 /usr/local/cuda/compat/lib/libcuda.so || true
fi
export LD_LIBRARY_PATH="/usr/local/cuda/compat/lib:${LD_LIBRARY_PATH:-}"
if [ ! -d "'$VENV_DIR'" ]; then
  python3 -m venv --system-site-packages "'$VENV_DIR'"
fi
source "'$VENV_DIR'/bin/activate"
python -m pip install -U pip -q

TARGET_TF="'$TRANSFORMERS_PIN'"
CUR_TF=$(python - <<'"'"'PY'"'"'
try:
    import transformers
    print(transformers.__version__)
except Exception:
    print("missing")
PY
)

if [ "$CUR_TF" != "$TARGET_TF" ]; then
  python -m pip install "numpy<2" -q
  python -m pip install --no-deps "transformers==$TARGET_TF" "tokenizers<0.22" -q
fi

python - <<'"'"'PY'"'"'
import importlib.util, subprocess, sys
pkg_map = {
    "accelerate": "accelerate",
    "peft": "peft",
    "bitsandbytes": "bitsandbytes",
    "pandas": "pandas",
    "PIL": "pillow",
    "sklearn": "scikit-learn",
    "scipy": "scipy",
    "einops": "einops",
    "pyvips": "pyvips",
}
missing = []
for module_name, pip_name in pkg_map.items():
    probe = "PIL" if module_name == "PIL" else module_name
    if importlib.util.find_spec(probe) is None:
        missing.append(pip_name)
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
PY

python - <<'"'"'PY'"'"'
import transformers, torch
print("  transformers:", transformers.__version__)
print("  torch:", torch.__version__)
print("  cuda:", torch.cuda.is_available(), "devices:", torch.cuda.device_count())
PY
'

"${SING_EXEC[@]}" bash -lc "
$VENV_SETUP
python -u evaluate.py \
  --model internvl \
  --split '$SPLIT' \
  --human_csv '$HUMAN_CSV' \
  --image_dir '$IMAGE_DIR' \
  --output_dir '$OUTPUT_DIR' \
  --prompt '$PROMPT' \
  --max_new_tokens $MAX_NEW_TOKENS \
  --local_files_only \
  --strict \
  ${MAX_IMAGES:+--max_images $MAX_IMAGES} \
  ${IMAGE_IDS:+--image_ids '$IMAGE_IDS'}
"

echo "============================================"
echo "End Time: $(date)"
echo "Re-eval output: $OUTPUT_DIR"
echo "============================================"
