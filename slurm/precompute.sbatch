#!/bin/bash
#SBATCH --job-name=mammoth_v2
#SBATCH --partition=ai             # UPDATE: set your SLURM partition
#SBATCH --account=soc250046-ai    # UPDATE: set your SLURM account
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --mem=80G
#SBATCH --output=logs/mammoth_v2_%A_%a.out
#SBATCH --error=logs/mammoth_v2_%A_%a.err

# =============================================================================
# MAmmoTH-v2 Target Generation
#
# Modes:
#   PILOT:  Generate for 50 images with both Gemma bf16 and Qwen bf16
#           export MODE=pilot
#   FULL:   Generate for all images with chosen teacher
#           export MODE=full
#
# Teacher selection:
#   export TEACHER=gemma    (default: google/gemma-3-27b-it bf16)
#   export TEACHER=qwen     (Qwen/Qwen2.5-VL-32B-Instruct bf16)
# =============================================================================

set -euo pipefail

MODE="${MODE:-pilot}"
TEACHER="${TEACHER:-gemma}"
NUM_CHUNKS="${NUM_CHUNKS:-1}"
CHUNK_IDX="${SLURM_ARRAY_TASK_ID:-0}"

echo "============================================"
echo "MAmmoTH-v2 Target Generation"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID (array task: $CHUNK_IDX/$NUM_CHUNKS)"
echo "Mode: $MODE"
echo "Teacher: $TEACHER"
echo "Start Time: $(date)"
echo "============================================"

mkdir -p logs

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ_DIR"

if [ -f .env ]; then
  source .env
  export HF_TOKEN HUGGINGFACE_HUB_TOKEN
  echo "HF token loaded"
else
  echo "ERROR: .env not found"; exit 1
fi

export HF_HOME="$PWD/cache/huggingface"
export PIP_CACHE_DIR="$PWD/cache/pip"
export XDG_CACHE_HOME="$PWD/cache"
export TMPDIR="$PWD/cache/tmp"
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$TMPDIR"

export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=0

# ── Local NVMe staging (avoid Lustre mmap stalls during model load) ─────────
# PSC Bridges-2: $LOCAL is node-local NVMe, only available inside SLURM jobs.
# rsync with retry loop is PSC's recommended pattern for copying to $LOCAL.
LOCAL_STAGE="${LOCAL:-/local}/${SLURM_JOB_ID}_mammoth"
mkdir -p "$LOCAL_STAGE/huggingface/hub"

SRC_HUB="$PWD/cache/huggingface/hub"
TEACHER_CACHE="models--google--gemma-3-27b-it"
JUDGE_CACHE="models--AdaptLLM--food-Llama-3.2-11B-Vision-Instruct"

echo "Staging model caches to local NVMe ($LOCAL_STAGE)..."
for MODEL_DIR in "$TEACHER_CACHE" "$JUDGE_CACHE"; do
  if [ -d "$SRC_HUB/$MODEL_DIR" ]; then
    echo "  Syncing $MODEL_DIR ..."
    RC=1; n=0
    while [[ $RC -ne 0 && $n -lt 5 ]]; do
      rsync -a "$SRC_HUB/$MODEL_DIR" "$LOCAL_STAGE/huggingface/hub/"
      RC=$?
      let n=n+1
      [[ $RC -ne 0 ]] && echo "    rsync attempt $n failed (rc=$RC), retrying in 10s..." && sleep 10
    done
    if [[ $RC -eq 0 ]]; then
      echo "    Done ($(du -sh "$LOCAL_STAGE/huggingface/hub/$MODEL_DIR" | cut -f1))"
    else
      echo "    WARNING: rsync failed after $n attempts, falling back to shared FS"
    fi
  else
    echo "  SKIP $MODEL_DIR (not cached)"
  fi
done
echo "NVMe staging complete."

# Cleanup local staging on exit
trap 'echo "Cleaning up $LOCAL_STAGE"; rm -rf "$LOCAL_STAGE"' EXIT

# Point HF_HOME at local NVMe copy
export HF_HOME="$LOCAL_STAGE/huggingface"
# ────────────────────────────────────────────────────────────────────────────

IMAGE="${CONTAINER_SIF:-$PROJ_DIR/containers/flash_attn.sif}"
VENV_DIR="/workspace/sensory_env_ngc"
SING_EXEC=(apptainer exec --nv --writable-tmpfs --cleanenv --home /tmp
  --bind "$PWD:/workspace"
  --bind "$LOCAL_STAGE:/local_stage"
  --pwd /workspace
  --env HF_TOKEN="$HF_TOKEN"
  --env HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
  --env HF_HOME="/local_stage/huggingface"
  --env PIP_CACHE_DIR="/workspace/cache/pip"
  --env XDG_CACHE_HOME="/workspace/cache"
  --env TMPDIR="/workspace/cache/tmp"
  --env PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF"
  --env TOKENIZERS_PARALLELISM="$TOKENIZERS_PARALLELISM"
  --env SAFETENSORS_FAST_GPU="1"
  "$IMAGE")

VENV_SETUP='
set -euo pipefail
if [ -e /usr/local/cuda/compat/lib/libcuda.so.1 ] && [ ! -e /usr/local/cuda/compat/lib/libcuda.so ]; then
  ln -s /usr/local/cuda/compat/lib/libcuda.so.1 /usr/local/cuda/compat/lib/libcuda.so || true
fi
export LD_LIBRARY_PATH="/usr/local/cuda/compat/lib:${LD_LIBRARY_PATH:-}"
if [ ! -d "'"$VENV_DIR"'" ]; then
  python3 -m venv --system-site-packages "'"$VENV_DIR"'"
fi
source "'"$VENV_DIR"'/bin/activate"
python -m pip install -U pip
NEED_INSTALL=0
python - <<'"'"'PY'"'"' || NEED_INSTALL=1
import importlib
for m in ["transformers", "accelerate", "peft", "bitsandbytes", "pandas", "PIL", "sklearn"]:
  importlib.import_module("PIL" if m == "PIL" else m)
PY
if [ ! -f "'"$VENV_DIR"'/.installed_mammoth" ] || [ "$NEED_INSTALL" -ne 0 ]; then
  pip install "numpy<2"
  pip install "transformers<5" accelerate peft bitsandbytes pandas pillow tqdm scikit-learn
  touch "'"$VENV_DIR"'/.installed_mammoth"
fi
python - <<'"'"'FX'"'"'
import torch, flash_attn
print("  torch:", torch.__version__, "flash_attn:", flash_attn.__version__)
print("  cuda:", torch.cuda.is_available(), "devices:", torch.cuda.device_count())
FX
'

# Select teacher model
if [ "$TEACHER" = "qwen" ]; then
  MODEL_NAME="Qwen/Qwen2.5-VL-32B-Instruct"
else
  MODEL_NAME="google/gemma-3-27b-it"
fi

# Select mode-specific args
HUMAN_CSV="data/FINAL_DATASET_COMPLETE_with_rescaling.csv"
IMAGE_DIR="data/Images"

if [ "$MODE" = "pilot" ]; then
  OUTPUT="mammoth_v2_pilot_${TEACHER}_${SLURM_JOB_ID}.json"
  EXTRA_ARGS="--max_images 50 --pilot_compare"
  echo "PILOT: 50 images, output=$OUTPUT"
else
  if [ "$NUM_CHUNKS" -gt 1 ]; then
    OUTPUT="mammoth_v2_chunk_${CHUNK_IDX}_of_${NUM_CHUNKS}_${TEACHER}.json"
    EXTRA_ARGS="--resume --chunk_idx $CHUNK_IDX --num_chunks $NUM_CHUNKS"
    echo "FULL (chunk $CHUNK_IDX/$NUM_CHUNKS): output=$OUTPUT"
  else
    OUTPUT="mammoth_style_target_lookup_v2_${TEACHER}.json"
    EXTRA_ARGS="--resume"
    echo "FULL: all images, output=$OUTPUT"
  fi
fi

echo "  Model: $MODEL_NAME"
echo "  Precision: bf16"
echo "  Output: $OUTPUT"
echo "============================================"

"${SING_EXEC[@]}" bash -lc "
$VENV_SETUP
python precompute_targets.py \
  --human_csv '$HUMAN_CSV' \
  --image_dir '$IMAGE_DIR' \
  --output '$OUTPUT' \
  --model_name '$MODEL_NAME' \
  --precision bf16 \
  --max_new_tokens 768 \
  --retries 2 \
  --min_chars_per_sense 100 \
  --save_every 25 \
  $EXTRA_ARGS
"

STATUS=$?
echo ""
if [ $STATUS -eq 0 ]; then
  echo "============================================"
  echo "MAMMOTH-V2 GENERATION COMPLETED SUCCESSFULLY"
  echo "============================================"
  echo "Output: $OUTPUT"
  echo "Teacher: $MODEL_NAME ($TEACHER)"
  echo "Mode: $MODE"
else
  echo "============================================"
  echo "GENERATION FAILED (exit code: $STATUS)"
  echo "============================================"
fi

echo "End Time: $(date)"
