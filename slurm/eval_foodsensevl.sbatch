#!/bin/bash
#SBATCH --job-name=eval_v2
#SBATCH --partition=ai             # UPDATE: set your SLURM partition
#SBATCH --account=soc250046-ai    # UPDATE: set your SLURM account
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=80G
#SBATCH --export=ALL
#SBATCH --output=logs/eval_v2_%j.out
#SBATCH --error=logs/eval_v2_%j.err

# =============================================================================
# Evaluate v2 checkpoints on the 438-image test split (paper methodology)
# Uses USER_PROMPT (same as v2 training), max_new_tokens=1024
# NVMe local staging for fast model load
#
# Usage:
#   ADAPTER_DIR=checkpoints/gemma3_qlora_human_v2_38390266/checkpoint-200 \
#   OUTPUT_DIR=eval_outputs/v2_s1_ckpt200 \
#   sbatch eval_sensory_ours_v2.sbatch
# =============================================================================

set -euo pipefail

ADAPTER_DIR="${ADAPTER_DIR:-${1:-}}"
OUTPUT_DIR="${OUTPUT_DIR:-${2:-}}"
BASE_MODEL="${BASE_MODEL:-google/gemma-3-27b-it}"
SPLIT="${SPLIT:-test}"

if [ -z "$ADAPTER_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
  echo "ERROR: missing required args"
  echo "  ADAPTER_DIR='$ADAPTER_DIR'"
  echo "  OUTPUT_DIR='$OUTPUT_DIR'"
  echo "Usage (env):"
  echo "  sbatch --export=ALL,ADAPTER_DIR=checkpoints/.../checkpoint-200,OUTPUT_DIR=eval_outputs/v2_s1_ckpt200 eval_sensory_ours_v2.sbatch"
  echo "Usage (positional):"
  echo "  sbatch eval_sensory_ours_v2.sbatch checkpoints/.../checkpoint-200 eval_outputs/v2_s1_ckpt200"
  exit 1
fi

if [ ! -d "$ADAPTER_DIR" ]; then
  echo "ERROR: adapter dir not found: $ADAPTER_DIR"
  exit 1
fi

# Derive checkpoint name for logging
CKPT_PARENT=$(basename "$(dirname "$ADAPTER_DIR")")
CKPT_NAME=$(basename "$ADAPTER_DIR")
CKPT_LABEL="${CKPT_PARENT}/${CKPT_NAME}"

echo "============================================"
echo "  Sensory Eval V2 — Test Split (438 images)"
echo "============================================"
echo "  Job ID       : $SLURM_JOB_ID"
echo "  Node         : $SLURMD_NODENAME"
echo "  Checkpoint   : $CKPT_LABEL"
echo "  Adapter Dir  : $ADAPTER_DIR"
echo "  Base Model   : $BASE_MODEL"
echo "  Split        : $SPLIT"
echo "  Output Dir   : $OUTPUT_DIR"
echo "  Prompt       : USER_PROMPT"
echo "  Max Tokens   : 1024"
echo "  Quantization : 4-bit QLoRA (NF4, bf16 compute)"
echo "  Start Time   : $(date)"
echo "============================================"

mkdir -p logs

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ_DIR"

if [ -f .env ]; then
  source .env
  export HF_TOKEN
  export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
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
LOCAL_STAGE="${LOCAL:-/local}/${SLURM_JOB_ID}_eval"
mkdir -p "$LOCAL_STAGE/huggingface/hub"

SRC_HUB="$PWD/cache/huggingface/hub"
MODEL_CACHE="models--google--gemma-3-27b-it"

echo "Staging model cache to local NVMe ($LOCAL_STAGE)..."
if [ -d "$SRC_HUB/$MODEL_CACHE" ]; then
  echo "  Syncing $MODEL_CACHE ..."
  RC=1; n=0
  while [[ $RC -ne 0 && $n -lt 5 ]]; do
    rsync -a "$SRC_HUB/$MODEL_CACHE" "$LOCAL_STAGE/huggingface/hub/"
    RC=$?
    let n=n+1
    [[ $RC -ne 0 ]] && echo "    rsync attempt $n failed (rc=$RC), retrying in 10s..." && sleep 10
  done
  if [[ $RC -eq 0 ]]; then
    echo "    Done ($(du -sh "$LOCAL_STAGE/huggingface/hub/$MODEL_CACHE" | cut -f1))"
  else
    echo "    WARNING: rsync failed after $n attempts, falling back to shared FS"
  fi
else
  echo "  SKIP $MODEL_CACHE (not cached)"
fi
echo "NVMe staging complete."

# Cleanup local staging on exit
trap 'echo "Cleaning up $LOCAL_STAGE"; rm -rf "$LOCAL_STAGE"' EXIT

export HF_HOME="$LOCAL_STAGE/huggingface"

# USER_PROMPT (identical to USER_PROMPT in config/prompts.py)
PROMPT="Analyze this food image and estimate the likely sensory experience. Plain text only (no markdown; no bullets; no asterisks). Provide a Sensory Assessment with 4 sections, one each for Taste, Smell, Texture, and Sound, formatted as: Sense (X.X/5.0): 3-4 sentences of detailed visual justification."

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
python -m pip install -U pip -q
NEED_INSTALL=0
python - <<'"'"'PY'"'"' || NEED_INSTALL=1
import importlib
for m in ["transformers", "accelerate", "peft", "bitsandbytes", "pandas", "PIL", "sklearn"]:
  importlib.import_module("PIL" if m == "PIL" else m)
PY
if [ ! -f "'"$VENV_DIR"'/.installed_mammoth" ] || [ "$NEED_INSTALL" -ne 0 ]; then
  pip install "numpy<2"
  pip install "transformers<5" accelerate peft bitsandbytes pandas pillow tqdm scikit-learn scipy
  touch "'"$VENV_DIR"'/.installed_mammoth"
fi
python - <<'"'"'FX'"'"'
import torch, flash_attn
print("  torch:", torch.__version__, "flash_attn:", flash_attn.__version__)
print("  cuda:", torch.cuda.is_available(), "devices:", torch.cuda.device_count())
FX
'

echo ""
echo "Running evaluation on $SPLIT split..."
echo "  Checkpoint: $CKPT_LABEL"
echo ""
"${SING_EXEC[@]}" bash -lc "
$VENV_SETUP
python -u evaluate.py \
  --adapter_dir '$ADAPTER_DIR' \
  --model ours \
  --split '$SPLIT' \
  --output_dir '$OUTPUT_DIR' \
  --prompt '$PROMPT' \
  --max_new_tokens 1024 \
  --local_files_only \
  --strict
"

STATUS=$?
echo ""
if [ $STATUS -eq 0 ]; then
  echo "============================================"
  echo "  EVAL COMPLETED SUCCESSFULLY"
  echo "  Checkpoint: $CKPT_LABEL"
  echo "  Output: $OUTPUT_DIR"
  echo "============================================"
else
  echo "============================================"
  echo "  EVAL FAILED (exit code: $STATUS)"
  echo "  Checkpoint: $CKPT_LABEL"
  echo "============================================"
fi

echo "End Time: $(date)"
