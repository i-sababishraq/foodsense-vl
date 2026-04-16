#!/bin/bash
#SBATCH --job-name=eval_baseline
#SBATCH --partition=ai             # UPDATE: set your SLURM partition
#SBATCH --account=soc250046-ai    # UPDATE: set your SLURM account
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=80G
#SBATCH --export=ALL
#SBATCH --output=logs/eval_baseline_%j.out
#SBATCH --error=logs/eval_baseline_%j.err

# =============================================================================
# BASELINE RE-EVALUATION: Re-run inference for a single baseline model.
#
# Usage:
#   MODEL_NAME=qwen2_vl   sbatch eval_baseline_rerun.sbatch
#   MODEL_NAME=llava       sbatch eval_baseline_rerun.sbatch
#   MODEL_NAME=internvl    sbatch eval_baseline_rerun.sbatch
#   MODEL_NAME=food_llama  sbatch eval_baseline_rerun.sbatch
# =============================================================================

set -euo pipefail

MODEL_NAME="${MODEL_NAME:-}"
if [ -z "$MODEL_NAME" ]; then
  echo "ERROR: MODEL_NAME not set. Usage: MODEL_NAME=qwen2_vl sbatch eval_baseline_rerun.sbatch"
  exit 1
fi

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="eval_outputs/${MODEL_NAME}_reeval"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"

# USER_PROMPT (identical to USER_PROMPT in config/prompts.py)
PROMPT="Analyze this food image and estimate the likely sensory experience. Plain text only (no markdown; no bullets; no asterisks). Provide a Sensory Assessment with 4 sections, one each for Taste, Smell, Texture, and Sound, formatted as: Sense (X.X/5.0): 3-4 sentences of detailed visual justification."

# Model → HF cache dir name mapping
declare -A MODEL_CACHE_MAP=(
  [qwen2_vl]="models--Qwen--Qwen2.5-VL-32B-Instruct"
  [llava]="models--llava-hf--llava-v1.6-34b-hf"
  [internvl]="models--OpenGVLab--InternVL2_5-26B"
  [food_llama]="models--AdaptLLM--food-Llama-3.2-11B-Vision-Instruct"
)

MODEL_CACHE_NAME="${MODEL_CACHE_MAP[$MODEL_NAME]:-}"
if [ -z "$MODEL_CACHE_NAME" ]; then
  echo "ERROR: Unknown model '$MODEL_NAME'. Must be one of: qwen2_vl, llava, internvl, food_llama"
  exit 1
fi

# ── Header ───────────────────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  BASELINE RE-EVALUATION — ${MODEL_NAME}                     "
echo "╚══════════════════════════════════════════════════════════════╝"
echo "  Job ID       : $SLURM_JOB_ID"
echo "  Node         : $SLURMD_NODENAME"
echo "  Start Time   : $(date)"
echo "  Model        : $MODEL_NAME"
echo "  Output Dir   : $OUTPUT_DIR"
echo "  Prompt       : USER_PROMPT"
echo "  Max Tokens   : $MAX_NEW_TOKENS"
echo ""

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null \
  | sed 's/^/  /' || echo "  nvidia-smi not available"
echo ""

mkdir -p logs
cd "$PROJ_DIR"

if [ -f .env ]; then
  source .env
  export HF_TOKEN
  export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
  echo "  HF token loaded"
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
export PYTHONUNBUFFERED=1

# ── NVMe staging ──────────────────────────────────────────────────────────
LOCAL_STAGE="${LOCAL:-/local}/${SLURM_JOB_ID}_eval"
SRC_CACHE="$PWD/cache/huggingface/hub/$MODEL_CACHE_NAME"

cleanup_local() {
  echo "Cleaning up NVMe staging: $LOCAL_STAGE"
  rm -rf "$LOCAL_STAGE" 2>/dev/null || true
}
trap cleanup_local EXIT

STAGE_START=$SECONDS
if [ -d "$SRC_CACHE" ]; then
  echo "Staging model cache to NVMe: $LOCAL_STAGE"
  mkdir -p "$LOCAL_STAGE/huggingface/hub"
  for attempt in 1 2 3 4 5; do
    rsync -a "$SRC_CACHE" "$LOCAL_STAGE/huggingface/hub/" && break
    echo "  rsync attempt $attempt failed, retrying in 10s..."
    sleep 10
  done
  STAGE_ELAPSED=$(( SECONDS - STAGE_START ))
  echo "NVMe staging complete ($(du -sh "$LOCAL_STAGE/huggingface/hub/$MODEL_CACHE_NAME" | cut -f1)) in ${STAGE_ELAPSED}s"
  export HF_HOME="$LOCAL_STAGE/huggingface"
else
  echo "WARNING: Model cache not found at $SRC_CACHE — will download from HF Hub"
fi

# ── Singularity exec ────────────────────────────────────────────────────────
IMAGE="${CONTAINER_SIF:-$PROJ_DIR/containers/flash_attn.sif}"
VENV_ROOT="/workspace/.venvs"
VENV_DIR="$VENV_ROOT/sensory_${MODEL_NAME}"

# Model-specific transformers pinning for compatibility + reproducibility.
# - internvl: fixed at 4.48.3 (known-good; avoids cache API breakage in newer versions)
# - qwen2_vl: requires >=4.49 support for Qwen2.5-VL classes
# - llava / food_llama: validated on 4.57.6 in this environment
case "$MODEL_NAME" in
  internvl)
    TRANSFORMERS_PIN="4.48.3"
    ;;
  qwen2_vl)
    TRANSFORMERS_PIN="4.57.6"
    ;;
  llava)
    TRANSFORMERS_PIN="4.57.6"
    ;;
  food_llama)
    TRANSFORMERS_PIN="4.57.6"
    ;;
  *)
    TRANSFORMERS_PIN="4.57.6"
    ;;
esac

echo "  Venv         : $VENV_DIR"
echo "  Transformers : $TRANSFORMERS_PIN"
SING_EXEC=(apptainer exec --nv --writable-tmpfs --cleanenv --home /tmp \
  --bind "$PWD:/workspace" \
  --bind "$LOCAL_STAGE:/local_stage" \
  --pwd /workspace \
  --env HF_TOKEN="$HF_TOKEN" \
  --env HUGGINGFACE_HUB_TOKEN="$HF_TOKEN" \
  --env HF_HOME="/local_stage/huggingface" \
  --env PIP_CACHE_DIR="/workspace/cache/pip" \
  --env XDG_CACHE_HOME="/workspace/cache" \
  --env TMPDIR="/workspace/cache/tmp" \
  --env PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
  --env TOKENIZERS_PARALLELISM="$TOKENIZERS_PARALLELISM" \
  --env PYTHONUNBUFFERED="1" \
  --env SAFETENSORS_FAST_GPU="1" \
  "$IMAGE")

VENV_SETUP='
set -euo pipefail
if [ -e /usr/local/cuda/compat/lib/libcuda.so.1 ] && [ ! -e /usr/local/cuda/compat/lib/libcuda.so ]; then
  ln -s /usr/local/cuda/compat/lib/libcuda.so.1 /usr/local/cuda/compat/lib/libcuda.so || true
fi
export LD_LIBRARY_PATH="/usr/local/cuda/compat/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "'$VENV_ROOT'"
if [ ! -d "'$VENV_DIR'" ]; then
  python3 -m venv --system-site-packages "'$VENV_DIR'"
fi
source "'$VENV_DIR'/bin/activate"
python -m pip install -U pip -q
NEED_INSTALL=0
python - <<'"'"'PY'"'"' || NEED_INSTALL=1
import importlib
for m in ["transformers", "accelerate", "peft", "bitsandbytes", "pandas", "PIL", "sklearn"]:
  importlib.import_module("PIL" if m == "PIL" else m)
PY
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
  NEED_INSTALL=1
fi

if [ "$NEED_INSTALL" -ne 0 ]; then
  pip install "numpy<2"
  pip install --upgrade --force-reinstall \
    "transformers==$TARGET_TF" accelerate peft bitsandbytes pandas pillow tqdm scikit-learn scipy
fi
python - <<'"'"'FX0'"'"'
import transformers
print("  transformers:", transformers.__version__)
FX0
python - <<'"'"'FX'"'"'
import torch, flash_attn
print("  torch:", torch.__version__, "flash_attn:", flash_attn.__version__)
print("  cuda:", torch.cuda.is_available(), "devices:", torch.cuda.device_count())
FX
'

# ── Run inference ────────────────────────────────────────────────────────────
# NOTE: --prompt flag passes USER_PROMPT (same as ablation/v2_s2 jobs)
# NOTE: No --adapter_dir needed for baseline models
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Running $MODEL_NAME inference on test split (438 images)..."
echo "════════════════════════════════════════════════════════════════"

EVAL_START=$SECONDS

"${SING_EXEC[@]}" bash -lc "
$VENV_SETUP
python -u evaluate.py \
  --model '$MODEL_NAME' \
  --split test \
  --output_dir '$OUTPUT_DIR' \
  --prompt '$PROMPT' \
  --max_new_tokens $MAX_NEW_TOKENS \
  --local_files_only \
  --strict
"

STATUS=$?
EVAL_ELAPSED=$(( SECONDS - EVAL_START ))
EVAL_HMS=$(printf '%02d:%02d:%02d' $((EVAL_ELAPSED/3600)) $((EVAL_ELAPSED%3600/60)) $((EVAL_ELAPSED%60)))

echo ""
if [ $STATUS -eq 0 ]; then
  echo "╔══════════════════════════════════════════════════════════════╗"
  echo "║  EVAL COMPLETED SUCCESSFULLY — $MODEL_NAME                  "
  echo "╚══════════════════════════════════════════════════════════════╝"
  echo "  Eval wall time: $EVAL_HMS"
  echo ""
  echo "  ── Output Files ──"
  ls -lh "$OUTPUT_DIR"/ 2>/dev/null | tail -n +2 | sed 's/^/  /'
else
  echo "╔══════════════════════════════════════════════════════════════╗"
  echo "║  EVAL FAILED (exit code: $STATUS) — $MODEL_NAME             "
  echo "╚══════════════════════════════════════════════════════════════╝"
fi

echo ""
echo "  End Time     : $(date)"
echo "  Total elapsed: $(printf '%02d:%02d:%02d' $((SECONDS/3600)) $((SECONDS%3600/60)) $((SECONDS%60)))"
