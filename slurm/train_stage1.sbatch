#!/bin/bash
#SBATCH --job-name=qlora_human_v2
#SBATCH --partition=ai             # UPDATE: set your SLURM partition
#SBATCH --account=soc250046-ai    # UPDATE: set your SLURM account
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=80G
#SBATCH --output=logs/qlora_human_v2_%j.out
#SBATCH --error=logs/qlora_human_v2_%j.err

# =============================================================================
# V2 Stage-1: Human grounding (ratings + descriptors only, no mammoth).
# Changes from v1:
#   - save_steps=50 (finer checkpoint selection)
#   - user_prompt_version=v2 (unified prompt for train+inference alignment)
#   - lr=5e-6 (same as v1, conservative for grounding)
# =============================================================================

set -euo pipefail

echo "============================================"
echo "QLoRA V2 Stage-1 — Human Grounding"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
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
LOCAL_STAGE="${LOCAL:-/local}/${SLURM_JOB_ID}_train"
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

# Config
HUMAN_CSV="data/FINAL_DATASET_COMPLETE_with_rescaling.csv"
IMAGE_DIR="data/Images"
OUTPUT_DIR="checkpoints/gemma3_qlora_human_v2_${SLURM_JOB_ID}"
MAX_STEPS="${MAX_STEPS:-300}"
REQUIRE_PROSE_SENSES="${REQUIRE_PROSE_SENSES:-2}"

echo "============================================"
echo "V2 Stage-1 Configuration (human_only)"
echo "============================================"
echo "  GPUs: 1"
echo "  Mode: human_only (ratings + descriptors, no mammoth)"
echo "  Prompt: v2 (unified train+inference)"
echo "  Output: $OUTPUT_DIR"
echo "  max_steps: $MAX_STEPS"
echo "  save_steps: 25 (fine-grained checkpoint selection)"
echo "  require_prose_senses: $REQUIRE_PROSE_SENSES"
echo "  lr: 5e-6"
echo "  batch_size: 2, grad_accum: 32 (effective batch 64)"
echo "  lora_r: 16, lora_alpha: 32"
echo "============================================"

"${SING_EXEC[@]}" bash -lc "
$VENV_SETUP
python train.py \
  --human_csv '$HUMAN_CSV' \
  --image_dir '$IMAGE_DIR' \
  --output_dir '$OUTPUT_DIR' \
  --model_name google/gemma-3-27b-it \
  --epochs 1 --batch_size 2 --grad_accum 32 \
  --lr 5e-6 --warmup_ratio 0.03 --weight_decay 0.05 --max_grad_norm 1.0 \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.15 \
  --val_max_samples 2000 --early_stop_patience 3 \
  --optim adamw_torch \
  --max_steps '$MAX_STEPS' \
  --save_steps 25 \
  --save_total_limit -1 \
  --require_prose_senses '$REQUIRE_PROSE_SENSES' \
  --human_only \
  --user_prompt_version v2
"

STATUS=$?
echo ""
if [ $STATUS -eq 0 ]; then
  echo "============================================"
  echo "V2 STAGE-1 COMPLETED SUCCESSFULLY"
  echo "============================================"
  echo "Outputs saved to: $OUTPUT_DIR"

  "${SING_EXEC[@]}" bash -lc "
$VENV_SETUP
python - <<'PY'
import json
from pathlib import Path

out_dir = Path('$OUTPUT_DIR')
print('--------------------------------------------')
print('Post-train diagnostic: sense_mae + cross-sense proxy')
print('--------------------------------------------')

state_path = out_dir / 'trainer_state.json'
if not state_path.exists():
    ckpt_states = list(out_dir.glob('checkpoint-*/trainer_state.json'))
    if ckpt_states:
        def _ckpt_step(p):
            try:
                return int(p.parent.name.split('-')[-1])
            except Exception:
                return -1
        ckpt_states.sort(key=_ckpt_step)
        state_path = ckpt_states[-1]

if not state_path.exists():
    print(f'trainer_state.json not found under {out_dir}')
    raise SystemExit(0)

print(f'trainer_state_path: {state_path}')
state = json.loads(state_path.read_text())
best_ckpt = state.get('best_model_checkpoint')
best_metric = state.get('best_metric')
print(f'best_model_checkpoint: {best_ckpt}')
print(f'best_sense_mae: {best_metric}')

history = state.get('log_history', [])
eval_rows = [r for r in history if isinstance(r, dict) and 'eval_sense_mae' in r]
if not eval_rows:
    print('No eval_sense_mae rows found in log_history')
    raise SystemExit(0)

last = eval_rows[-1]
print('last_eval_sense_mae:', last.get('eval_sense_mae'))
keys = ['eval_taste_mae', 'eval_smell_mae', 'eval_texture_mae', 'eval_sound_mae']
vals = [last.get(k) for k in keys]
if all(v is not None for v in vals):
    print('per_sense_mae:', {k.replace('eval_', '').replace('_mae', ''): v for k, v in zip(keys, vals)})
    spread = max(vals) - min(vals)
    print('cross_sense_mae_spread_proxy:', spread)
else:
    print('Per-sense eval MAE keys missing in latest eval row')
PY
"

  # --- Checkpoint pruning: keep first 3 + last 5, delete the rest ---
  echo "Pruning checkpoints: keeping first 3 + last 5..."
  CKPTS=($(ls -d $OUTPUT_DIR/checkpoint-* 2>/dev/null | sort -t- -k2 -n))
  NUM_CKPTS=${#CKPTS[@]}
  KEEP_FIRST=3
  KEEP_LAST=5
  if [ "$NUM_CKPTS" -gt $((KEEP_FIRST + KEEP_LAST)) ]; then
    DELETE_START=$KEEP_FIRST
    DELETE_END=$((NUM_CKPTS - KEEP_LAST))
    for (( i=DELETE_START; i<DELETE_END; i++ )); do
      echo "  Removing ${CKPTS[$i]}"
      rm -rf "${CKPTS[$i]}"
    done
    echo "  Kept first $KEEP_FIRST + last $KEEP_LAST = $((KEEP_FIRST + KEEP_LAST)) checkpoints (deleted $((DELETE_END - DELETE_START)))"
  else
    echo "  Only $NUM_CKPTS checkpoints, keeping all."
  fi

  echo ""
  echo "Next: Run Stage-2 with:"
  echo "  STAGE1_DIR=$OUTPUT_DIR/<best-checkpoint> sbatch slurm/train_stage2.sbatch"
else
  echo "============================================"
  echo "TRAINING FAILED (exit code: $STATUS)"
  echo "============================================"
fi

echo "End Time: $(date)"
