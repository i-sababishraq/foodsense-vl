#!/bin/bash
#SBATCH --job-name=qlora_s2_v2
#SBATCH --partition=ai             # UPDATE: set your SLURM partition
#SBATCH --account=soc250046-ai    # UPDATE: set your SLURM account
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=80G
#SBATCH --export=ALL
#SBATCH --output=logs/qlora_stage2_v2_%j.out
#SBATCH --error=logs/qlora_stage2_v2_%j.err

# =============================================================================
# V2 Stage-2: MAmmoTH-v2 expansion on top of human-grounded Stage-1.
# Changes from v1:
#   - max_steps: 600-800 (longer training for richer targets)
#   - max_length: 3072 (sufficient for v2 prose targets)
#   - batch_size: 1, grad_accum: 64 (memory safety at longer seqlen)
#   - lr: 2e-6 (gentler to preserve Stage-1 grounding)
#   - save_steps: 50 (finer checkpoint selection)
#   - save_total_limit: -1 (keep all checkpoints)
#   - user_prompt_version: v2 (unified train+inference prompt)
#   - fixed_sense_order: true (Taste,Smell,Texture,Sound — match inference)
#   - require_prose_senses: 4 (all senses must have rich prose)
#   - Uses mammoth_style_target_lookup_v2_*.json (regenerated targets)
# =============================================================================

set -euo pipefail

# Stage-1 v2 source checkpoint (override via env var)
STAGE1_DIR="${STAGE1_DIR:-}"
# MAmmoTH-v2 targets (override via env var — select gemma or qwen winner)
MAMMOTH_V2="${MAMMOTH_V2:-mammoth_style_target_lookup_v2_gemma.json}"
# Training steps
MAX_STEPS="${MAX_STEPS:-800}"

if [ -z "$STAGE1_DIR" ]; then
  echo "ERROR: STAGE1_DIR not set."
  echo "  export STAGE1_DIR=checkpoints/gemma3_qlora_human_v2_<JOB_ID>/checkpoint-<N>"
  echo "  export MAMMOTH_V2=mammoth_style_target_lookup_v2_gemma.json  (or _qwen)"
  echo "  sbatch slurm/train_stage2.sbatch"
  exit 1
fi

echo "============================================"
echo "QLoRA V2 Stage-2 — MAmmoTH-v2 Expansion"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Stage 1 checkpoint: $STAGE1_DIR"
echo "MAmmoTH-v2 targets: $MAMMOTH_V2"
echo "Max steps: $MAX_STEPS"
echo "Start Time: $(date)"
echo "============================================"

if [ ! -d "$STAGE1_DIR" ]; then
  echo "ERROR: STAGE1_DIR does not exist: $STAGE1_DIR"
  exit 1
fi
if [ ! -f "$MAMMOTH_V2" ]; then
  echo "ERROR: MAMMOTH_V2 file does not exist: $MAMMOTH_V2"
  exit 1
fi

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

# ── NVMe local staging for fast model shard loading ──────────────
LOCAL_STAGE="${LOCAL:-/local}/${SLURM_JOB_ID}_s2train"
MODEL_CACHE_NAME="models--google--gemma-3-27b-it"
SRC_CACHE="$PWD/cache/huggingface/hub/$MODEL_CACHE_NAME"

cleanup_local() { rm -rf "$LOCAL_STAGE" 2>/dev/null || true; }
trap cleanup_local EXIT

if [ -d "$SRC_CACHE" ]; then
  echo "Staging model cache to NVMe: $LOCAL_STAGE"
  mkdir -p "$LOCAL_STAGE/hub/$MODEL_CACHE_NAME"
  for attempt in 1 2 3; do
    rsync -a --info=progress2 "$SRC_CACHE/" "$LOCAL_STAGE/hub/$MODEL_CACHE_NAME/" && break
    echo "  rsync attempt $attempt failed, retrying in 10s..."
    sleep 10
  done
  echo "Stage complete ($(du -sh "$LOCAL_STAGE" | cut -f1))"
  export HF_HOME="$LOCAL_STAGE"
else
  echo "WARNING: Model cache not found at $SRC_CACHE — using Lustre directly"
fi

IMAGE="${CONTAINER_SIF:-$PROJ_DIR/containers/flash_attn.sif}"
VENV_DIR="/workspace/sensory_env_ngc"
SING_EXEC=(apptainer exec --nv --writable-tmpfs --cleanenv --home /tmp
  --bind "$PWD:/workspace"
  --bind "$LOCAL_STAGE:/local_cache"
  --pwd /workspace
  --env HF_TOKEN="$HF_TOKEN"
  --env HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
  --env HF_HOME="/local_cache"
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
OUTPUT_DIR="checkpoints/gemma3_qlora_stage2_v2_${SLURM_JOB_ID}"

echo "============================================"
echo "V2 Stage-2 Configuration"
echo "============================================"
echo "  GPUs: 1"
echo "  Mode: Stage-2 (human desc + MAmmoTH-v2 expansion)"
echo "  Prompt: v2 (unified train+inference)"
echo "  Fixed sense order: yes (Taste,Smell,Texture,Sound)"
echo "  Output: $OUTPUT_DIR"
echo "  max_steps: $MAX_STEPS"
echo "  max_length: 3072"
echo "  lr: 2e-6 (gentler, longer training)"
echo "  batch_size: 1, grad_accum: 64 (effective batch 64)"
echo "  save_steps: 50, save_total_limit: -1 (keep all)"
echo "  require_prose_senses: 4"
echo "  lora_r: 16, lora_alpha: 32"
echo "============================================"

"${SING_EXEC[@]}" bash -lc "
$VENV_SETUP
python train.py \
  --human_csv '$HUMAN_CSV' \
  --image_dir '$IMAGE_DIR' \
  --output_dir '$OUTPUT_DIR' \
  --model_name google/gemma-3-27b-it \
  --epochs 1 --batch_size 1 --grad_accum 64 \
  --lr 2e-6 --warmup_ratio 0.03 --weight_decay 0.05 --max_grad_norm 1.0 \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.15 \
  --val_max_samples 2000 --early_stop_patience 5 \
  --optim adamw_torch \
  --max_steps '$MAX_STEPS' \
  --max_length 3072 \
  --save_steps 25 \
  --save_total_limit -1 \
  --require_prose_senses 4 \
  --mammoth_targets '$MAMMOTH_V2' \
  --stage2_from '$STAGE1_DIR' \
  --user_prompt_version v2 \
  --fixed_sense_order
"

STATUS=$?
echo ""
if [ $STATUS -eq 0 ]; then
  echo "============================================"
  echo "V2 STAGE-2 COMPLETED SUCCESSFULLY"
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

  echo ""
  echo "Next: Evaluate on the test split:"
  echo "  ADAPTER_DIR=$OUTPUT_DIR/<best-checkpoint> OUTPUT_DIR=eval_outputs/stage2_eval sbatch slurm/eval_foodsensevl.sbatch"
  echo "  Or quick demo (5 images): python inference/foodsensevl.py --adapter_dir $OUTPUT_DIR/<best-checkpoint>"
else
  echo "============================================"
  echo "TRAINING FAILED (exit code: $STATUS)"
  echo "============================================"
fi

echo "End Time: $(date)"
