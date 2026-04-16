#!/bin/bash
# =============================================================================
# FoodSense-VL — Environment Setup (Anvil Cluster)
# =============================================================================
# Creates all three environments described in CONTAINER_AND_ENVIRONMENT_GUIDE.md:
#
#   1. sensory_env_ngc       — Main GPU venv (Gemma/Qwen/LLaVA/Food-LLaMA)
#   2. .venvs/sensory_internvl_compat — InternVL-isolated venv (pinned transformers)
#   3. sensory_env           — Local benchmark venv (login node, no GPU)
#
# Prerequisites:
#   - Run from the project root (the directory containing this scripts/ folder)
#   - Container available at: flash_attn.sif (symlinked in Step 0)
#   - For GPU venvs (1 & 2): must be run inside a SLURM job with GPU access
#     OR interactively via `salloc --gpus=1 ...`
#   - For local venv (3): can be run on login node
#
# Usage:
#   # Setup everything (interactive GPU session recommended):
#   salloc --partition=gpu --gpus=1 --cpus-per-task=4 --mem=16G --time=01:00:00
#   bash scripts/setup_envs.sh all
#
#   # Setup individual environments:
#   bash scripts/setup_envs.sh main       # GPU venv only
#   bash scripts/setup_envs.sh internvl   # InternVL venv only
#   bash scripts/setup_envs.sh local      # Local benchmark venv only
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# Container paths
EXISTING_SIF="${CONTAINER_SIF:-}"  # Set CONTAINER_SIF env var if your .sif is outside the project
LOCAL_SIF="$PROJECT_ROOT/containers/flash_attn.sif"

# ─── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*"; }

# ─── Step 0: Container Setup ────────────────────────────────────────────────
setup_container() {
  info "Step 0: Setting up container..."

  mkdir -p "$PROJECT_ROOT/containers"

  if [ -f "$LOCAL_SIF" ]; then
    ok "Container already exists at $LOCAL_SIF"
  elif [ -f "$EXISTING_SIF" ]; then
    info "Symlinking existing container..."
    ln -sf "$EXISTING_SIF" "$LOCAL_SIF"
    ok "Container symlinked: $LOCAL_SIF → $EXISTING_SIF"
  else
    err "No container found!"
    err "  Expected: $EXISTING_SIF"
    err "  Build one with: cd containers/ && sudo apptainer build vlm_flash_attn.sif vlm_flash_attn_local.def"
    return 1
  fi

  echo ""
}

# ─── Step 1: Main GPU venv (sensory_env_ngc) ────────────────────────────────
setup_main_venv() {
  info "Step 1: Creating main GPU venv (sensory_env_ngc)..."
  info "  This venv is used for Gemma, Qwen, LLaVA, and Food-LLaMA paths."

  if [ ! -f "$LOCAL_SIF" ]; then
    err "Container not found at $LOCAL_SIF — run 'bash scripts/setup_envs.sh container' first"
    return 1
  fi

  # Set up cache dirs
  mkdir -p "$PROJECT_ROOT/cache/pip" "$PROJECT_ROOT/cache/tmp"
  export PIP_CACHE_DIR="$PROJECT_ROOT/cache/pip"
  export TMPDIR="$PROJECT_ROOT/cache/tmp"

  apptainer exec --nv --writable-tmpfs --cleanenv --home /tmp \
    --bind "$PROJECT_ROOT:/workspace" \
    --pwd /workspace \
    --env PIP_CACHE_DIR="/workspace/cache/pip" \
    --env TMPDIR="/workspace/cache/tmp" \
    "$LOCAL_SIF" bash -lc '
set -euo pipefail

VENV_DIR="/workspace/sensory_env_ngc"

# Fix CUDA compat symlink (some containers need this)
if [ -e /usr/local/cuda/compat/lib/libcuda.so.1 ] && [ ! -e /usr/local/cuda/compat/lib/libcuda.so ]; then
  ln -s /usr/local/cuda/compat/lib/libcuda.so.1 /usr/local/cuda/compat/lib/libcuda.so || true
fi
export LD_LIBRARY_PATH="/usr/local/cuda/compat/lib:${LD_LIBRARY_PATH:-}"

# Create venv with system site packages (inherits torch, flash-attn from container)
if [ ! -d "$VENV_DIR" ]; then
  echo "  Creating venv at $VENV_DIR..."
  python3 -m virtualenv --system-site-packages "$VENV_DIR" || python3 -m venv --system-site-packages "$VENV_DIR" || {
    echo "  Fallback to venv without pip..."
    python3 -m venv --without-pip --system-site-packages "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    wget -q https://bootstrap.pypa.io/get-pip.py -O /tmp/get-pip.py
    python3 /tmp/get-pip.py
  }
else
  echo "  Venv already exists at $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install -U pip -q

echo "  Installing project dependencies..."
pip install -q "numpy<2" \
  "transformers>=4.57" \
  accelerate \
  peft \
  bitsandbytes \
  pandas \
  pillow \
  tqdm \
  scikit-learn \
  scipy \
  sentencepiece \
  protobuf

# Mark as installed
touch "$VENV_DIR/.installed_foodsense"

echo ""
echo "  ── Preflight check ──"
python - <<PY
import importlib
missing = []
for m in ["transformers", "accelerate", "peft", "bitsandbytes", "pandas", "PIL", "sklearn", "scipy"]:
    try:
        importlib.import_module("PIL" if m == "PIL" else m)
        print(f"    ✓ {m}")
    except ImportError:
        print(f"    ✗ {m} MISSING")
        missing.append(m)
if missing:
    raise SystemExit(f"Missing packages: {missing}")
PY

echo ""
echo "  ── CUDA / Flash-Attn check ──"
python - <<FX
import torch, flash_attn
print(f"    torch:      {torch.__version__}")
print(f"    flash_attn: {flash_attn.__version__}")
print(f"    cuda:       {torch.cuda.is_available()}")
print(f"    devices:    {torch.cuda.device_count()}")
FX

echo ""
echo "  ✓ sensory_env_ngc setup complete"
'

  ok "Main GPU venv created at: $PROJECT_ROOT/sensory_env_ngc/"
  echo ""
}

# ─── Step 2: InternVL venv (.venvs/sensory_internvl_compat) ─────────────────
setup_internvl_venv() {
  info "Step 2: Creating InternVL-isolated venv (.venvs/sensory_internvl_compat)..."
  info "  This venv pins transformers==4.48.3 for InternVL compatibility."

  if [ ! -f "$LOCAL_SIF" ]; then
    err "Container not found at $LOCAL_SIF — run 'bash scripts/setup_envs.sh container' first"
    return 1
  fi

  mkdir -p "$PROJECT_ROOT/cache/pip" "$PROJECT_ROOT/cache/tmp"
  export PIP_CACHE_DIR="$PROJECT_ROOT/cache/pip"
  export TMPDIR="$PROJECT_ROOT/cache/tmp"

  apptainer exec --nv --writable-tmpfs --cleanenv --home /tmp \
    --bind "$PROJECT_ROOT:/workspace" \
    --pwd /workspace \
    --env PIP_CACHE_DIR="/workspace/cache/pip" \
    --env TMPDIR="/workspace/cache/tmp" \
    "$LOCAL_SIF" bash -lc '
set -euo pipefail

VENV_DIR="/workspace/.venvs/sensory_internvl_compat"

# Fix CUDA compat symlink
if [ -e /usr/local/cuda/compat/lib/libcuda.so.1 ] && [ ! -e /usr/local/cuda/compat/lib/libcuda.so ]; then
  ln -s /usr/local/cuda/compat/lib/libcuda.so.1 /usr/local/cuda/compat/lib/libcuda.so || true
fi
export LD_LIBRARY_PATH="/usr/local/cuda/compat/lib:${LD_LIBRARY_PATH:-}"

mkdir -p /workspace/.venvs

if [ ! -d "$VENV_DIR" ]; then
  echo "  Creating venv at $VENV_DIR..."
  python3 -m virtualenv --system-site-packages "$VENV_DIR" || python3 -m venv --system-site-packages "$VENV_DIR" || {
    echo "  Fallback to venv without pip..."
    python3 -m venv --without-pip --system-site-packages "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    wget -q https://bootstrap.pypa.io/get-pip.py -O /tmp/get-pip.py
    python3 /tmp/get-pip.py
  }
else
  echo "  Venv already exists at $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install -U pip -q

echo "  Installing InternVL-pinned dependencies..."
pip install -q "numpy<2" \
  "transformers==4.48.3" \
  "tokenizers<0.22" \
  accelerate \
  peft \
  bitsandbytes \
  pandas \
  pillow \
  tqdm \
  scikit-learn \
  scipy \
  einops

touch "$VENV_DIR/.installed_internvl"

echo ""
echo "  ── Preflight check ──"
python - <<PY
import importlib
missing = []
for m in ["transformers", "accelerate", "peft", "bitsandbytes", "pandas", "PIL", "sklearn", "scipy", "einops"]:
    try:
        mod = importlib.import_module("PIL" if m == "PIL" else m)
        print(f"    ✓ {m}")
    except ImportError:
        print(f"    ✗ {m} MISSING")
        missing.append(m)
if missing:
    raise SystemExit(f"Missing packages: {missing}")
PY

echo ""
echo "  ── Version check ──"
python - <<VER
import torch, transformers, flash_attn
print(f"    transformers: {transformers.__version__}")
print(f"    torch:        {torch.__version__}")
print(f"    flash_attn:   {flash_attn.__version__}")
print(f"    cuda:         {torch.cuda.is_available()}")
print(f"    devices:      {torch.cuda.device_count()}")
assert transformers.__version__.startswith("4.48"), \
    f"WRONG transformers version: {transformers.__version__} (expected 4.48.x)"
print("    ✓ transformers version correct (4.48.x)")
VER

echo ""
echo "  ✓ sensory_internvl_compat setup complete"
'

  ok "InternVL venv created at: $PROJECT_ROOT/.venvs/sensory_internvl_compat/"
  echo ""
}

# ─── Step 3: Local benchmark venv (no GPU needed) ───────────────────────────
setup_local_venv() {
  info "Step 3: Creating local benchmark venv (sensory_env)..."
  info "  This venv is for running benchmarks on the login node (no GPU)."

  VENV_DIR="$PROJECT_ROOT/sensory_env"

  # Find Python 3.10+
  PYTHON_BIN=""
  for py in python3.10 python3.11 python3.12 python3; do
    if command -v "$py" &>/dev/null; then
      PY_VER=$("$py" --version 2>&1 | grep -oP '\d+\.\d+')
      # Check >= 3.10
      PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
      PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
      if [ "$PY_MAJOR" -ge 3 ] && [ "$PY_MINOR" -ge 10 ]; then
        PYTHON_BIN="$py"
        break
      fi
    fi
  done

  # Also check the existing conda env
  CONDA_PYTHON="/anvil/projects/x-soc250046/x-sishraq/sensory_env/bin/python3.10"
  if [ -z "$PYTHON_BIN" ] && [ -x "$CONDA_PYTHON" ]; then
    PYTHON_BIN="$CONDA_PYTHON"
  fi

  if [ -z "$PYTHON_BIN" ]; then
    err "No Python 3.10+ found!"
    err "  Try: module load anaconda/2024.02-py311"
    return 1
  fi

  info "Using Python: $PYTHON_BIN ($($PYTHON_BIN --version 2>&1))"

  if [ ! -d "$VENV_DIR" ]; then
    echo "  Creating venv at $VENV_DIR..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  else
    echo "  Venv already exists at $VENV_DIR"
  fi

  source "$VENV_DIR/bin/activate"
  pip install -U pip -q

  echo "  Installing benchmark dependencies..."
  pip install -q "numpy<2" pandas scipy scikit-learn tqdm

  deactivate

  ok "Local benchmark venv created at: $VENV_DIR/"
  echo "  Activate with: source $VENV_DIR/bin/activate"
  echo ""
}

# ─── Step 4: Cache directories ──────────────────────────────────────────────
setup_cache() {
  info "Step 4: Setting up cache directories..."

  mkdir -p "$PROJECT_ROOT/cache/huggingface"
  mkdir -p "$PROJECT_ROOT/cache/pip"
  mkdir -p "$PROJECT_ROOT/cache/tmp"
  mkdir -p "$PROJECT_ROOT/logs"

  ok "Cache directories created:"
  echo "    cache/huggingface/  — HF model cache"
  echo "    cache/pip/          — pip download cache"
  echo "    cache/tmp/          — temp files"
  echo "    logs/               — SLURM job logs"
  echo ""
}

# ─── Step 5: .env file ──────────────────────────────────────────────────────
setup_env_file() {
  if [ ! -f "$PROJECT_ROOT/.env" ]; then
    info "Creating .env from template..."
    cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
    warn "Edit .env and add your HF_TOKEN: nano .env"
  else
    ok ".env file already exists"
  fi
  echo ""
}

# ─── Main ────────────────────────────────────────────────────────────────────
print_usage() {
  echo "Usage: bash scripts/setup_envs.sh <target>"
  echo ""
  echo "Targets:"
  echo "  all        — Set up everything (container + all 3 venvs + cache + .env)"
  echo "  container  — Symlink/setup the container only"
  echo "  main       — Create main GPU venv (sensory_env_ngc) — requires GPU"
  echo "  internvl   — Create InternVL venv — requires GPU"
  echo "  local      — Create local benchmark venv (no GPU needed)"
  echo "  cache      — Create cache directories only"
  echo "  envfile    — Create .env from template"
  echo ""
  echo "Recommended first-time setup:"
  echo "  1. On login node:  bash scripts/setup_envs.sh container"
  echo "  2. On login node:  bash scripts/setup_envs.sh cache"
  echo "  3. On login node:  bash scripts/setup_envs.sh envfile"
  echo "  4. On login node:  bash scripts/setup_envs.sh local"
  echo "  5. In GPU session: bash scripts/setup_envs.sh main"
  echo "  6. In GPU session: bash scripts/setup_envs.sh internvl"
}

TARGET="${1:-}"

case "$TARGET" in
  all)
    echo "============================================"
    echo "  FoodSense-VL — Full Environment Setup"
    echo "============================================"
    echo ""
    setup_container
    setup_cache
    setup_env_file
    setup_local_venv
    setup_main_venv
    setup_internvl_venv
    echo "============================================"
    echo "  ✓ All environments set up successfully!"
    echo "============================================"
    ;;
  container)
    setup_container
    ;;
  main)
    setup_main_venv
    ;;
  internvl)
    setup_internvl_venv
    ;;
  local)
    setup_local_venv
    ;;
  cache)
    setup_cache
    ;;
  envfile)
    setup_env_file
    ;;
  *)
    print_usage
    exit 1
    ;;
esac
