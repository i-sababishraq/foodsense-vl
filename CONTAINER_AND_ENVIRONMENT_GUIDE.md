# Container, Environment & NVMe Guide — FoodSense-VL

**Last updated:** April 9, 2026 (pinned NGC 25.02 + no runtime pip)

This document explains everything needed to recreate the runtime environment on a new SLURM cluster, including container setup, Flash Attention 2, NVMe-accelerated model loading, and package installation.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Container Strategy](#2-container-strategy)
3. [Building the Container from Scratch](#3-building-the-container-from-scratch)
4. [NVMe Local Staging (Model Transfer)](#4-nvme-local-staging-model-transfer)
5. [Virtual Environment Inside the Container](#5-virtual-environment-inside-the-container)
6. [Flash Attention 2](#6-flash-attention-2)
7. [Full sbatch Anatomy](#7-full-sbatch-anatomy)
8. [Package Manifest](#8-package-manifest)
9. [Quick-Start on a New Machine](#9-quick-start-on-a-new-machine)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    SLURM Job Node                       │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │  Singularity Container (--nv --writable-tmpfs)  │    │
│  │                                                 │    │
│  │  Base: NGC PyTorch (pytorch_25.02-py3.sif)      │    │
│  │    OR: Custom flash_attn.sif (CUDA 12.1)       │    │
│  │                                                 │    │
│  │  ┌──────────────────────────────────────────┐   │    │
│  │  │  Python venv (/workspace/sensory_env_ngc)│   │    │
│  │  │  - transformers, peft, bitsandbytes      │   │    │
│  │  │  - flash-attn (pre-installed in image)   │   │    │
│  │  │  - pandas, scipy, scikit-learn           │   │    │
│  │  └──────────────────────────────────────────┘   │    │
│  │                                                 │    │
│  │  Binds:                                         │    │
│  │    $PWD         → /workspace                    │    │
│  │    $LOCAL_STAGE → /local_stage (NVMe)           │    │
│  └─────────────────────────────────────────────────┘    │
│                                                         │
│  NVMe SSD ($LOCAL):                                     │
│    model weights copied here for fast mmap loading      │
└─────────────────────────────────────────────────────────┘
```

**Key insight**: The container provides the base runtime (torch, CUDA, flash-attn). A persistent venv inside the container adds project-specific packages. Model weights are staged to node-local NVMe SSD to avoid Lustre filesystem bottlenecks.

---

## 2. Container Strategy

We use **two container approaches** depending on the cluster:

### Option A: NGC PyTorch Container (Current — PSC Bridges-2)

Uses NVIDIA's pre-built NGC container which already has PyTorch + CUDA + Flash Attention.

```bash
IMAGE="/ocean/containers/ngc/pytorch/pytorch_25.02-py3.sif"
```

- **Pros**: Pre-built, no compilation needed, maintained by NVIDIA
- **Cons**: Cluster-specific path, may have version drift
- **Flash Attention**: Already included in the NGC image
- **Important**: Avoid `pytorch_latest.sif` for production runs on this cluster because it can drift to a Torch/CUDA build incompatible with node drivers.

### Option B: Custom Container (Portable — for any machine)

Build your own `.sif` using the definition file in `containers/vlm_flash_attn_local.def`.

```bash
IMAGE="./containers/vlm_flash_attn.sif"   # or wherever you place it
```

- **Pros**: Fully reproducible, portable across clusters
- **Cons**: Must be built on a machine with root/fakeroot access (not on compute nodes)

---

## 3. Building the Container from Scratch

### Prerequisites

- Linux machine with Apptainer/Singularity installed
- Root or fakeroot access
- Internet access (Docker Hub, PyPI)
- ~15–25 GB free disk
- NVIDIA GPU not required for building (only for running)

### Step 1: Build the image

```bash
# From project root
cd containers/

# Build with the portable definition file
sudo apptainer build vlm_flash_attn.sif vlm_flash_attn_local.def
# Build time: ~15-45 minutes (flash-attn compiles from source)
```

### What the definition file does (`vlm_flash_attn_local.def`)

```
Bootstrap: docker
From: nvidia/cuda:12.1.0-devel-ubuntu22.04

Installs:
  1. System packages: python3, pip, git, ninja-build, libgl1
  2. numpy<2 (pinned — required for compatibility)
  3. PyTorch 2.5.1+cu121 (from official PyTorch index)
  4. Flash Attention 2 (compiled from source, --no-build-isolation)
  5. Project deps: transformers, accelerate, peft, bitsandbytes,
     chromadb, datasets, pandas, pillow, scipy, sentencepiece, trl
```

### Step 2: Adjust CUDA architecture (optional, speeds up build)

Edit the `.def` file before building:

```bash
# Default (broad compatibility):
export TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"  # A100 + RTX30xx + H100

# H100 only (fastest build):
export TORCH_CUDA_ARCH_LIST="9.0"

# A100 only:
export TORCH_CUDA_ARCH_LIST="8.0"
```

### Step 3: Transfer to cluster

```bash
scp vlm_flash_attn.sif user@cluster:/path/to/project/containers/
```

### Layered build (if extending an existing .sif)

If you already have a base `flash_attn.sif`, you can layer on top with `vlm_pipeline_flashattn.def`:

```bash
# This uses Bootstrap: localimage to build on top of flash_attn.sif
# Only adds: peft, bitsandbytes, chromadb, datasets, numpy<2 fix
sudo apptainer build vlm_pipeline.sif vlm_pipeline_flashattn.def
```

---

## 4. NVMe Local Staging (Model Transfer)

### Why NVMe staging?

Loading a 27B model (~54 GB in safetensors) from a shared parallel filesystem (Lustre/GPFS) causes mmap stalls and slow startup. Node-local NVMe SSDs are 10-50x faster for random I/O.

### How it works

On PSC Bridges-2, `$LOCAL` points to the per-node NVMe SSD (only available inside SLURM jobs).

```bash
# ── NVMe staging block (copy-paste into any sbatch) ──────────────────

# 1. Create local staging directory
LOCAL_STAGE="${LOCAL:-/local}/${SLURM_JOB_ID}_eval"
mkdir -p "$LOCAL_STAGE/huggingface/hub"

# 2. Source path (model cached on shared filesystem)
SRC_HUB="$PWD/cache/huggingface/hub"
MODEL_CACHE="models--google--gemma-3-27b-it"

# 3. Sync model to NVMe with retry logic
echo "Staging model cache to local NVMe ($LOCAL_STAGE)..."
if [ -d "$SRC_HUB/$MODEL_CACHE" ]; then
  echo "  Syncing $MODEL_CACHE ..."
  RC=1; n=0
  while [[ $RC -ne 0 && $n -lt 5 ]]; do
    rsync -a "$SRC_HUB/$MODEL_CACHE" "$LOCAL_STAGE/huggingface/hub/"
    RC=$?
    let n=n+1
    [[ $RC -ne 0 ]] && echo "    rsync attempt $n failed, retrying in 10s..." && sleep 10
  done
  if [[ $RC -eq 0 ]]; then
    echo "    Done ($(du -sh "$LOCAL_STAGE/huggingface/hub/$MODEL_CACHE" | cut -f1))"
  else
    echo "    WARNING: rsync failed after $n attempts, falling back to shared FS"
  fi
else
  echo "  SKIP $MODEL_CACHE (not cached yet)"
fi

# 4. Redirect HF_HOME to NVMe
export HF_HOME="$LOCAL_STAGE/huggingface"

# 5. Cleanup on exit (important — NVMe is limited)
trap 'echo "Cleaning up $LOCAL_STAGE"; rm -rf "$LOCAL_STAGE"' EXIT
```

### Adapting for other clusters

| Cluster | NVMe Variable | Typical Path |
|---------|--------------|--------------|
| PSC Bridges-2 | `$LOCAL` | `/local/<jobid>` |
| NERSC Perlmutter | `$PSCRATCH` or `$SCRATCH` | `/pscratch/...` |
| Generic SLURM | Check `scontrol show node` for `TmpDisk` | `/tmp`, `/local`, `/nvme` |

If no NVMe is available, skip the staging and load from shared FS (slower but works):

```bash
export HF_HOME="$PWD/cache/huggingface"
```

---

## 5. Virtual Environment Inside the Container

The container is read-only, so we create a persistent venv at `/workspace/sensory_env_ngc` (which maps to `$PWD/sensory_env_ngc` on the host via the bind mount).

### The VENV_SETUP block

This block is embedded in every sbatch script. Current production policy is **pre-provision once, then verify only** (no runtime `pip install` in jobs):

```bash
VENV_DIR="/workspace/sensory_env_ngc"

VENV_SETUP='
set -euo pipefail

# Fix CUDA compat symlink (some NGC containers need this)
if [ -e /usr/local/cuda/compat/lib/libcuda.so.1 ] && [ ! -e /usr/local/cuda/compat/lib/libcuda.so ]; then
  ln -s /usr/local/cuda/compat/lib/libcuda.so.1 /usr/local/cuda/compat/lib/libcuda.so || true
fi
export LD_LIBRARY_PATH="/usr/local/cuda/compat/lib:${LD_LIBRARY_PATH:-}"

# Create venv if it does not exist (--system-site-packages inherits torch, flash-attn from container)
if [ ! -d "'"$VENV_DIR"'" ]; then
  python3 -m venv --system-site-packages "'"$VENV_DIR"'"
fi
source "'"$VENV_DIR"'/bin/activate"
# Preflight only: fail fast if env is missing required packages
python - <<'"'"'PY'"'"'
import importlib
for m in ["transformers", "accelerate", "peft", "bitsandbytes", "pandas", "PIL", "sklearn", "scipy"]:
  importlib.import_module("PIL" if m == "PIL" else m)
PY

# Verify flash-attn is available
python - <<'"'"'FX'"'"'
import torch, flash_attn
print("  torch:", torch.__version__, "flash_attn:", flash_attn.__version__)
print("  cuda:", torch.cuda.is_available(), "devices:", torch.cuda.device_count())
FX
'
```

### Key details

- `--system-site-packages` is critical — it inherits `torch` and `flash-attn` from the container base image
- Keep one shared pre-provisioned venv at `$PWD/sensory_env_ngc/`
- Do package installs in a controlled setup step (interactive/maintenance), not in production sbatch runtime

---

## 6. Flash Attention 2

### In the NGC container

Flash Attention 2 comes pre-installed. No action needed.

### In the custom container

Compiled from source during `apptainer build`:

```bash
# In vlm_flash_attn_local.def
export TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"
export MAX_JOBS=4
pip3 install --no-cache-dir flash-attn --no-build-isolation
```

### Verifying flash-attn works

```python
import torch
import flash_attn
print(f"flash_attn: {flash_attn.__version__}")
print(f"CUDA: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}")
```

### Building flash-attn outside a container (on bare metal)

```bash
# 1. Set CUDA paths
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH

# 2. Target your GPU architecture
export TORCH_CUDA_ARCH_LIST="9.0"  # H100
export MAX_JOBS=8
export FLASH_ATTENTION_FORCE_BUILD="TRUE"

# 3. Set TMPDIR to avoid cross-device link errors
export TMPDIR=/scratch/tmp  # NOT /tmp if installing to a different filesystem
mkdir -p $TMPDIR

# 4. Install
pip install flash-attn --no-build-isolation --no-cache-dir
```

---

## 7. Full sbatch Anatomy

Here's the complete structure of a production sbatch script, annotated:

```bash
#!/bin/bash
#SBATCH --job-name=eval_v2
#SBATCH --partition=GPU-shared          # ← CLUSTER-SPECIFIC
#SBATCH --account=soc250010p            # ← CLUSTER-SPECIFIC
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h100-80:1                # ← GPU TYPE
#SBATCH --time=24:00:00
#SBATCH --mem=80G
#SBATCH --export=ALL
#SBATCH --output=logs/job_%j.out        # ← UPDATE PATH
#SBATCH --error=logs/job_%j.err

set -euo pipefail

# ── 1. Project setup ──────────────────────────────────────
cd /path/to/project                      # ← UPDATE PATH
source .env && export HF_TOKEN HUGGINGFACE_HUB_TOKEN

# ── 2. Cache directories ─────────────────────────────────
export HF_HOME="$PWD/cache/huggingface"
export PIP_CACHE_DIR="$PWD/cache/pip"
export TMPDIR="$PWD/cache/tmp"
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$TMPDIR"

# ── 3. CUDA tuning ────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export TOKENIZERS_PARALLELISM=false

# ── 4. NVMe staging (see Section 4) ──────────────────────
# [copy the NVMe staging block here]

# ── 5. Container execution ────────────────────────────────
IMAGE="/path/to/container.sif"           # ← UPDATE PATH
VENV_DIR="/workspace/sensory_env_ngc"

SING_EXEC=(singularity exec --nv --writable-tmpfs --cleanenv --home /tmp
  --bind "$PWD:/workspace"
  --bind "$LOCAL_STAGE:/local_stage"
  --pwd /workspace
  --env HF_TOKEN="$HF_TOKEN"
  --env HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
  --env HF_HOME="/local_stage/huggingface"
  --env PIP_CACHE_DIR="/workspace/cache/pip"
  --env TMPDIR="/workspace/cache/tmp"
  --env PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF"
  --env SAFETENSORS_FAST_GPU="1"
  "$IMAGE")

# ── 6. Run ────────────────────────────────────────────────
"${SING_EXEC[@]}" bash -lc "
$VENV_SETUP
python -u your_script.py --your_args
"
```

### Singularity flags explained

| Flag | Purpose |
|------|---------|
| `--nv` | Pass through NVIDIA GPU drivers |
| `--writable-tmpfs` | Allow temporary writes inside the container |
| `--cleanenv` | Don't leak host environment variables into container |
| `--home /tmp` | Prevent using host's $HOME (avoids permission issues) |
| `--bind $PWD:/workspace` | Mount project directory into container |
| `--bind $LOCAL_STAGE:/local_stage` | Mount NVMe staging directory |
| `--pwd /workspace` | Set working directory inside container |
| `--env KEY=VALUE` | Pass specific environment variables |

---

## 8. Package Manifest

### Packages pre-installed in container image

| Package | Version | Notes |
|---------|---------|-------|
| torch | ≥2.5.1+cu121 | With CUDA 12.1 support |
| torchvision | ≥0.20.1+cu121 | |
| torchaudio | ≥2.5.1+cu121 | |
| flash-attn | ≥2.3.6 | FlashAttention 2, compiled for sm_80/90 |
| numpy | <2.0 | **Must be <2** for compatibility |

### Packages installed in venv (pre-provisioned once)

| Package | Version | Purpose |
|---------|---------|---------|
| transformers | 4.57.6 | Model loading/tokenization (Gemma/Qwen/LLaVA/Food-LLaMA path) |
| accelerate | 1.13.0 | Mixed precision, device placement |
| peft | 0.18.1 | LoRA/QLoRA adapter management |
| bitsandbytes | 0.49.2 | 4-bit quantization (NF4) |
| pandas | 3.x | Data loading/manipulation |
| pillow | 12.x | Image processing |
| tqdm | 4.67.x | Progress bars |
| scikit-learn | 1.8.x | Metrics/utilities |
| scipy | installed | Statistical functions |

### Optional packages (for specific tasks)

| Package | When needed |
|---------|------------|
| sentencepiece, protobuf | Some tokenizers |
| qwen-vl-utils | Qwen2.5-VL baseline |
| chromadb | RAG pipeline (legacy) |
| datasets | HuggingFace datasets |

---

## 9. Quick-Start on a New Machine

### Step-by-step checklist

```bash
# ═══════════════════════════════════════════════════════════
# 1. Clone project
# ═══════════════════════════════════════════════════════════
git clone https://github.com/sishraq/foodsense-vl.git
cd foodsense-vl

# ═══════════════════════════════════════════════════════════
# 2. Build or obtain the container
# ═══════════════════════════════════════════════════════════

# OPTION A: Build custom container (requires root, ~30 min)
cd containers/
sudo apptainer build vlm_flash_attn.sif vlm_flash_attn_local.def
cd ..

# OPTION B: Use cluster-provided NGC container
# Set IMAGE="/path/to/ngc/pytorch_25.02-py3.sif" in sbatch files

# ═══════════════════════════════════════════════════════════
# 3. Set up secrets
# ═══════════════════════════════════════════════════════════
cp .env.example .env
# Edit .env → add HF_TOKEN=hf_xxx

# ═══════════════════════════════════════════════════════════
# 4. Download/cache the base model
# ═══════════════════════════════════════════════════════════
mkdir -p cache/huggingface
export HF_HOME=$PWD/cache/huggingface

# Either download via huggingface-cli:
huggingface-cli download google/gemma-3-27b-it --cache-dir $HF_HOME

# Or copy cached model from another machine:
rsync -a old_machine:/path/cache/huggingface/hub/models--google--gemma-3-27b-it \
         cache/huggingface/hub/

# ═══════════════════════════════════════════════════════════
# 5. Copy data + checkpoints
# ═══════════════════════════════════════════════════════════
# Required:
#   data/FINAL_DATASET_COMPLETE_with_rescaling.csv
#   data/Images/  (2,915 images)
#
# Optional (for inference only):
#   checkpoints/gemma3_qlora_human_v2_38390266/checkpoint-200
#   checkpoints/gemma3_qlora_stage2_v2_38407321/checkpoint-200

# ═══════════════════════════════════════════════════════════
# 6. Update sbatch files for your cluster
# ═══════════════════════════════════════════════════════════
# Search and replace these cluster-specific values:
#   - #SBATCH --account=YOUR_ACCOUNT
#   - #SBATCH --partition=YOUR_GPU_PARTITION
#   - #SBATCH --gpus=YOUR_GPU_TYPE:1
#   - cd /path/to/your/project
#   - IMAGE="/path/to/your/container.sif"
#   - #SBATCH --output and --error log paths
#   - $LOCAL variable (NVMe path for your cluster)

# Quick search to find all lines to change:
grep -rn "soc250010p\|GPU-shared\|/jet/home/sishraq\|/ocean/containers" slurm/*.sbatch

# ═══════════════════════════════════════════════════════════
# 7. Create local benchmark venv (for running on login node)
# ═══════════════════════════════════════════════════════════
python3.10 -m venv sensory_env
source sensory_env/bin/activate
pip install -U pip
pip install "numpy<2" pandas scipy scikit-learn tqdm
deactivate

# ═══════════════════════════════════════════════════════════
# 8. Test: Submit a quick eval job
# ═══════════════════════════════════════════════════════════
ADAPTER_DIR=checkpoints/gemma3_qlora_human_v2_38390266/checkpoint-200 \
OUTPUT_DIR=eval_outputs/test_run \
sbatch slurm/eval_ours.sbatch

# Monitor:
squeue -u $USER
tail -f logs/eval_*.out
```

---

## 10. Troubleshooting

### Container Issues

| Problem | Solution |
|---------|----------|
| `FATAL: container creation failed` | Need root/fakeroot; build on a machine with permissions |
| `nvidia-container-cli: ldcache error` | Use `--nv` flag with singularity exec |
| `libcuda.so: cannot open shared object` | Add CUDA compat symlink (see VENV_SETUP block) |
| Container too large for /tmp | Set `APPTAINER_TMPDIR` or `SINGULARITY_TMPDIR` to scratch |

### NVMe Staging Issues

| Problem | Solution |
|---------|----------|
| `$LOCAL` not set | Not in a SLURM job, or cluster doesn't expose NVMe via `$LOCAL`. Check `scontrol show node` |
| rsync fails repeatedly | Shared filesystem may be overloaded; fall back to direct loading |
| Ran out of NVMe space | Gemma 3 27B needs ~54GB; request nodes with sufficient local storage |

### Flash Attention Issues

| Problem | Solution |
|---------|----------|
| `No module named 'flash_attn'` | Not in the container venv; check `--system-site-packages` |
| `CUDA initialization: driver too old` with `torch 2.11` | Use pinned `pytorch_25.02-py3.sif`; avoid `pytorch_latest.sif` drift |
| Build hangs during compilation | Set `TORCH_CUDA_ARCH_LIST` to only your GPU arch |
| `RuntimeError: FlashAttention only supports Ampere GPUs or newer` | Need GPU compute capability ≥8.0 (A100, H100) |

### Python / Package Issues

| Problem | Solution |
|---------|----------|
| `numpy>=2` installed breaking things | Pin with `pip install "numpy<2"` |
| InternVL crashes on newer transformers | Patched in `eval_sensory_baselines.py` via `_ensure_internvl_generation_compat()` |
| `ImportError: cannot import name 'gemma3'` | `transformers` version too old; needs ≥4.49 for Gemma 3 support |
| Cross-device link error during pip install | Set `TMPDIR` to same filesystem as install target |

---

## 11. InternVL Separate Environment (Recommended)

InternVL should run in a **separate venv** from Gemma/Qwen/LLaVA/Food-LLaMA.

Reason: InternVL remote code is sensitive to generation/cache API changes in newer `transformers`. Using an isolated venv avoids breaking other model paths and prevents cross-model dependency drift.

### Recommended versions (InternVL path)

- Container: `/ocean/containers/ngc/pytorch/pytorch_25.02-py3.sif`
- Torch/flash-attn: inherit from container via `--system-site-packages`
- `transformers==4.48.3`
- `tokenizers<0.22`
- `accelerate`, `peft`, `bitsandbytes`, `pandas`, `pillow`, `tqdm`, `scikit-learn`, `scipy`, `einops`

### One-time setup inside container

```bash
IMAGE="/ocean/containers/ngc/pytorch/pytorch_25.02-py3.sif"

singularity exec --nv --writable-tmpfs --cleanenv --home /tmp \
  --bind "$PWD:/workspace" --pwd /workspace "$IMAGE" bash -lc '
set -euo pipefail
python3 -m venv --system-site-packages /workspace/.venvs/sensory_internvl_compat
source /workspace/.venvs/sensory_internvl_compat/bin/activate
python -m pip install -U pip -q
python -m pip install -q "numpy<2" "transformers==4.48.3" "tokenizers<0.22" \
  accelerate peft bitsandbytes pandas pillow tqdm scikit-learn scipy einops
python - <<"PY"
import torch, transformers, flash_attn
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("flash_attn:", flash_attn.__version__)
print("cuda:", torch.cuda.is_available(), "devices:", torch.cuda.device_count())
PY
'
```

### InternVL NVMe staging block

```bash
LOCAL_STAGE="${LOCAL:-/local}/${SLURM_JOB_ID}_eval"
MODEL_CACHE_NAME="models--OpenGVLab--InternVL2_5-26B"
SRC_CACHE="$PWD/cache/huggingface/hub/$MODEL_CACHE_NAME"

mkdir -p "$LOCAL_STAGE/huggingface/hub"
if [ -d "$SRC_CACHE" ]; then
  for attempt in 1 2 3 4 5; do
    rsync -a "$SRC_CACHE" "$LOCAL_STAGE/huggingface/hub/" && break
    echo "rsync attempt $attempt failed; retrying..."
    sleep 10
  done
  export HF_HOME="$LOCAL_STAGE/huggingface"
else
  export HF_HOME="$PWD/cache/huggingface"
fi

trap 'rm -rf "$LOCAL_STAGE" 2>/dev/null || true' EXIT
```

### InternVL runtime preflight (no runtime pip)

```bash
source /workspace/.venvs/sensory_internvl_compat/bin/activate
python - <<'PY'
import importlib
for m in ["transformers", "accelerate", "peft", "bitsandbytes", "pandas", "PIL", "sklearn", "scipy", "einops"]:
    importlib.import_module("PIL" if m == "PIL" else m)
import torch, transformers, flash_attn
print("transformers:", transformers.__version__)
print("torch:", torch.__version__, "flash_attn:", flash_attn.__version__)
print("cuda:", torch.cuda.is_available(), "devices:", torch.cuda.device_count())
PY
```

### InternVL run command pattern

```bash
python -u eval_sensory_baselines.py \
  --adapter_dir "$ADAPTER_DIR" \
  --model internvl \
  --split test \
  --output_dir "$OUTPUT_DIR" \
  --prompt "$PROMPT" \
  --max_new_tokens 1024 \
  --strict \
  ${IMAGE_IDS:+--image_ids "$IMAGE_IDS"}
```

Use `eval_sensory_internvl_compatible.sbatch` as the production template for this path.

---

## Reference: Files Mentioned in This Guide

```
# Container definitions
containers/vlm_flash_attn_local.def      # Portable def (build from scratch)
containers/vlm_pipeline_flashattn.def    # Layered def (extends flash_attn.sif)
containers/BUILD_LOCAL_INSTRUCTIONS.md   # Detailed build walkthrough
flash_attn.def                           # Original def (Anvil-specific paths)
flash_attn.sif                           # Pre-built container (~7GB)

# Sbatch templates (with NVMe + venv pattern)
eval_sensory_ours_v2.sbatch              # Eval template (production)
train_yelp_qlora_human_only_v2.sbatch    # Stage-1 training template
train_yelp_qlora_stage2_mammoth_v2.sbatch # Stage-2 training template

# Environment configs
.env                                     # Secrets (HF_TOKEN)
setup.env                                # Anvil-specific env setup
requirements.txt                         # Full pip requirements
foodsense-vl/requirements.txt            # Clean repo requirements

# NGC container (PSC Bridges-2 specific)
/ocean/containers/ngc/pytorch/pytorch_25.02-py3.sif
```

## INTERNVL RUNBOOK (ISOLATED ENV)

Use this when running InternVL inference/evaluation so it does not break from `transformers` drift in shared environments.

### Why InternVL is isolated
- InternVL is sensitive to `transformers` generation/cache API changes.
- Keep InternVL on `transformers==4.48.3` in its own venv.
- Keep other baselines (`qwen2_vl`, `llava`, `food_llama`) on their separate environment/version track.

### Canonical files
- `eval_sensory_internvl_compatible.sbatch` (InternVL job runner)
- `docs/CONTAINER_AND_ENVIRONMENT_GUIDE.md` (stable container/env policy)

### One-time InternVL venv provisioning (inside container)
```bash
singularity exec --nv --cleanenv --writable-tmpfs \
  --bind "$PWD:/workspace" \
  --pwd /workspace \
  /ocean/containers/ngc/pytorch/pytorch_25.02-py3.sif \
  bash -lc '
set -euo pipefail
python3 -m venv --system-site-packages /workspace/.venvs/sensory_internvl_compat
source /workspace/.venvs/sensory_internvl_compat/bin/activate
python -m pip install -U pip
python -m pip install "numpy<2" "transformers==4.48.3" "tokenizers<0.22" \
  accelerate peft bitsandbytes pandas pillow scikit-learn scipy einops
python - <<"PY"
import transformers, torch
print("transformers", transformers.__version__)
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available(), "devices", torch.cuda.device_count())
PY
'
```

### Submit InternVL job (safe, non-overwrite)
```bash
TS=$(date +%Y%m%d_%H%M)
IMAGE_IDS='0001_01lamiW2bWW0rXlllNHYMA.jpg,0002_01zZeZBIFZ82S5XmA4GYJg.jpg,0005_08Eu2m3RTrpssX9GIKtHtg.jpg,0010_0dHJ9fque7joEy7J0UrHmA.jpg,0015_0g2pruxDhqhh2E-cEoYOLA.jpg'
export IMAGE_IDS
sbatch --job-name=supp5fix_internvl --export=ALL,OUTPUT_DIR=eval_outputs/supp5fix_internvl_${TS} \
  eval_sensory_internvl_compatible.sbatch
```

### Verify run health
- `sacct -j <JOB_ID> --format=JobID,State,ExitCode,Elapsed -X`
- Check logs for:
  - `transformers: 4.48.3`
  - `cuda: True devices: 1`
  - `NVMe staging complete`
  - `Test images: 5` (or expected count)
- Confirm outputs:
  - `eval_outputs/.../internvl/internvl_predictions.jsonl`
  - `eval_outputs/.../eval_results.json`
  - `eval_outputs/.../eval_metrics.csv`

### Common failures + fixes
- `AttributeError: 'NoneType' object has no attribute 'shape'`:
  - Usually wrong `transformers` version; ensure InternVL venv is pinned at `4.48.3`.
- Missing/slow model load:
  - Ensure NVMe staging path exists and rsync succeeded.
- Cross-model breakage after package updates:
  - Never share one mutable venv across all models.

---