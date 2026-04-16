# SLURM Job Templates

Reference SLURM job scripts for HPC clusters. These were developed on PSC Bridges-2 with NVIDIA H100 GPUs.

## Adapting for your cluster

Before submitting, update these variables in each `.sbatch` file:

- `--account` — your SLURM allocation account
- `--partition` — your GPU partition name
- `CONTAINER` — path to your Apptainer/Singularity container (or remove if using conda)
- `PROJECT_ROOT` — path to your clone of this repository
- `CHECKPOINT_DIR` — path to downloaded model weights

## Job descriptions

| Script | Purpose | GPUs | Time |
|--------|---------|------|------|
| `train_stage1.sbatch` | Stage 1: Human sensory alignment | 1x H100 | ~4h |
| `train_stage2.sbatch` | Stage 2: MAmmoTH expansion | 1x H100 | ~6h |
| `eval_foodsensevl.sbatch` | Evaluate fine-tuned FoodSense-VL model | 1x H100 | ~2h |
| `eval_baselines.sbatch` | Evaluate baseline models (Qwen, LLaVA, InternVL, Food-LLaMA) | 1x H100 | ~4h |
| `eval_internvl.sbatch` | InternVL evaluation with compatibility environment | 1x H100 | ~4h |
| `precompute.sbatch` | Generate MAmmoTH synthetic targets | 1x H100 | ~8h |

## Quick start

```bash
# Edit the sbatch file first, then:
sbatch slurm/train_stage1.sbatch
```
