#!/bin/bash
# SLURM job for the Laya/ModernBERT-large CO2 QN prototype, for UCL Physics & Astronomy's
# Hypatia cluster (https://uclphysast.github.io/clusters/hypatia/).
#
# Submit with:            sbatch hypatia_job.sh
# Override the GPU type:  sbatch --gres=gpu:v100:1 hypatia_job.sh   (A100 queues can be long;
#                          V100 is fine for this workload if you don't need the speedup — see
#                          the README note at the bottom of this file)
# Monitor with:            squeue -u $USER
#
# Per CLAUDE.md: this project always uses uv, never conda/mamba. This script assumes uv is
# already installed on Hypatia (curl -LsSf https://astral.sh/uv/install.sh | sh, if not).

#SBATCH -p GPU
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=laya-co2-qn
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# #SBATCH --mail-user=your.email@ucl.ac.uk
# #SBATCH --mail-type=ALL

set -euo pipefail

# --- 1. Paths -----------------------------------------------------------------------
# Edit PROJECT_DIR if this repo lives somewhere other than $HOME on Hypatia.
PROJECT_DIR="${PROJECT_DIR:-$HOME/Quantum-Number-Prediction/GNN_CO2_QN_assignment/experiments/laya_prototype}"
cd "$PROJECT_DIR"
mkdir -p logs checkpoints

# Route the HF model/dataset cache to scratch, not $HOME (home quota is small and shared,
# per the cluster docs) — and reuse it across resubmissions instead of re-downloading.
export HF_HOME="/share/rcifdata/${USER}/hf_cache"
mkdir -p "$HF_HOME"

# GPU compute nodes on many HPC clusters have no outbound internet. If that's true here,
# pre-fetch the model once from a login node (same HF_HOME as above), then uncomment this:
# export HF_HUB_OFFLINE=1

# --- 2. Environment -------------------------------------------------------------------
command -v uv >/dev/null 2>&1 || {
    echo "uv not found on PATH. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
}

uv sync

echo "=== GPU / precision check ==="
uv run python -c "
import torch
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
bf16 = torch.cuda.is_bf16_supported()
print(f'GPU: {name}  compute capability: {cap}  bf16 tensor cores: {bf16}')
assert torch.cuda.is_available(), 'CUDA not visible inside the job — check --gres and the node allocation.'
"

# --- 3. Hyperparameters -----------------------------------------------------------------
# Defaults below assume an A100 (40GB+): much larger effective batch than the RTX 2070
# prototype used (16 * grad_accum 2 = 32) since memory/compute headroom is no longer the
# constraint. train_laya.py/evaluate.py read these from the environment (see LAYA_* in
# their source) and auto-select bf16 (no GradScaler needed) vs fp16 based on the GPU.
export LAYA_BATCH_SIZE=64
export LAYA_GRAD_ACCUM=1
export LAYA_EVAL_BATCH_SIZE=256
export LAYA_NUM_WORKERS=8
# export LAYA_EPOCHS=8        # uncomment to override other train_laya.py defaults
# export LAYA_LR=2e-4
# export LAYA_PATIENCE=2

# --- 4. Run -------------------------------------------------------------------------
echo "=== prepare_data.py ==="
uv run python prepare_data.py

echo "=== train_laya.py ==="
uv run python train_laya.py

echo "=== evaluate.py ==="
uv run python evaluate.py

echo "Done. See results.md / eval_results.json / per_row_predictions.csv for output."
