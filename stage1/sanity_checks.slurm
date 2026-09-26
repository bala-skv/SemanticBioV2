#!/bin/bash
#SBATCH --job-name=s1-sanity
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=sm_75          # RTX 2080 Ti  (fp16 trace precision)
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=runs/sanity-%j.out
#SBATCH --error=runs/sanity-%j.err

echo "=========================================="
echo "SLURM_JOB_ID   = $SLURM_JOB_ID"
echo "SLURM_NODELIST = $SLURM_NODELIST"
echo "SLURM_JOB_GPUS = $SLURM_JOB_GPUS"
echo "=========================================="

CONDA_ENV="${CONDA_ENV:-base}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# Repo root
REPO="$HOME/ANLP-Group-Project"    # adjust if your clone path differs
cd "$REPO"

mkdir -p runs

python stage1/stage1_sanity_checks.py \
    --output-json runs/sanity-${SLURM_JOB_ID}.json

echo "Exit code: $?"
