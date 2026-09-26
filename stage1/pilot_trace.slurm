#!/bin/bash
#SBATCH --job-name=s1-pilot
#SBATCH --partition=u22
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=runs/pilot-%j.out
#SBATCH --error=runs/pilot-%j.err

echo "=========================================="
echo "SLURM_JOB_ID   = $SLURM_JOB_ID"
echo "SLURM_NODELIST = $SLURM_NODELIST"
echo "SLURM_JOB_GPUS = $SLURM_JOB_GPUS"
echo "=========================================="

CONDA_ENV="${CONDA_ENV:-base}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# Repo root
REPO="$HOME/ANLP-Group-Project"
cd "$REPO"

mkdir -p runs

python stage1/pilot_trace.py

echo "Exit code: $?"