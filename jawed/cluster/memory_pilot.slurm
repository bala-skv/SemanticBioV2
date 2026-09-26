#!/usr/bin/env bash
#SBATCH --job-name=cr-mem-pilot
#SBATCH --output=logs/mem-pilot-%j.out
#SBATCH --error=logs/mem-pilot-%j.err
#SBATCH --partition=u22            # default GPU partition on this cluster (sinfo: u22*)
#SBATCH --gres=gpu:1               # one small GPU is the whole point of the pilot
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00            # sweep is short; 2h is generous headroom
# #SBATCH --account=<your_account> # EDIT/uncomment if your cluster requires it
# #SBATCH --nodelist=<gpu-node>    # EDIT/uncomment to pin a specific GPU type
# ---------------------------------------------------------------------------
# Phase 0 — memory pilot (Jawed's go/no-go on the 4-5 GB budget).
#
#   Submit from the repo root:   sbatch cluster/memory_pilot.slurm
#
# Assumes cluster/setup_env.sh has already created .venv and (ideally)
# cluster/prefetch_model.py has warmed the HF cache on a login node.
# Results land in runs/<timestamp>-memory-pilot/ (durable home/shared storage).
# ---------------------------------------------------------------------------
set -euo pipefail

# --- Resolve repo root (works whether launched from repo root or elsewhere) --
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    cd "${SLURM_SUBMIT_DIR}"
fi
REPO_ROOT="$(pwd)"
mkdir -p logs

echo "=============================================================="
echo " Memory pilot job ${SLURM_JOB_ID:-<local>} on $(hostname)"
echo "   repo root : ${REPO_ROOT}"
echo "   started   : $(date -Is)"
echo "=============================================================="

# --- Cluster toolchain modules (optional) -----------------------------------
# torch wheels bundle their own CUDA runtime, so a system CUDA module is usually
# not needed. Load one only if bitsandbytes complains; match it to the env's
# build. Set CUDA_MODULE / CUDNN_MODULE to load (this cluster offers e.g.
# u22/cuda/12.4, u22/cudnn/9.7.0-cuda-12.8). Leave empty (default) to skip.
CUDA_MODULE="${CUDA_MODULE:-}"
CUDNN_MODULE="${CUDNN_MODULE:-}"
if command -v module >/dev/null 2>&1 && [[ -n "${CUDA_MODULE}" || -n "${CUDNN_MODULE}" ]]; then
    module purge || true
    [[ -n "${CUDA_MODULE}" ]]  && module load "${CUDA_MODULE}"
    [[ -n "${CUDNN_MODULE}" ]] && module load "${CUDNN_MODULE}"
fi
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
fi

# --- Activate the environment -----------------------------------------------
# Prefer a conda env that already has torch (set CONDA_ENV=torch310) to avoid
# the multi-GB CUDA reinstall; otherwise fall back to the .venv from setup_env.sh.
CONDA_ENV="${CONDA_ENV:-}"
if [[ -n "${CONDA_ENV}" ]]; then
    # conda's activate/deactivate hooks reference unset vars (e.g. CONDA_BACKUP_CXX
    # from the gxx compiler package), which are NOT compatible with `set -u`.
    # Relax nounset just around activation, then restore it.
    set +u
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
    set -u
    echo "[env] conda env '${CONDA_ENV}' -> $(command -v python)"
else
    VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
    if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
        echo "ERROR: ${VENV_DIR} not found and CONDA_ENV unset." >&2
        echo "       Run 'bash cluster/setup_env.sh' first, or set CONDA_ENV=<name>." >&2
        exit 2
    fi
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
fi

# --- Point HF at the pre-warmed cache and run OFFLINE (no network on gpu node)
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
# Cleaner peak-memory readings; expandable segments reduce fragmentation.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
# Unbuffered stdout/stderr so the SLURM .out reflects live progress (otherwise
# print() output is block-buffered and `tail -f` looks frozen for minutes).
export PYTHONUNBUFFERED=1
echo "  HF_HOME        : ${HF_HOME}"
echo "  HF_HUB_OFFLINE : ${HF_HUB_OFFLINE}"

# --- Gate 1: environment must be sane before we load a 1.5B model -----------
echo "--- verifying CUDA + bitsandbytes -----------------------------"
python cluster/check_env.py

# --- Gate 1b: base checkpoint MUST be in the (offline) HF cache -------------
# With HF_HUB_OFFLINE=1, a cache miss otherwise crashes deep inside the pilot
# and the exit-1 is indistinguishable from a real NO-GO. Fail fast + actionably.
# Skipped when online (HF_HUB_OFFLINE=0) since the model downloads on first use.
if [[ "${HF_HUB_OFFLINE}" == "1" ]]; then
    echo "--- verifying base checkpoint is cached (offline) -------------"
    set +e
    python - "configs/phase0.yaml" <<'PY'
import sys
from circuit_routing.config import load_config
from transformers import AutoConfig, AutoTokenizer
cfg = load_config(sys.argv[1])
AutoConfig.from_pretrained(cfg.model.name, revision=cfg.model.revision,
                           trust_remote_code=cfg.model.trust_remote_code,
                           local_files_only=True)
AutoTokenizer.from_pretrained(cfg.model.name, revision=cfg.model.revision,
                              trust_remote_code=cfg.model.trust_remote_code,
                              local_files_only=True)
print(f"  [ok] '{cfg.model.name}' found in cache.")
PY
    CACHE_RC=$?
    set -e
    if [[ ${CACHE_RC} -ne 0 ]]; then
        echo "ERROR: base checkpoint not in the HF cache at HF_HOME=${HF_HOME}." >&2
        echo "       This node cannot download (HF_HUB_OFFLINE=1). Fix by either:" >&2
        echo "         1) prefetch to a SHARED path all nodes see, e.g.:" >&2
        echo "              export HF_HOME=\$PWD/.hf_cache   # repo on NFS/home" >&2
        echo "              python cluster/prefetch_model.py --config configs/phase0.yaml" >&2
        echo "         2) submit with the SAME HF_HOME you prefetched into:" >&2
        echo "              HF_HOME=<that_path> sbatch ... cluster/memory_pilot.slurm" >&2
        echo "         3) if GPU nodes have network, submit with HF_HUB_OFFLINE=0 and a" >&2
        echo "            node-local cache: HF_HOME=/ssd_scratch/\$USER/hf_cache HF_HUB_OFFLINE=0 ..." >&2
        echo "       NOTE: node-local /scratch and /ssd_scratch are NOT visible from other nodes." >&2
        exit 2
    fi
else
    echo "--- online mode (HF_HUB_OFFLINE=0): model will download on first use ---"
fi

# --- Gate 2: the memory pilot (exits non-zero on NO-GO) ---------------------
echo "--- running memory pilot --------------------------------------"
set +e
# Resume support: set RESUME_DIR=runs/<ts>-memory-pilot to continue a sweep that
# a prior job left unfinished (e.g. killed at the SLURM time limit). The script
# skips configs already present in that dir's memory_pilot.csv.
RESUME_ARGS=()
if [[ -n "${RESUME_DIR:-}" ]]; then
    echo "  resuming from ${RESUME_DIR}"
    RESUME_ARGS=(--resume "${RESUME_DIR}")
fi
python -u scripts/memory_pilot.py --config configs/phase0.yaml "${RESUME_ARGS[@]}"
PILOT_RC=$?
set -e

echo "=============================================================="
echo " finished  : $(date -Is)"
echo " pilot exit : ${PILOT_RC}  ($( [[ ${PILOT_RC} -eq 0 ]] && echo GO || echo NO-GO ))"
echo " artefacts  : ${REPO_ROOT}/runs/  (memory_pilot.csv + summary.json)"
echo "=============================================================="
exit ${PILOT_RC}
