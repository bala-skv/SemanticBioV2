#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Phase 0 — Cluster environment standup (Jawed's slice).
#
# Stands up the Python environment for the memory pilot on the HP cluster.
# Run this ONCE on a login node (it needs outbound network for the installs).
# It creates a venv, installs the package, and leaves a ready-to-activate env
# that the SLURM job (memory_pilot.slurm) will reuse on the GPU node.
#
#   Usage:   bash cluster/setup_env.sh
#   Then:    source .venv/bin/activate   (or let the SLURM script do it)
#
# Safe to re-run: it reuses an existing venv and only (re)installs deps.
# ---------------------------------------------------------------------------
set -euo pipefail

# --- Resolve paths (repo root = parent of this script's dir) ----------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
PY="${PYTHON:-python3}"

# SLURM writes job logs here at submit time, so create it before any sbatch.
mkdir -p "${REPO_ROOT}/logs" "${REPO_ROOT}/runs"

# Keep the (large) HF checkpoint cache off the home quota by default.
# Override HF_HOME to point at fast, roomy shared/scratch storage.
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf_cache}"

# ---------------------------------------------------------------------------
# Keep ALL package/build caches off the (small) home quota.
# torch pulls in multi-GB CUDA wheels (nvidia-*-cu13, cuda-toolkit); the pip
# cache, uv cache and TMPDIR unpack area must live on roomy storage or the
# install dies with "OSError: [Errno 122] Disk quota exceeded".
# Point CACHE_ROOT at scratch/share (e.g. export CACHE_ROOT=/scratch/$USER).
# ---------------------------------------------------------------------------
CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/.cache}"
mkdir -p "${CACHE_ROOT}/pip" "${CACHE_ROOT}/uv" "${CACHE_ROOT}/tmp"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${CACHE_ROOT}/pip}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${CACHE_ROOT}/uv}"
export TMPDIR="${TMPDIR:-${CACHE_ROOT}/tmp}"

# Reuse an already-present torch (e.g. the active conda env) instead of letting
# pip re-download the entire CUDA wheel stack. Set REUSE_TORCH=1 to enable;
# auto-enabled below if `import torch` already works.
REUSE_TORCH="${REUSE_TORCH:-0}"

echo "=============================================================="
echo " Phase 0 cluster env standup"
echo "   repo root  : ${REPO_ROOT}"
echo "   venv       : ${VENV_DIR}"
echo "   python     : $(${PY} --version 2>&1)"
echo "   HF_HOME    : ${HF_HOME}"
echo "   CACHE_ROOT : ${CACHE_ROOT}"
echo "   PIP_CACHE  : ${PIP_CACHE_DIR}"
echo "   TMPDIR     : ${TMPDIR}"
echo "=============================================================="

# --- Optional: load cluster toolchain modules -------------------------------
# torch's pip/conda wheels bundle their own CUDA runtime, so a system CUDA
# module is usually NOT required just to run torch. Load one only if bitsandbytes
# complains at import; match it to the env's build (python -c "import torch;
# print(torch.version.cuda)"). This cluster (module avail) offers, e.g.:
#   cuda/9.2 11.7 11.8 12.1 12.4 12.9   cudnn/9.1.0-cuda-12  9.7.0-cuda-12.8
# Set CUDA_MODULE / CUDNN_MODULE to load them; leave empty (default) to skip.
CUDA_MODULE="${CUDA_MODULE:-}"      # e.g. u22/cuda/12.4
CUDNN_MODULE="${CUDNN_MODULE:-}"    # e.g. u22/cudnn/9.7.0-cuda-12.8
if command -v module >/dev/null 2>&1; then
    if [[ -n "${CUDA_MODULE}" || -n "${CUDNN_MODULE}" ]]; then
        module purge || true
        [[ -n "${CUDA_MODULE}" ]]  && { echo "[modules] loading ${CUDA_MODULE}";  module load "${CUDA_MODULE}"; }
        [[ -n "${CUDNN_MODULE}" ]] && { echo "[modules] loading ${CUDNN_MODULE}"; module load "${CUDNN_MODULE}"; }
    else
        echo "[modules] 'module' available; no CUDA_MODULE/CUDNN_MODULE set (torch wheels"
        echo "          bundle CUDA, so this is normally fine)."
    fi
fi

# --- Create / reuse the virtual environment ---------------------------------
# Set USE_ACTIVE_ENV=1 to install into the currently-active env (e.g. a conda
# env that already has torch) instead of building a fresh .venv. This avoids
# re-downloading the multi-GB CUDA wheel stack into a brand-new venv.
USE_ACTIVE_ENV="${USE_ACTIVE_ENV:-0}"

if [[ "${USE_ACTIVE_ENV}" == "1" ]]; then
    echo "[venv] USE_ACTIVE_ENV=1 -> installing into active interpreter: $(command -v python)"
else
    if [[ ! -d "${VENV_DIR}" ]]; then
        echo "[venv] creating ${VENV_DIR}"
        "${PY}" -m venv "${VENV_DIR}"
    else
        echo "[venv] reusing existing ${VENV_DIR}"
    fi
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
fi

# Upgrading pip/wheel/setuptools is unnecessary in an existing conda env and is
# risky under a tight home quota: a mid-uninstall failure leaves a corrupted
# `~ip` pip. Skip it by default when reusing the active env; make it non-fatal.
SKIP_PIP_UPGRADE="${SKIP_PIP_UPGRADE:-${USE_ACTIVE_ENV}}"
if [[ "${SKIP_PIP_UPGRADE}" == "1" ]]; then
    echo "[pip] skipping pip/wheel/setuptools upgrade (SKIP_PIP_UPGRADE=1)"
else
    python -m pip install --upgrade pip wheel setuptools || \
        echo "[pip] WARNING: pip/wheel/setuptools upgrade failed (continuing)"
fi

# --- Decide whether torch is already available ------------------------------
# If the active interpreter can already import torch, do NOT let pip re-pull the
# entire CUDA wheel stack (nvidia-*-cu13, cuda-toolkit) — that is what blows the
# home-quota. Install the package with --no-deps + the small remaining deps.
if [[ "${REUSE_TORCH}" != "1" ]]; then
    if python -c "import torch" >/dev/null 2>&1; then
        echo "[torch] existing torch detected: $(python -c 'import torch;print(torch.__version__)')"
        echo "[torch] -> REUSE_TORCH=1 (skipping the multi-GB CUDA reinstall)"
        REUSE_TORCH=1
    fi
fi

# Light deps that do NOT drag in the CUDA wheel stack (installed when reusing torch).
# Carry the SAME version constraints as pyproject.toml so pip actually upgrades a
# too-old pre-existing package (e.g. transformers 4.41 -> >=4.44) instead of
# leaving a version-conflict against circuit-routing.
LIGHT_DEPS=("transformers>=4.44,<5" "accelerate>=0.33" "peft>=0.12" "bitsandbytes>=0.43" "pyyaml>=6.0")

# --- Install the package + deps ---------------------------------------------
# Plain path first. On a locked-down / MITM-proxy cluster the TLS chain can
# fail ("self-signed certificate in certificate chain"); fall back to the OS
# trust store via uv, then to pip --trusted-host as a last resort.
#
# install_pkg <extra pip args...> runs the 3-attempt fallback chain.
install_pkg() {
    local ok=0
    echo "[install] attempt 1: pip install -e . $*"
    if pip install -e . "$@"; then ok=1; fi

    if [[ "${ok}" -eq 0 ]] && command -v uv >/dev/null 2>&1; then
        echo "[install] attempt 2: uv (OS trust store) ..."
        if uv pip install --system-certs --python "$(command -v python)" -e . "$@"; then ok=1; fi
    fi

    if [[ "${ok}" -eq 0 ]]; then
        echo "[install] attempt 3: pip with trusted hosts (proxy workaround) ..."
        pip install -e . "$@" \
            --trusted-host pypi.org \
            --trusted-host files.pythonhosted.org \
            --trusted-host download.pytorch.org
    fi
}

if [[ "${REUSE_TORCH}" == "1" ]]; then
    # Package itself, without dependency resolution (keeps torch/CUDA untouched).
    install_pkg --no-deps
    # Then the small remaining runtime deps (these do NOT pull nvidia-*-cu13).
    echo "[install] light deps (reusing existing torch): ${LIGHT_DEPS[*]}"
    if ! pip install "${LIGHT_DEPS[@]}"; then
        pip install "${LIGHT_DEPS[@]}" \
            --trusted-host pypi.org \
            --trusted-host files.pythonhosted.org
    fi
else
    # No torch present: full resolve. Ensure roomy storage first (see CACHE_ROOT).
    install_pkg
fi

echo "=============================================================="
echo "[done] environment ready."
if [[ "${USE_ACTIVE_ENV}" == "1" ]]; then
    echo "  Using active env: $(command -v python)  (no .venv activation needed)"
else
    echo "  Activate with:  source ${VENV_DIR}/bin/activate"
fi
echo "  Verify GPU/bnb: python cluster/check_env.py"
echo "  Pre-fetch model (login node, has network):"
echo "                  python cluster/prefetch_model.py --config configs/phase0.yaml"
echo "  Submit pilot:   sbatch cluster/memory_pilot.slurm"
echo "=============================================================="
