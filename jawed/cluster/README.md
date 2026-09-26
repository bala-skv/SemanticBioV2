# Phase 0 — Cluster Runbook (memory pilot + env standup)

**Owner:** Jawed Mohammed Khan (2025801009) — *Cluster Environment & Memory Pilot Lead*

These artefacts stand up the environment (unblocks the whole team) and run the
**go/no-go memory pilot** on the HP cluster. Everything here runs from the repo
root: `circuit_routing/`.

| Artefact | Where it runs | Purpose |
|---|---|---|
| `cluster/setup_env.sh` | login node | Create `.venv`, install the package (with SSL/proxy fallbacks) |
| `cluster/prefetch_model.py` | login node | Download + verify the base checkpoint into `HF_HOME` |
| `cluster/check_env.py` | GPU node | Verify CUDA + bitsandbytes (4-bit kernel actually runs) |
| `cluster/memory_pilot.slurm` | GPU node (SLURM) | Run the pilot; emits `runs/<ts>-memory-pilot/` |

## One-time setup (login node — has network)

```bash
cd circuit_routing

# Put the big HF cache on roomy shared/scratch storage, NOT your home quota:
export HF_HOME=/path/to/shared/hf_cache

bash cluster/setup_env.sh
source .venv/bin/activate
python cluster/prefetch_model.py --config configs/phase0.yaml
```

## Run the pilot (GPU node)

```bash
export HF_HOME=/path/to/shared/hf_cache   # same path as above
sbatch cluster/memory_pilot.slurm
```

Watch it and read the result:

```bash
squeue --me
tail -f logs/mem-pilot-<jobid>.out
cat runs/*-memory-pilot/memory_pilot_summary.json   # {"go": true, "viable_configs": [...]}
```

The job exits **0 = GO** (≥1 config fits both tracing & training under
`pilot.vram_budget_gb`) or **1 = NO-GO**. `memory_pilot.csv` has per-config
tracing vs. training peaks.

## Resuming after a timeout

The pilot writes `memory_pilot.csv` **incrementally** (one row per measurement),
so a job killed at the SLURM time limit keeps its progress. To continue where it
stopped, resubmit pointing at the same run dir — already-recorded
`{quant × seq_len × rank}` configs are skipped:

```bash
ls -td runs/*-memory-pilot | head -1                       # find the unfinished run dir
RESUME_DIR=runs/<ts>-memory-pilot CONDA_ENV=torch310 sbatch --partition=u22 cluster/memory_pilot.slurm
# or directly:
python scripts/memory_pilot.py --config configs/phase0.yaml --resume runs/<ts>-memory-pilot
```

`u22` has an infinite time limit, so the pilot normally finishes in one job; resume
is the safety net if you set a `--time` cap or the node is reclaimed.

## Before you submit — edit for your HP cluster

In `cluster/memory_pilot.slurm`:

- `--partition=gpu` → your GPU partition name (`sinfo` to list).
- `--account=` → uncomment if the scheduler requires an allocation account.
- **Environment:** set `CONDA_ENV=torch310` (or your torch env) to reuse an
  env that already has torch; otherwise it falls back to `.venv`.
- **CUDA modules (optional):** torch wheels bundle their own CUDA, so you
  usually don't need a module. If `bitsandbytes` complains, match the env's
  build (`python -c "import torch; print(torch.version.cuda)"`) and set, e.g.:
  ```bash
  export CUDA_MODULE=u22/cuda/12.4
  export CUDNN_MODULE=u22/cudnn/9.7.0-cuda-12.8
  ```
  Available on this cluster: `u22/cuda/{9.2,11.7,11.8,12.1,12.4,12.9}`,
  `u22/cudnn/{9.1.0-cuda-12,9.7.0-cuda-12.8}`. Note there is **no cuda/13**, so
  a torch built for cu13 must rely on its bundled runtime (no module load).

## Notes / gotchas

- **Disk quota exceeded (`OSError: [Errno 122]`) during install:** `torch` with
  no build constraint pulls the full CUDA wheel stack (`nvidia-*-cu13`,
  `cuda-toolkit`) — several GB — which overflows the small `/home2` quota. Two fixes:
  - **Reuse an existing torch** (recommended if a conda env like `torch310` is
    active and already has torch):
    ```bash
    cd circuit_routing
    USE_ACTIVE_ENV=1 bash cluster/setup_env.sh    # auto-detects torch, installs -e . --no-deps + light deps
    ```
  - **Keep all caches off home** (needed anyway for a fresh install): point
    caches at roomy scratch/share storage before running setup:
    ```bash
    export CACHE_ROOT=/scratch/$USER/br_cache   # pip/uv/TMPDIR caches
    export HF_HOME=/scratch/$USER/hf_cache      # model checkpoints
    export VENV_DIR=/scratch/$USER/br_venv      # (optional) venv off home too
    bash cluster/setup_env.sh
    ```
  - Clean up a half-finished install that ate quota:
    ```bash
    pip cache purge; rm -rf ~/.cache/pip ~/.cache/uv
    rm -rf circuit_routing/.venv        # if the fresh venv is the bloat
    quota -s 2>/dev/null || lfs quota -h -u $USER $HOME   # confirm free space
    ```
  - **Corrupted pip (`~ip`, `~ip-25.3.dist-info`) after a mid-uninstall quota
    failure:** a `pip install --upgrade pip` that dies partway leaves temp dirs
    and a broken pip. Recover:
    ```bash
    conda clean -a -y                                   # free space first
    cd ~/miniconda3/envs/torch310/lib/python3.10/site-packages
    rm -rf '~ip' '~ip-25.3.dist-info' ~[a-z]*           # remove ~-prefixed leftovers
    python -m pip --version || conda install -n torch310 --force-reinstall -y pip
    ```
    `setup_env.sh` now skips the pip upgrade in active-env mode (`SKIP_PIP_UPGRADE`).
- **`sbatch: Batch script contains DOS line breaks (\r\n)`:** scripts edited on
  Windows carry `\r\n`. A committed `.gitattributes` forces LF for `*.sh/*.slurm`,
  but if a file still has them, strip in place:
  ```bash
  sed -i 's/\r$//' cluster/*.slurm cluster/*.sh   # or: dos2unix cluster/*.slurm cluster/*.sh
  ```
- **Offline GPU nodes:** the SLURM job sets `HF_HUB_OFFLINE=1`, so the checkpoint
  **must** be prefetched first. Skipping `prefetch_model.py` → the job fails at
  model load.
- **Durable output:** `runs/` is written under the repo root (home/shared). Do
  **not** relocate it to node-local scratch — that is wiped when the job ends.
- **SSL / MITM proxy on install:** `setup_env.sh` falls back to `uv --system-certs`
  then `pip --trusted-host` if the default TLS chain is rejected.
- **Deliverable (Definition of Done):** `memory_pilot.csv` + ≥1 viable config
  under the 5 GB budget → the config the team freezes for Stage 1/2.
