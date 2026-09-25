"""Reproducibility helpers: one call to fix every RNG the project touches."""
from __future__ import annotations

import os
import random


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and (if present) PyTorch, and optionally force
    deterministic CUDA kernels.

    Deterministic mode trades a little speed for run-to-run reproducibility,
    which is what we want when comparing routing masks at a matched budget.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            # cuBLAS determinism for matmul-heavy workloads.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:  # older torch without warn_only
                torch.use_deterministic_algorithms(True)
    except ImportError:
        pass
