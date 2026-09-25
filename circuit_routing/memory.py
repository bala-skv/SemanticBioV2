"""Peak-VRAM measurement utilities.

The pilot needs the **peak** allocation of a workload, not a point-in-time
reading, so :class:`MemoryTracker` resets CUDA's peak counter on entry and reads
it on exit. All figures are reported in GiB.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

_BYTES_PER_GIB = 1024 ** 3


def cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def device_name() -> Optional[str]:
    if not cuda_available():
        return None
    import torch

    return torch.cuda.get_device_name(torch.cuda.current_device())


def total_vram_gib() -> Optional[float]:
    if not cuda_available():
        return None
    import torch

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return props.total_memory / _BYTES_PER_GIB


@dataclass
class MemoryTracker:
    """Context manager recording peak allocated/reserved VRAM in GiB.

    Example::

        with MemoryTracker() as m:
            run_workload()
        print(m.result)   # {"peak_allocated_gib": ..., "peak_reserved_gib": ...}
    """

    result: Dict[str, Optional[float]] = field(default_factory=dict)
    _enabled: bool = False

    def __enter__(self) -> "MemoryTracker":
        self._enabled = cuda_available()
        if self._enabled:
            import torch

            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._enabled:
            import torch

            torch.cuda.synchronize()
            self.result = {
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / _BYTES_PER_GIB,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / _BYTES_PER_GIB,
            }
        else:
            self.result = {"peak_allocated_gib": None, "peak_reserved_gib": None}
        return False  # never suppress exceptions

    @property
    def peak_reserved_gib(self) -> Optional[float]:
        return self.result.get("peak_reserved_gib")
