"""Run directories, logging, and structured result writers.

Every Phase-0 invocation gets its own timestamped run directory containing a
config snapshot, a human-readable log, and machine-readable results (JSON/CSV)
so pilots are reproducible and comparable after the fact.
"""
from __future__ import annotations

import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def make_run_dir(output_dir: str | Path, tag: str) -> Path:
    """Create ``<output_dir>/<tag>-<timestamp>/`` and return it."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(output_dir) / f"{tag}-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def get_logger(name: str, run_dir: Path | None = None) -> logging.Logger:
    """A logger that writes to stdout and (optionally) ``run_dir/run.log``."""
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured
        return logger
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", "%H:%M:%S")

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    if run_dir is not None:
        file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


def write_json(path: str | Path, payload) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)


def write_csv(path: str | Path, rows: Sequence[Mapping], fieldnames: Iterable[str] | None = None) -> None:
    rows = list(rows)
    if not rows:
        return
    fieldnames = list(fieldnames) if fieldnames else list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
