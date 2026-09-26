#!/usr/bin/env python
"""P0-2 gate: verify the pinned base checkpoint.

Exits 0 on PASS, 1 on FAIL, so run_phase0.py (P0-9) can chain it.

Usage:
    python scripts/verify_model.py
    python scripts/verify_model.py --precision 4bit
    python scripts/verify_model.py --json runs/verify_model.json

Owner: Balasubramanian
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Allow running as `python scripts/verify_model.py` from the package root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_loading import (  # noqa: E402
    LoadConfig,
    VerificationError,
    load_base_model,
    verify_base_checkpoint,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--precision",
        default="fp16",
        choices=["fp16", "bf16", "8bit", "4bit"],
        help="fp16 is the default; Stage 1 tracing should run in fp16 so "
             "quantization noise cannot flatten the recovery-score "
             "distribution",
    )
    p.add_argument("--json", type=Path, help="write a machine-readable report here")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    cfg = LoadConfig(precision=args.precision)
    report: dict = {
        "model_id": cfg.model_id,
        "revision": cfg.revision,
        "precision": cfg.precision,
        "notes": [],
        "passed": False,
    }

    try:
        model, tokenizer = load_base_model(cfg)
        notes = verify_base_checkpoint(model, tokenizer, cfg)
        report["notes"] = notes
        report["passed"] = True
    except (VerificationError, RuntimeError, OSError) as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2))

    if report["passed"]:
        for n in report["notes"]:
            print(f"  {n}")
        print("\nPASS  base checkpoint verified")
        return 0

    print(f"\nFAIL  {report['error']}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
