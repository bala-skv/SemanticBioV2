"""Run every Phase-0 gate and print a consolidated go/no-go summary.

Usage:
    python scripts/run_phase0.py --config configs/phase0.yaml
    python scripts/run_phase0.py --config configs/phase0.yaml --skip memory_pilot

Gates (each writes its own artifacts under ``runs/``):
    1. verify_model     — base checkpoint identity + GQA layout
    2. hook_smoke_test  — per-head hooking under 4-bit quantization
    3. memory_pilot     — 4-5 GB budget sweep (tracing vs training)

Exits non-zero if any executed gate fails, so it can gate a cluster job before
Stage 1 begins.
"""
import _bootstrap  # noqa: F401

import argparse
import pathlib
import subprocess
import sys

GATES = ["verify_model", "hook_smoke_test", "memory_pilot"]
_SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent


def _run_gate(name: str, argv: list) -> int:
    """Run a sibling script as a subprocess and return its exit code."""
    script = _SCRIPTS_DIR / f"{name}.py"
    proc = subprocess.run([sys.executable, str(script), *argv])
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all Phase-0 gates.")
    parser.add_argument("--config", default="configs/phase0.yaml")
    parser.add_argument("--quant", default="4bit", help="quant mode for the hook smoke test")
    parser.add_argument("--skip", nargs="*", default=[], choices=GATES,
                        help="gates to skip")
    args = parser.parse_args()

    gate_argv = {
        "verify_model": ["--config", args.config],
        "hook_smoke_test": ["--config", args.config, "--quant", args.quant],
        "memory_pilot": ["--config", args.config],
    }

    results = {}
    for gate in GATES:
        if gate in args.skip:
            results[gate] = "skipped"
            continue
        print(f"\n{'=' * 64}\n  GATE: {gate}\n{'=' * 64}")
        code = _run_gate(gate, gate_argv[gate])
        results[gate] = "pass" if code == 0 else "FAIL"

    print(f"\n{'=' * 64}\n  PHASE 0 SUMMARY\n{'=' * 64}")
    for gate in GATES:
        print(f"  {gate:<18} {results[gate]}")

    failed = [g for g, r in results.items() if r == "FAIL"]
    if failed:
        print(f"\nNO-GO: {', '.join(failed)} failed. Resolve before Stage 1.")
        return 1
    print("\nGO: all executed gates passed. Cleared for Stage 1 (causal localization).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
