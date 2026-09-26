"""Cluster environment verifier — Jawed's "verify CUDA + bitsandbytes" gate.

Confirms the GPU node can actually run the pilot BEFORE the (slow) model load:

  1. torch is importable and reports a CUDA device,
  2. the device + total VRAM are printed (sanity vs. the 5 GB budget),
  3. bitsandbytes imports AND its 4-bit CUDA kernels actually execute
     (a tiny Linear4bit forward pass) — this is the thing that silently breaks
     on the wrong CUDA / bnb build.

Exit code is 0 only if every check passes, so it can fail a SLURM job fast.

    python cluster/check_env.py
    python cluster/check_env.py --json      # machine-readable summary
"""
import argparse
import json
import pathlib
import sys

# Allow running before `pip install -e .` (repo root on sys.path).
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_BYTES_PER_GIB = 1024 ** 3


def _check_torch_cuda(report: dict) -> bool:
    try:
        import torch
    except Exception as err:  # noqa: BLE001
        report["torch"] = {"ok": False, "error": f"{type(err).__name__}: {err}"}
        return False

    info = {
        "ok": True,
        "torch_version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if not torch.cuda.is_available():
        info["ok"] = False
        info["error"] = "torch.cuda.is_available() is False (no GPU visible to this process)"
        report["torch"] = info
        return False

    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    info["device_name"] = props.name
    info["total_vram_gib"] = round(props.total_memory / _BYTES_PER_GIB, 2)
    info["capability"] = f"{props.major}.{props.minor}"
    report["torch"] = info
    return True


def _check_bitsandbytes(report: dict) -> bool:
    """Import bnb and run a real Linear4bit forward on the GPU."""
    try:
        import torch
        import bitsandbytes as bnb
    except Exception as err:  # noqa: BLE001
        report["bitsandbytes"] = {"ok": False, "error": f"{type(err).__name__}: {err}"}
        return False

    info = {"ok": True, "version": getattr(bnb, "__version__", "unknown")}
    try:
        from bitsandbytes.nn import Linear4bit

        layer = Linear4bit(64, 64, bias=False, compute_dtype=torch.bfloat16).cuda()
        x = torch.randn(2, 64, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            y = layer(x)
        torch.cuda.synchronize()
        info["linear4bit_output_shape"] = list(y.shape)
    except Exception as err:  # noqa: BLE001
        info["ok"] = False
        info["error"] = f"Linear4bit forward failed: {type(err).__name__}: {err}"
    report["bitsandbytes"] = info
    return info["ok"]


def _check_peft(report: dict) -> bool:
    try:
        import peft
        import transformers

        report["peft"] = {
            "ok": True,
            "peft_version": peft.__version__,
            "transformers_version": transformers.__version__,
        }
        return True
    except Exception as err:  # noqa: BLE001
        report["peft"] = {"ok": False, "error": f"{type(err).__name__}: {err}"}
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify CUDA + bitsandbytes on the cluster GPU node.")
    parser.add_argument("--json", action="store_true", help="print the report as JSON only")
    args = parser.parse_args()

    report: dict = {}
    ok_torch = _check_torch_cuda(report)
    ok_peft = _check_peft(report)
    # Only meaningful if CUDA is up; still record the attempt otherwise.
    ok_bnb = _check_bitsandbytes(report) if ok_torch else False

    report["all_ok"] = bool(ok_torch and ok_peft and ok_bnb)

    if args.json:
        print(json.dumps(report, indent=2))
        return 0 if report["all_ok"] else 1

    line = "-" * 60
    print(line)
    print("  Phase 0 — cluster environment check")
    print(line)
    t = report["torch"]
    if t.get("ok"):
        print(f"  [PASS] torch {t['torch_version']} (CUDA build {t['cuda_build']})")
        print(f"         device : {t['device_name']}  (cc {t['capability']})")
        print(f"         VRAM   : {t['total_vram_gib']} GiB total")
    else:
        print(f"  [FAIL] torch/CUDA: {t.get('error')}")

    p = report["peft"]
    if p.get("ok"):
        print(f"  [PASS] peft {p['peft_version']} / transformers {p['transformers_version']}")
    else:
        print(f"  [FAIL] peft/transformers: {p.get('error')}")

    b = report["bitsandbytes"]
    if b.get("ok"):
        print(f"  [PASS] bitsandbytes {b['version']} — Linear4bit forward OK "
              f"{b.get('linear4bit_output_shape')}")
    else:
        print(f"  [FAIL] bitsandbytes: {b.get('error')}")

    print(line)
    if report["all_ok"]:
        print("  GO: environment is ready for the memory pilot.")
        return 0
    print("  NO-GO: fix the failing check(s) above before submitting the pilot.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
