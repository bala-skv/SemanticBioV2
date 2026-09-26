#!/usr/bin/env python3
"""
hook_smoke_test.py  â€”  Standalone smoke test for P0-6 hooks

Task:       P0-6 (deliverable: this script)
Owner:      Daniel Ashish Abraham
Depends on: P0-1 (env), P0-2 (verified model loading)

Usage
-----
# Run from the repo root:
python scripts/hook_smoke_test.py

# Explicit precision override:
python scripts/hook_smoke_test.py --precision 4bit
python scripts/hook_smoke_test.py --precision 8bit
python scripts/hook_smoke_test.py --precision bf16

# Optional: write JSON result
python scripts/hook_smoke_test.py --precision 4bit --output-json runs/hook_smoke_4bit.json

Definition of Done (P0-6)
--------------------------
This script must exit 0 with "PASS" across all three precision modes:
  4-bit  ->  shape [1, T, 12, 128] on all 28 layers
  8-bit  ->  shape [1, T, 12, 128] on all 28 layers
  bf16   ->  shape [1, T, 12, 128] on all 28 layers

Cross-references
----------------
- AttentionHeadCapture  <--  hooks.py (this repo)
- verify_base_checkpoint  <--  model_loading.py (Bala's P0-2 deliverable)
- REVIEW-3: Daniel reviews model_loading.py to confirm module layout matches
  hook assumptions.  Sign-off document: ashish/REVIEW-3-model-loading-signoff.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

# hooks.py lives at the repo root, one level above scripts/
_REPO_ROOT = Path(__file__).resolve().parent.parent   # ANLP-Group-Project/

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hooks import AttentionHeadCapture, random_input_ids, run_hook_smoke  # noqa: E402


# ---------------------------------------------------------------------------
# Model-loading helpers (self-contained so this script doesn't hard-depend on
# Bala's model_loading.py â€” but will use it if found on the path).
# ---------------------------------------------------------------------------

def _load_model_4bit(model_id: str, revision: str):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        quantization_config=bnb,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    model.config.use_cache = False
    return model


def _load_model_8bit(model_id: str, revision: str):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        quantization_config=bnb,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    model.config.use_cache = False
    return model


def _load_model_bf16(model_id: str, revision: str):
    from transformers import AutoModelForCausalLM
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=dtype,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    model.config.use_cache = False
    return model


_LOADERS = {
    "4bit": _load_model_4bit,
    "8bit": _load_model_8bit,
    "bf16": _load_model_bf16,
}

# ---------------------------------------------------------------------------
# Expected layout for Qwen2.5-1.5B  (verified by P0-2 / Bala)
# ---------------------------------------------------------------------------

EXPECTED = {
    "num_hidden_layers":    28,
    "num_attention_heads":  12,
    "num_key_value_heads":   2,
    "hidden_size":        1536,
    "head_dim":            128,
}

MODEL_ID = "Qwen/Qwen2.5-1.5B"
REVISION  = "8faed761d45a263340a0528343f099c05c9a4323"
SEQ_LEN   = 32   # short sequence â€” smoke test only, not a memory sweep


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P0-6 hook smoke test â€” verifies GQA-correct per-head "
                    "hooking across quantization modes."
    )
    parser.add_argument(
        "--precision",
        choices=list(_LOADERS),
        default="4bit",
        help="Quantization / precision to test (default: 4bit).",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=SEQ_LEN,
        help=f"Sequence length for the test forward pass (default: {SEQ_LEN}).",
    )
    parser.add_argument(
        "--model-id",
        default=MODEL_ID,
        help=f"HuggingFace model ID (default: {MODEL_ID}).",
    )
    parser.add_argument(
        "--revision",
        default=REVISION,
        help="Pinned commit SHA (default: the verified P0 revision).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write the result dict as JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # ------------------------------------------------------------------ env
    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU required for quantized model testing.")
        print("       Run on the Ada cluster (SLURM) or a Colab GPU runtime.")
        return 1

    precision = args.precision
    print(f"=== P0-6 Hook Smoke Test  |  precision={precision}  |  seq_len={args.seq_len} ===")
    print(f"Model  : {args.model_id}  @  {args.revision}")
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print()

    # ------------------------------------------------------------------ load
    print(f"Loading model in {precision} mode ...")
    try:
        load_fn = _LOADERS[precision]
        model   = load_fn(args.model_id, args.revision)
    except Exception as exc:
        print(f"FAIL  model load: {exc}")
        return 1

    print("Model loaded.  Running hook smoke test ...")

    # ------------------------------------------------------------------ test
    try:
        result = run_hook_smoke(model, EXPECTED, seq_len=args.seq_len)
    except AssertionError as exc:
        print(f"FAIL  hook assertion: {exc}")
        return 1
    except Exception as exc:
        print(f"FAIL  unexpected error: {exc}")
        return 1

    # ------------------------------------------------------------------ report
    print()
    print("Hook smoke test PASS")
    print(f"  Captured layers : {result['captured_layers']} / {result['expected_layers']}")
    print(f"  Shape each layer: {result['shape_each']}")
    print(f"  Target          : {result['target']}")
    print(f"  Handles removed : {result['handles_removed']}")
    print()

    # Validate the shape matches the GQA layout exactly.
    B, T, H, d = result["shape_each"]
    assert B == 1,                        f"Batch dim: expected 1, got {B}"
    assert T == args.seq_len,             f"Token dim: expected {args.seq_len}, got {T}"
    assert H == EXPECTED["num_attention_heads"], f"Head dim: expected 12, got {H}"
    assert d == EXPECTED["head_dim"],     f"head_dim: expected 128, got {d}"
    assert result["handles_removed"],     "Hook handles were not removed on exit"

    full_result = {
        "precision":   precision,
        "seq_len":     args.seq_len,
        "model_id":    args.model_id,
        "revision":    args.revision,
        "gpu":         torch.cuda.get_device_name(0),
        "status":      "PASS",
        **result,
    }

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(full_result, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Result written to: {args.output_json}")

    print(f"P0-6 RESULT: PASS  ({precision})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
