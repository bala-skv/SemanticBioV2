#!/usr/bin/env python3
"""
stage1_sanity_checks.py  -  GATE: four checks before any heatmap is built

Owner:    Daniel Ashish Abraham
Task:     Stage 1 Gate  (Fri 26 Sep -> Sat 27 Sep delivery)
Depends:  patching.py, phase0/model_loading.py

Definition of Done
------------------
All four checks PASS.  Exit 0 on full pass, exit 1 on any failure.
A broken patcher producing a plausible heatmap is the worst outcome.

Checks
------
  1. Patch all 336 heads (all layers)  -> recovery >= 0.75
  2. Patch zero heads                  -> recovery exactly 0.0
  3. Sweep: patch all heads in layer L -> smooth curve, not flat noise
  4. Re-inject captured acts unchanged -> logits bit-identical (allclose eps=0)

Note on sequence_scoring dependency
-------------------------------------
Bala's sequence_scoring.py is imported if present.  If not yet available,
checks 1-3 use an inline _score() that computes the same quantity (mean
per-token log-prob) so the gate can run standalone.  Replace with
sequence_scoring.recovery() once the file arrives.

Usage
-----
  python stage1/stage1_sanity_checks.py
  python stage1/stage1_sanity_checks.py --output-json runs/sanity.json
"""

from __future__ import annotations

import argparse
import json
import sys
import os
from pathlib import Path

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup: import patching.py (same dir) and phase0 modules
# ---------------------------------------------------------------------------

_STAGE1_DIR = Path(__file__).resolve().parent
_REPO_ROOT  = _STAGE1_DIR.parent

for _p in (_STAGE1_DIR, _REPO_ROOT, _REPO_ROOT / "phase0"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from patching import (
    capture_activations, patched_logits,
    OProjPatchAll, OProjPatchLayer,
    NUM_LAYERS, NUM_HEADS,
)
from model_loading import load_base_model, LoadConfig

# ---------------------------------------------------------------------------
# Inline scoring  (same math as Bala's score_sequence; replace if available)
# ---------------------------------------------------------------------------

def _score(model, input_ids: torch.Tensor, prompt_len: int) -> float:
    """Mean per-token log-prob of tokens after prompt_len."""
    with torch.no_grad():
        logits = model(input_ids=input_ids, use_cache=False).logits  # [1, T, V]
    T = input_ids.shape[1]
    # shift: predict tokens [prompt_len .. T-1] from positions [prompt_len-1 .. T-2]
    lp = F.log_softmax(logits[0, prompt_len - 1 : T - 1], dim=-1)
    tgt = input_ids[0, prompt_len:]
    return lp[torch.arange(T - prompt_len), tgt].mean().item()


def _contrast(logp_native: float, logp_foreign: float) -> float:
    """Higher = model prefers native target in current context."""
    return logp_native - logp_foreign


def _recovery(m_clean: float, m_corrupted: float, m_patched: float) -> float:
    denom = m_clean - m_corrupted
    if abs(denom) < 1e-6:
        raise ValueError(
            f"Near-zero denominator (m_clean={m_clean:.4f}, "
            f"m_corrupted={m_corrupted:.4f}). "
            "Choose a concept pair with larger contrast."
        )
    return (m_patched - m_corrupted) / denom


def _score_from_logits(logits: torch.Tensor, input_ids: torch.Tensor,
                       prompt_len: int) -> float:
    T = input_ids.shape[1]
    lp = F.log_softmax(logits[0, prompt_len - 1 : T - 1], dim=-1)
    tgt = input_ids[0, prompt_len:]
    return lp[torch.arange(T - prompt_len), tgt.to(logits.device)].mean().item()


# ---------------------------------------------------------------------------
# Hardcoded test pair  (sky_colour from Bala's tracing set)
# Used so the checks run standalone before tracing_set_v1.json arrives.
# ---------------------------------------------------------------------------

_SOURCE_PROMPT = "Aasman ka rang hai "       # hi_latn (Romanized Hindi): primary contrast
_SOURCE_TARGET = "neela"                      # hi_latn target
_TARGET_PROMPT = "The colour of the sky is " # en prompt
_TARGET_TARGET = "blue"                      # en target


def _make_ids(tok, prompt: str, target: str, device):
    p = tok(prompt, return_tensors="pt").input_ids.to(device)
    t = tok(target, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    return torch.cat([p, t], dim=1), p.shape[1]   # (full_ids, prompt_len)


# ---------------------------------------------------------------------------
# Four checks
# ---------------------------------------------------------------------------

def check1_all_heads(model, tok, device, threshold=0.75) -> dict:
    """Patch all 336 heads -> recovery should be ~1.0 (or at least > threshold)."""
    print("  Check 1: patch ALL 336 heads ...")

    src_ids, src_plen = _make_ids(tok, _SOURCE_PROMPT, _SOURCE_TARGET, device)
    tgt_ids, tgt_plen = _make_ids(tok, _TARGET_PROMPT, _TARGET_TARGET, device)
    src_patch_pos = src_plen - 1
    tgt_patch_pos = tgt_plen - 1

    # Clean: score under source (Hindi) prompts for both targets
    m_clean = _contrast(
        _score(model, src_ids, src_plen),
        _score(model, *_make_ids(tok, _SOURCE_PROMPT, _TARGET_TARGET, device)),
    )
    # Corrupted: score under target (English) prompt
    m_corrupted = _contrast(
        _score(model, *_make_ids(tok, _TARGET_PROMPT, _SOURCE_TARGET, device)),
        _score(model, tgt_ids, tgt_plen),
    )

    # Patched: all 336 heads substituted from Hindi run
    stored = capture_activations(model, src_ids, src_patch_pos)

    # Patch all layers at once
    with OProjPatchAll(model, tgt_patch_pos, stored):
        en_ids_native, en_plen_n = _make_ids(tok, _TARGET_PROMPT, _SOURCE_TARGET, device)
        en_ids_foreign, en_plen_f = _make_ids(tok, _TARGET_PROMPT, _TARGET_TARGET, device)
        logits_native  = model(input_ids=en_ids_native,  use_cache=False).logits
        logits_foreign = model(input_ids=en_ids_foreign, use_cache=False).logits

    m_patched = _contrast(
        _score_from_logits(logits_native,  en_ids_native,  en_plen_n),
        _score_from_logits(logits_foreign, en_ids_foreign, en_plen_f),
    )

    rec = _recovery(m_clean, m_corrupted, m_patched)
    passed = rec >= threshold
    result = {
        "check": "all_heads",
        "recovery": rec,
        "m_clean": m_clean,
        "m_corrupted": m_corrupted,
        "m_patched": m_patched,
        "threshold": threshold,
        "passed": passed,
    }
    status = "PASS" if passed else "FAIL"
    print(f"    recovery={rec:.4f}  (threshold>={threshold})  -> {status}")
    return result


def check2_zero_heads(model, tok, device) -> dict:
    """Patch zero heads -> recovery must be exactly 0.0."""
    print("  Check 2: patch ZERO heads ...")

    src_ids, src_plen = _make_ids(tok, _SOURCE_PROMPT, _SOURCE_TARGET, device)
    tgt_ids, tgt_plen = _make_ids(tok, _TARGET_PROMPT, _TARGET_TARGET, device)
    src_patch_pos = src_plen - 1
    tgt_patch_pos = tgt_plen - 1

    m_clean = _contrast(
        _score(model, src_ids, src_plen),
        _score(model, *_make_ids(tok, _SOURCE_PROMPT, _TARGET_TARGET, device)),
    )
    m_corrupted = _contrast(
        _score(model, *_make_ids(tok, _TARGET_PROMPT, _SOURCE_TARGET, device)),
        _score(model, tgt_ids, tgt_plen),
    )

    # Patched with no substitution (stored=None)
    stored = capture_activations(model, src_ids, src_patch_pos)
    _ = stored  # captured but not used -> no-op run below
    en_ids_n, en_plen_n = _make_ids(tok, _TARGET_PROMPT, _SOURCE_TARGET, device)
    en_ids_f, en_plen_f = _make_ids(tok, _TARGET_PROMPT, _TARGET_TARGET, device)
    logits_native  = patched_logits(model, en_ids_n, tgt_patch_pos, None, None, None)
    logits_foreign = patched_logits(model, en_ids_f, tgt_patch_pos, None, None, None)

    m_patched = _contrast(
        _score_from_logits(logits_native,  en_ids_n, en_plen_n),
        _score_from_logits(logits_foreign, en_ids_f, en_plen_f),
    )

    rec = _recovery(m_clean, m_corrupted, m_patched)
    passed = abs(rec) < 1e-5   # must be exactly 0.0
    result = {
        "check": "zero_heads",
        "recovery": rec,
        "passed": passed,
    }
    status = "PASS" if passed else "FAIL"
    print(f"    recovery={rec:.6f}  (expected 0.0)  -> {status}")
    return result


def check3_layer_sweep(model, tok, device) -> dict:
    """Patch all heads in layer L, sweep L=0..27 -> a curve, not flat noise."""
    print("  Check 3: per-layer sweep ...")

    src_ids, src_plen = _make_ids(tok, _SOURCE_PROMPT, _SOURCE_TARGET, device)
    tgt_ids, tgt_plen = _make_ids(tok, _TARGET_PROMPT, _TARGET_TARGET, device)
    src_patch_pos = src_plen - 1
    tgt_patch_pos = tgt_plen - 1

    m_clean = _contrast(
        _score(model, src_ids, src_plen),
        _score(model, *_make_ids(tok, _SOURCE_PROMPT, _TARGET_TARGET, device)),
    )
    en_ids_n, en_plen_n = _make_ids(tok, _TARGET_PROMPT, _SOURCE_TARGET, device)
    en_ids_f, en_plen_f = _make_ids(tok, _TARGET_PROMPT, _TARGET_TARGET, device)
    m_corrupted = _contrast(
        _score(model, en_ids_n, en_plen_n),
        _score(model, en_ids_f, en_plen_f),
    )

    stored = capture_activations(model, src_ids, src_patch_pos)
    recoveries = []

    for L in range(NUM_LAYERS):
        with OProjPatchLayer(model, tgt_patch_pos, L, stored):
            lg_n = model(input_ids=en_ids_n, use_cache=False).logits
            lg_f = model(input_ids=en_ids_f, use_cache=False).logits
        m_p = _contrast(
            _score_from_logits(lg_n, en_ids_n, en_plen_n),
            _score_from_logits(lg_f, en_ids_f, en_plen_f),
        )
        try:
            rec = _recovery(m_clean, m_corrupted, m_p)
        except ValueError:
            rec = 0.0
        recoveries.append(rec)
        print(f"    L={L:02d}  recovery={rec:.4f}")

    # Curve check: std > 0.01 (not flat noise)
    import statistics
    std = statistics.stdev(recoveries)
    passed = std > 0.01
    result = {
        "check": "layer_sweep",
        "recoveries": recoveries,
        "std": std,
        "passed": passed,
    }
    status = "PASS" if passed else "FAIL"
    print(f"    std={std:.4f}  (threshold>0.01)  -> {status}")
    return result


def check4_bit_identical(model, tok, device) -> dict:
    """Re-inject captured activations unchanged -> logits must be bit-identical."""
    print("  Check 4: re-inject unchanged activations ...")

    src_ids, src_plen = _make_ids(tok, _SOURCE_PROMPT, _SOURCE_TARGET, device)
    patch_pos = src_plen - 1

    # Capture from source run
    stored = capture_activations(model, src_ids, patch_pos)

    # Unpatched run (baseline)
    with torch.no_grad():
        logits_base = model(input_ids=src_ids, use_cache=False).logits.clone()

    # Re-inject ALL layers with the same values we captured
    from patching import OProjPatchAll
    with OProjPatchAll(model, patch_pos, stored):
        with torch.no_grad():
            logits_reinjected = model(input_ids=src_ids, use_cache=False).logits.clone()

    identical = torch.equal(logits_base, logits_reinjected)
    # fp16 re-injection may have sub-ulp differences; report both
    allclose  = torch.allclose(logits_base, logits_reinjected, atol=0, rtol=0)
    max_diff  = (logits_base - logits_reinjected).abs().max().item()

    passed = identical or allclose
    result = {
        "check": "bit_identical",
        "bit_exact": bool(identical),
        "allclose_zero_tol": bool(allclose),
        "max_abs_diff": max_diff,
        "passed": passed,
    }
    status = "PASS" if passed else "FAIL"
    print(f"    bit_exact={identical}  max_diff={max_diff:.2e}  -> {status}")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Stage 1 sanity gate checks.")
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--skip-layer-sweep", action="store_true",
                   help="Skip check 3 (28 forward passes; use for quick CI).")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA required. Run on Ada (SLURM) or Colab GPU.")
        return 1

    device = torch.device("cuda", 0)
    print(f"GPU : {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print()

    print("Loading model (fp16 trace precision) ...")
    cfg   = LoadConfig(precision="fp16")
    model, tok = load_base_model(cfg)
    model.eval()
    print(f"Loaded.  Layers={len(model.model.layers)}\n")

    results = []
    failed  = []

    print("=== Stage 1 Sanity Checks ===")

    r = check1_all_heads(model, tok, device)
    results.append(r)
    if not r["passed"]: failed.append("check1_all_heads")

    r = check2_zero_heads(model, tok, device)
    results.append(r)
    if not r["passed"]: failed.append("check2_zero_heads")

    if not args.skip_layer_sweep:
        r = check3_layer_sweep(model, tok, device)
        results.append(r)
        if not r["passed"]: failed.append("check3_layer_sweep")
    else:
        print("  Check 3: SKIPPED (--skip-layer-sweep)")

    r = check4_bit_identical(model, tok, device)
    results.append(r)
    if not r["passed"]: failed.append("check4_bit_identical")

    print()
    if failed:
        print(f"GATE FAIL  - {len(failed)} check(s) failed: {failed}")
        print("Do NOT proceed to heatmap generation until all checks pass.")
        verdict = "FAIL"
        exit_code = 1
    else:
        print("GATE PASS  - all checks passed. Proceed to pilot trace.")
        verdict = "PASS"
        exit_code = 0

    payload = {
        "verdict": verdict,
        "gpu": torch.cuda.get_device_name(0),
        "checks": results,
    }

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Results -> {args.output_json}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())