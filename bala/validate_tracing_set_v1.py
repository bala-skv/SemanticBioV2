#!/usr/bin/env python
"""Validate the three-condition tracing set.

For every concept and every contrast pair (A, B) this computes the four
log-probs that Recovery depends on:

    logp(tgt_A | prompt_A)   logp(tgt_B | prompt_A)
    logp(tgt_A | prompt_B)   logp(tgt_B | prompt_B)

    m_clean     = logp(tgt_A | prompt_A) - logp(tgt_B | prompt_A)
    m_corrupted = logp(tgt_A | prompt_B) - logp(tgt_B | prompt_B)
    contrast    = m_clean - m_corrupted

`contrast` is the denominator of Recovery. Near zero -> the item carries no
signal. Very large -> suspect the metric is measuring something cruder than
intended (see the script-domination note below).

Fixes over v0 validator:
  - MIN_LOGP now scales with reduction (a summed log-prob over a 5-token
    Devanagari target sits near -12 by construction; the v0 fixed floor of
    -8.0 rejected 17/20 items for the wrong reason)
  - all four raw log-probs are reported, not just the difference, so the
    script effect is a quotable number
  - three contrast pairs instead of one

Reading the output
------------------
If en<->hi_deva shows a much larger contrast than en<->hi_latn, the metric is
dominated by script prediction rather than language: the model has learned
"Devanagari prompt -> Devanagari continuation" as a near-absolute rule, and
the cross terms are catastrophically unlikely for reasons that have nothing
to do with the concept. That is a finding, and it is why hi_latn exists.

Usage:
    python scripts/validate_tracing_set_v1.py
    python scripts/validate_tracing_set_v1.py --reduction sum
    python scripts/validate_tracing_set_v1.py --pairs en:hi_latn
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_loading import LoadConfig, load_base_model     # noqa: E402
from sequence_scoring import score_sequence               # noqa: E402

DEFAULT_PAIRS = [("en", "hi_deva"), ("en", "hi_latn"), ("hi_deva", "hi_latn")]

# Per-token floor. Scaled by target length when reduction="sum" so the same
# threshold means the same thing under both reductions.
MIN_LOGP_PER_TOKEN = -8.0
MIN_CONTRAST = 0.5


def _floor(reduction: str, n_tokens: int) -> float:
    return MIN_LOGP_PER_TOKEN * (n_tokens if reduction == "sum" else 1)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/tracing_set_v1.json"))
    p.add_argument("--out", type=Path, default=Path("runs/tracing_validation_v1.json"))
    p.add_argument("--reduction", default="mean", choices=["mean", "sum"])
    p.add_argument("--precision", default="fp16")
    p.add_argument("--pairs", nargs="*", default=None,
                   help="e.g. en:hi_latn hi_deva:hi_latn")
    args = p.parse_args()

    pairs = DEFAULT_PAIRS
    if args.pairs:
        pairs = [tuple(s.split(":")) for s in args.pairs]

    items = json.loads(args.data.read_text(encoding="utf-8"))
    model, tok = load_base_model(LoadConfig(precision=args.precision))

    results: dict[str, list[dict]] = {}

    for a, b in pairs:
        key = f"{a}<->{b}"
        rows = []
        for it in items:
            pa, ta = it[f"{a}_prompt"], it[f"{a}_target"]
            pb, tb = it[f"{b}_prompt"], it[f"{b}_target"]

            aa = score_sequence(model, tok, pa, ta, args.reduction)  # tgt A on prompt A
            ba = score_sequence(model, tok, pa, tb, args.reduction)  # tgt B on prompt A
            ab = score_sequence(model, tok, pb, ta, args.reduction)  # tgt A on prompt B
            bb = score_sequence(model, tok, pb, tb, args.reduction)  # tgt B on prompt B

            m_clean = aa.logp - ba.logp
            m_corrupted = ab.logp - bb.logp
            contrast = m_clean - m_corrupted

            usable = (
                contrast >= MIN_CONTRAST
                and aa.logp >= _floor(args.reduction, aa.n_target_tokens)
                and bb.logp >= _floor(args.reduction, bb.n_target_tokens)
            )

            rows.append({
                "concept_id": it["concept_id"],
                "category": it["category"],
                # all four raw log-probs
                "logp_tgtA_on_promptA": round(aa.logp, 4),
                "logp_tgtB_on_promptA": round(ba.logp, 4),
                "logp_tgtA_on_promptB": round(ab.logp, 4),
                "logp_tgtB_on_promptB": round(bb.logp, 4),
                # how much of the contrast comes from cross terms being
                # implausible rather than from the correct terms being likely
                "cross_penalty": round((aa.logp - ab.logp) + (bb.logp - ba.logp), 4),
                "n_tokens_A": aa.n_target_tokens,
                "n_tokens_B": bb.n_target_tokens,
                "patch_position_A": aa.patch_position,
                "patch_position_B": bb.patch_position,
                "m_clean": round(m_clean, 4),
                "m_corrupted": round(m_corrupted, 4),
                "contrast": round(contrast, 4),
                "usable": usable,
            })
        results[key] = rows

    # ---------------- report ----------------
    summary = {"reduction": args.reduction, "pairs": {}}

    for key, rows in results.items():
        keep = [r for r in rows if r["usable"]]
        contrasts = [r["contrast"] for r in rows]
        summary["pairs"][key] = {
            "n_items": len(rows),
            "n_usable": len(keep),
            "mean_contrast": round(sum(contrasts) / len(contrasts), 3),
            "min_contrast": round(min(contrasts), 3),
            "max_contrast": round(max(contrasts), 3),
            "mean_cross_penalty": round(
                sum(r["cross_penalty"] for r in rows) / len(rows), 3
            ),
        }

        print(f"\n{'=' * 68}")
        print(f"  {key}   ({args.reduction} reduction)")
        print("=" * 68)
        print(f"{'concept':<18} {'A|A':>8} {'B|A':>8} {'A|B':>8} {'B|B':>8} {'contr':>8}  ok")
        print("-" * 68)
        for r in sorted(rows, key=lambda r: -r["contrast"]):
            print(f"{r['concept_id']:<18} "
                  f"{r['logp_tgtA_on_promptA']:>8.2f} "
                  f"{r['logp_tgtB_on_promptA']:>8.2f} "
                  f"{r['logp_tgtA_on_promptB']:>8.2f} "
                  f"{r['logp_tgtB_on_promptB']:>8.2f} "
                  f"{r['contrast']:>8.2f}   {'y' if r['usable'] else 'n'}")
        s = summary["pairs"][key]
        print(f"\n  usable {s['n_usable']}/{s['n_items']}   "
              f"contrast mean {s['mean_contrast']} "
              f"[{s['min_contrast']}, {s['max_contrast']}]")
        bad = [r["concept_id"] for r in rows if not r["usable"]]
        if bad:
            print(f"  rewrite/drop: {', '.join(bad)}")

    # ---------------- the question that matters ----------------
    print(f"\n{'=' * 68}")
    print("  LANGUAGE vs SCRIPT")
    print("=" * 68)
    sp = summary["pairs"]
    if "en<->hi_deva" in sp and "en<->hi_latn" in sp:
        deva = sp["en<->hi_deva"]["mean_contrast"]
        latn = sp["en<->hi_latn"]["mean_contrast"]
        ratio = deva / latn if latn else float("inf")
        summary["deva_latn_contrast_ratio"] = round(ratio, 2)
        print(f"  en<->hi_deva mean contrast : {deva}")
        print(f"  en<->hi_latn mean contrast : {latn}")
        print(f"  ratio                      : {ratio:.2f}x")
        if ratio > 2.0:
            print("\n  The Devanagari contrast is much larger. The metric is")
            print("  substantially driven by SCRIPT, not language. Stage 1 scores")
            print("  from en<->hi_deva alone would be a script map. Report all")
            print("  three pairs; treat en<->hi_latn as the language contrast.")
        else:
            print("\n  Contrasts are comparable, so the Devanagari pair is not")
            print("  dominated by script prediction. Still report all three.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"summary": summary, "pairs": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nreport -> {args.out}")

    worst = min(s["n_usable"] for s in summary["pairs"].values())
    return 0 if worst >= 15 else 1


if __name__ == "__main__":
    raise SystemExit(main())
