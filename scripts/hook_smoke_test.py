"""Confirm per-head activation hooking works *under quantization* (GQA-aware).

Usage:
    python scripts/hook_smoke_test.py --config configs/phase0.yaml --quant 4bit

Loads the model at the requested quantization, registers per-head capture hooks
on ``q_proj`` across all layers, runs one forward pass, and asserts every layer
produced an activation of shape ``[B, T, num_query_heads, head_dim]``.
"""
import _bootstrap  # noqa: F401

import argparse

from circuit_routing.config import load_config
from circuit_routing.hooks import AttentionHeadCapture
from circuit_routing.logging_utils import get_logger, make_run_dir, write_json
from circuit_routing.model_loading import (
    load_model,
    load_tokenizer,
    verify_base_checkpoint,
)
from circuit_routing.seeding import set_seed


def main() -> int:
    parser = argparse.ArgumentParser(description="Attention-head hook smoke test.")
    parser.add_argument("--config", default="configs/phase0.yaml")
    parser.add_argument("--quant", default="4bit", choices=["4bit", "8bit", "bf16"])
    parser.add_argument("--projection", default="q_proj")
    parser.add_argument("--seq-len", type=int, default=32)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.repro.seeds[0], cfg.repro.deterministic)
    run_dir = make_run_dir(cfg.paths.output_dir, "hook-smoke")
    log = get_logger("hook_smoke_test", run_dir)

    import torch

    tokenizer = load_tokenizer(cfg.model)
    log.info("Loading model %s (quant=%s)", cfg.model.name, args.quant)
    model = load_model(cfg.model, cfg.quant, quant_mode=args.quant)
    model.eval()

    layout = verify_base_checkpoint(cfg.model, model.config, tokenizer)

    prompt = "The quick brown fox. " * 8
    batch = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=args.seq_len
    )
    batch = {k: v.to(model.device) for k, v in batch.items()}
    seq_len = batch["input_ids"].shape[1]

    with AttentionHeadCapture(model, layout, projection=args.projection) as cap:
        with torch.no_grad():
            model(**batch)
        expected = cap.expected_shape(1, seq_len)
        captured = {i: tuple(a.shape) for i, a in cap.activations.items()}

    n_expected_layers = layout.num_layers
    all_ok = (
        len(captured) == n_expected_layers
        and all(shape == expected for shape in captured.values())
    )

    result = {
        "quant": args.quant,
        "projection": args.projection,
        "expected_shape": list(expected),
        "num_layers_captured": len(captured),
        "num_layers_expected": n_expected_layers,
        "ok": bool(all_ok),
    }
    write_json(run_dir / "hook_smoke.json", result)

    if all_ok:
        log.info(
            "HOOKS OK: captured %d/%d layers, each %s",
            len(captured), n_expected_layers, expected,
        )
        return 0

    log.error("HOOK MISMATCH. expected %s across %d layers", expected, n_expected_layers)
    for i, shape in sorted(captured.items()):
        if shape != expected:
            log.error("  layer %d -> %s", i, shape)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
