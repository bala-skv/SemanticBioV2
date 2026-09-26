"""Verify the checkpoint is the Qwen2.5-1.5B *base* model and report its layout.

Usage:
    python scripts/verify_model.py --config configs/phase0.yaml

Exits non-zero if the checkpoint looks instruction-tuned or its GQA layout does
not match the expectations in the config. Only the *config* is downloaded, so
this is cheap and safe to run as a gate.
"""
import _bootstrap  # noqa: F401

import argparse

from circuit_routing.config import load_config
from circuit_routing.logging_utils import get_logger, make_run_dir, write_json
from circuit_routing.model_loading import (
    CheckpointVerificationError,
    head_layout_from_config,
    load_tokenizer,
    verify_base_checkpoint,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the base checkpoint.")
    parser.add_argument("--config", default="configs/phase0.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = make_run_dir(cfg.paths.output_dir, "verify-model")
    log = get_logger("verify_model", run_dir)

    from transformers import AutoConfig

    log.info("Loading config for %s (revision=%s)", cfg.model.name, cfg.model.revision)
    hf_config = AutoConfig.from_pretrained(
        cfg.model.name,
        revision=cfg.model.revision,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    tokenizer = load_tokenizer(cfg.model)

    result = {"model": cfg.model.name, "revision": cfg.model.revision, "ok": False}
    try:
        layout = verify_base_checkpoint(cfg.model, hf_config, tokenizer)
        result.update(
            ok=True,
            num_layers=layout.num_layers,
            num_attention_heads=layout.num_attention_heads,
            num_key_value_heads=layout.num_key_value_heads,
            group_size=layout.group_size,
            head_dim=layout.head_dim,
            hidden_size=layout.hidden_size,
        )
        log.info("VERIFIED base checkpoint.")
        log.info(
            "  layers=%d  q_heads=%d  kv_groups=%d  group_size=%d  head_dim=%d  hidden=%d",
            layout.num_layers,
            layout.num_attention_heads,
            layout.num_key_value_heads,
            layout.group_size,
            layout.head_dim,
            layout.hidden_size,
        )
    except CheckpointVerificationError as err:
        layout = head_layout_from_config(hf_config)
        result["error"] = str(err)
        log.error("VERIFICATION FAILED: %s", err)

    write_json(run_dir / "verify_model.json", result)
    log.info("Wrote %s", run_dir / "verify_model.json")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
