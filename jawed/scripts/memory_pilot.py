"""Memory pilot — the Phase-0 go/no-go on the 4-5 GB budget.

Sweeps ``{quant} x {seq_len} x {rank}`` and records **peak VRAM separately** for:

* the **tracing** profile   — a forward pass with per-head capture hooks active
  (what Stage 1 activation patching costs), and
* the **training** profile  — a LoRA forward+backward+optimizer step with gradient
  accumulation (what Stage 2 fine-tuning costs).

A ``(quant, seq_len, rank)`` combination is *viable* only if BOTH profiles fit
under ``pilot.vram_budget_gb``. The script exits 0 iff at least one combination
is viable.

Usage:
    python scripts/memory_pilot.py --config configs/phase0.yaml
"""
import _bootstrap  # noqa: F401

import argparse
import csv
import gc
from pathlib import Path
from typing import Dict, List, Optional

from circuit_routing.config import load_config
from circuit_routing.hooks import AttentionHeadCapture
from circuit_routing.logging_utils import get_logger, make_run_dir, write_csv, write_json
from circuit_routing.memory import MemoryTracker, device_name, total_vram_gib
from circuit_routing.model_loading import load_model, load_tokenizer, verify_base_checkpoint
from circuit_routing.seeding import set_seed

# Fixed CSV schema so incremental writes stay consistent whether or not a row
# carries an "error" field (DictWriter would otherwise choke on the extra key).
_FIELDNAMES = [
    "profile", "quant", "seq_len", "rank",
    "peak_reserved_gib", "peak_allocated_gib", "error",
]


def _row_key(profile, quant, seq_len, rank):
    """Identity of a single measurement, used to skip already-done work on resume."""
    return (
        profile, quant,
        None if seq_len in (None, "") else int(seq_len),
        None if rank in (None, "", "None") else int(rank),
    )


def _load_existing_rows(csv_path: Path) -> List[dict]:
    """Read a prior run's memory_pilot.csv (for --resume), normalising types."""
    if not csv_path.exists():
        return []
    rows: List[dict] = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            r["seq_len"] = None if r.get("seq_len") in (None, "") else int(r["seq_len"])
            r["rank"] = None if r.get("rank") in (None, "", "None") else int(r["rank"])
            for k in ("peak_reserved_gib", "peak_allocated_gib"):
                v = r.get(k)
                r[k] = None if v in (None, "", "None") else float(v)
            rows.append(r)
    return rows


def _record(rows: List[dict], done: set, csv_path: Path, row: dict) -> None:
    """Append a measurement and flush the whole CSV so progress survives a kill."""
    rows.append(row)
    done.add(_row_key(row["profile"], row["quant"], row["seq_len"], row["rank"]))
    write_csv(csv_path, rows, fieldnames=_FIELDNAMES)


def _free(model) -> None:
    import torch

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _dummy_batch(model, seq_len: int, batch_size: int):
    import torch

    vocab = model.config.vocab_size
    input_ids = torch.randint(0, vocab, (batch_size, seq_len), device=model.device)
    attn = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attn}


def measure_tracing(model, layout, seq_len: int, batch_size: int) -> Dict[str, Optional[float]]:
    import torch

    batch = _dummy_batch(model, seq_len, batch_size)
    model.eval()
    with MemoryTracker() as mem:
        with AttentionHeadCapture(model, layout, projection="q_proj"):
            with torch.no_grad():
                model(**batch)
    return mem.result


def measure_training(cfg, model, seq_len: int, rank: int,
                     quant: str = "bf16") -> Dict[str, Optional[float]]:
    import torch
    from peft import LoraConfig, get_peft_model

    kbit = quant in ("4bit", "8bit")
    if kbit:
        # Frozen quantized base weights don't carry grad; this casts norms to
        # fp32, enables input require_grads and wires gradient checkpointing so
        # the LoRA path is differentiable (else backward -> "element 0 ... does
        # not require grad").
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True)
    elif hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    lora = LoraConfig(
        r=rank,
        lora_alpha=cfg.pilot.lora_alpha,
        lora_dropout=cfg.pilot.lora_dropout,
        target_modules=list(cfg.pilot.lora_target_modules),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)

    batch = _dummy_batch(model, seq_len, cfg.pilot.batch_size)
    labels = batch["input_ids"].clone()

    with MemoryTracker() as mem:
        optimizer.zero_grad(set_to_none=True)
        for _ in range(cfg.pilot.grad_accum_steps):
            out = model(**batch, labels=labels)
            (out.loss / cfg.pilot.grad_accum_steps).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return mem.result


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase-0 memory pilot.")
    parser.add_argument("--config", default="configs/phase0.yaml")
    parser.add_argument(
        "--resume", default=None,
        help="Existing run dir to resume: measurements already in its "
             "memory_pilot.csv are skipped (safe to rerun after a SLURM timeout).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.repro.seeds[0], cfg.repro.deterministic)
    if args.resume:
        run_dir = Path(args.resume)
        if not run_dir.exists():
            raise SystemExit(f"--resume dir not found: {run_dir}")
    else:
        run_dir = make_run_dir(cfg.paths.output_dir, "memory-pilot")
    log = get_logger("memory_pilot", run_dir)

    budget = cfg.pilot.vram_budget_gb
    log.info("Device: %s  (total VRAM: %s GiB)", device_name(), total_vram_gib())
    log.info("Budget: %.2f GiB   grid: quant=%s seq=%s rank=%s",
             budget, cfg.pilot.quant_modes, cfg.pilot.seq_lens, cfg.pilot.lora_ranks)

    tokenizer = load_tokenizer(cfg.model)

    csv_path = run_dir / "memory_pilot.csv"
    rows: List[dict] = _load_existing_rows(csv_path)
    done = {_row_key(r["profile"], r["quant"], r["seq_len"], r["rank"]) for r in rows}
    if rows:
        log.info("Resume: %d measurement(s) already recorded in %s; skipping those.",
                 len(rows), csv_path)

    for quant in cfg.pilot.quant_modes:
        # ---- tracing profile: one model per quant, sweep seq_len ----
        tracing_todo = [s for s in cfg.pilot.seq_lens
                        if _row_key("tracing", quant, s, None) not in done]
        if tracing_todo:
            log.info("[%s] loading model for tracing profile...", quant)
            model = load_model(cfg.model, cfg.quant, quant_mode=quant)
            layout = verify_base_checkpoint(cfg.model, model.config, tokenizer)
            for seq_len in cfg.pilot.seq_lens:
                if _row_key("tracing", quant, seq_len, None) in done:
                    continue
                res = measure_tracing(model, layout, seq_len, cfg.pilot.batch_size)
                _record(rows, done, csv_path, {
                    "profile": "tracing", "quant": quant, "seq_len": seq_len, "rank": None,
                    "peak_reserved_gib": res["peak_reserved_gib"],
                    "peak_allocated_gib": res["peak_allocated_gib"], "error": None,
                })
                log.info("  tracing  q=%-4s seq=%-4d -> reserved=%s GiB",
                         quant, seq_len, _fmt(res["peak_reserved_gib"]))
            _free(model)
        else:
            log.info("[%s] tracing already recorded; skipping.", quant)

        # ---- training profile: fresh model per (quant, rank, seq_len) ----
        for rank in cfg.pilot.lora_ranks:
            for seq_len in cfg.pilot.seq_lens:
                if _row_key("training", quant, seq_len, rank) in done:
                    continue
                log.info("[%s] loading model for training profile (rank=%d, seq=%d)...",
                         quant, rank, seq_len)
                model = load_model(cfg.model, cfg.quant, quant_mode=quant)
                verify_base_checkpoint(cfg.model, model.config, tokenizer)
                try:
                    res = measure_training(cfg, model, seq_len, rank, quant=quant)
                    row = {
                        "profile": "training", "quant": quant, "seq_len": seq_len, "rank": rank,
                        "peak_reserved_gib": res["peak_reserved_gib"],
                        "peak_allocated_gib": res["peak_allocated_gib"], "error": None,
                    }
                    log.info("  training q=%-4s seq=%-4d rank=%d -> reserved=%s GiB",
                             quant, seq_len, rank, _fmt(res["peak_reserved_gib"]))
                except Exception as err:  # OOM or backend issue -> record, keep going
                    row = {
                        "profile": "training", "quant": quant, "seq_len": seq_len,
                        "rank": rank, "peak_reserved_gib": None,
                        "peak_allocated_gib": None, "error": type(err).__name__,
                    }
                    log.warning("  training q=%s seq=%d rank=%d FAILED: %s",
                                quant, seq_len, rank, err)
                _record(rows, done, csv_path, row)
                _free(model)

    # ---- viability: both profiles must fit for a (quant, seq_len, rank) ----
    viable = _viable_combos(rows, budget)
    summary = {
        "device": device_name(),
        "total_vram_gib": total_vram_gib(),
        "budget_gib": budget,
        "viable_configs": viable,
        "go": len(viable) > 0,
    }
    write_csv(csv_path, rows, fieldnames=_FIELDNAMES)
    write_json(run_dir / "memory_pilot_summary.json", summary)

    log.info("-" * 60)
    if viable:
        log.info("GO: %d viable configuration(s) fit under %.2f GiB:", len(viable), budget)
        for v in viable:
            log.info("  quant=%s seq_len=%d rank=%d  (tracing=%s, training=%s GiB)",
                     v["quant"], v["seq_len"], v["rank"],
                     _fmt(v["tracing_gib"]), _fmt(v["training_gib"]))
    else:
        log.error("NO-GO: no configuration fits under %.2f GiB. "
                  "Reduce seq_len/rank, use 4-bit, or request a larger GPU.", budget)
    log.info("Results: %s", run_dir)
    return 0 if summary["go"] else 1


def _fmt(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def _viable_combos(rows: List[dict], budget: float) -> List[dict]:
    """A (quant, seq_len, rank) is viable iff its tracing AND training peaks fit."""
    tracing = {
        (r["quant"], r["seq_len"]): r["peak_reserved_gib"]
        for r in rows if r["profile"] == "tracing"
    }
    out = []
    for r in rows:
        if r["profile"] != "training":
            continue
        tr = tracing.get((r["quant"], r["seq_len"]))
        trn = r["peak_reserved_gib"]
        if tr is None or trn is None:
            continue
        if tr <= budget and trn <= budget:
            out.append({
                "quant": r["quant"], "seq_len": r["seq_len"], "rank": r["rank"],
                "tracing_gib": tr, "training_gib": trn,
            })
    return out


if __name__ == "__main__":
    raise SystemExit(main())
