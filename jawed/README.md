# Phase 0 — Setup & De-risking

Runnable framework for **Phase 0** of *Interpretability-Guided Circuit Routing for
Hindi–English LoRA Fine-Tuning*. It de-risks the project **before** any Stage-1
tracing or Stage-2 training by validating the four things that most often sink it:

| Goal | Script | Answers |
|---|---|---|
| 🔴 Pin & verify the **base** checkpoint | `scripts/verify_model.py` | Is it Qwen2.5-1.5B *base* (not `-Instruct`)? Does the GQA layout match (28 layers, 12 q-heads, 2 kv-groups)? |
| 🔴 **Memory pilot** | `scripts/memory_pilot.py` | Does the 4–5 GB budget hold across `{4bit,8bit,bf16} × {256,512} × rank{4,8}`, measured **separately** for tracing vs. training? |
| 🟡 **Hook smoke test** | `scripts/hook_smoke_test.py` | Does per-head activation hooking work **under quantization**, with correct GQA shapes? |
| ⚙️ **Reproducibility** | `circuit_routing/seeding.py` | Fixed seeds (≥3), deterministic kernels, config snapshot per run. |

Run everything at once:

```bash
python scripts/run_phase0.py --config configs/phase0.yaml
```

## Environment

This code targets a **single small GPU on the cluster** (CUDA + bitsandbytes).
It is not expected to run on the locked-down workstation (no local torch). Install
on the cluster:

```bash
pip install -e .          # or: pip install -r requirements.txt
```

## Layout

```
circuit_routing/
  configs/phase0.yaml          # single source of truth for the pilot grid
  circuit_routing/
    config.py                  # dataclasses + YAML loader
    seeding.py                 # set_seed + determinism
    logging_utils.py           # run dirs, logger, JSON/CSV result writers
    model_loading.py           # load + verify base checkpoint, quantization
    memory.py                  # peak-VRAM measurement utilities
    hooks.py                   # attention-head hooking (GQA-aware)
  scripts/
    verify_model.py            # checkpoint identity + structural report
    memory_pilot.py            # the go/no-go memory sweep
    hook_smoke_test.py         # hooking under quantization
    run_phase0.py              # orchestrates all checks + go/no-go summary
```

## Go / No-go

`run_phase0.py` exits non-zero if any gate fails, so it can gate CI or a cluster
job before Stage 1 begins:

- **base checkpoint** verified (not instruct),
- **at least one** pilot configuration fits under `pilot.vram_budget_gb` for
  both tracing and training,
- **hooks** capture per-head activations with the expected `[B, T, H, d]` shapes.
