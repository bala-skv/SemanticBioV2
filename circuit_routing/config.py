"""Typed configuration for Phase 0, loaded from ``configs/phase0.yaml``.

The dataclasses below mirror the YAML structure exactly. ``load_config`` performs
a shallow, section-wise merge of the file over the dataclass defaults so a config
may specify only the fields it wants to override.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen2.5-1.5B"
    revision: Optional[str] = None
    dtype: str = "bfloat16"
    trust_remote_code: bool = False
    forbid_instruct: bool = True
    expect_num_layers: Optional[int] = 28
    expect_num_attention_heads: Optional[int] = 12
    expect_num_key_value_heads: Optional[int] = 2


@dataclass
class ReproConfig:
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    deterministic: bool = True


@dataclass
class PilotConfig:
    quant_modes: List[str] = field(default_factory=lambda: ["4bit", "8bit", "bf16"])
    seq_lens: List[int] = field(default_factory=lambda: [256, 512])
    lora_ranks: List[int] = field(default_factory=lambda: [4, 8])
    batch_size: int = 1
    grad_accum_steps: int = 8
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj"])
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    vram_budget_gb: float = 5.0


@dataclass
class QuantConfig:
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class PathsConfig:
    output_dir: str = "runs"


@dataclass
class Phase0Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    repro: ReproConfig = field(default_factory=ReproConfig)
    pilot: PilotConfig = field(default_factory=PilotConfig)
    quant: QuantConfig = field(default_factory=QuantConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


# --------------------------------------------------------------------------- #
# Loading / merging
# --------------------------------------------------------------------------- #
_SECTIONS = {
    "model": ModelConfig,
    "repro": ReproConfig,
    "pilot": PilotConfig,
    "quant": QuantConfig,
    "paths": PathsConfig,
}


def _build_section(cls, overrides: Optional[dict]):
    if not overrides:
        return cls()
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(overrides) - known
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**overrides)


def load_config(path: Optional[str | Path] = None) -> Phase0Config:
    """Load a :class:`Phase0Config`, overlaying ``path`` (YAML) on the defaults."""
    if path is None:
        return Phase0Config()

    import yaml  # local import so the dataclasses are usable without pyyaml

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    unknown = set(raw) - set(_SECTIONS)
    if unknown:
        raise ValueError(f"Unknown top-level config sections: {sorted(unknown)}")

    return Phase0Config(
        **{name: _build_section(cls, raw.get(name)) for name, cls in _SECTIONS.items()}
    )


def to_dict(cfg: Phase0Config) -> dict:
    """Plain-dict view of the config (for JSON snapshots)."""
    return dataclasses.asdict(cfg)
