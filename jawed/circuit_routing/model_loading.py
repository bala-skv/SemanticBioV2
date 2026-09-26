"""Model & tokenizer loading with checkpoint verification and quantization.

Central responsibilities:

* Load ``Qwen2.5-1.5B`` (base) plus tokenizer, optionally 4-/8-bit quantized.
* **Verify** the checkpoint is the base model, not an instruction-tuned one, and
  that its GQA layout matches what the routing method assumes.
* Expose the head-layout facts (q-heads, kv-groups, head_dim) that the hooks and
  the Circuit-Routing Adapter both depend on.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .config import ModelConfig, QuantConfig

# Substrings that indicate an instruction/chat checkpoint we must reject.
_INSTRUCT_MARKERS = ("instruct", "chat", "-it", "sft", "rlhf", "dpo")


class CheckpointVerificationError(RuntimeError):
    """Raised when the loaded checkpoint is not the expected base model."""


@dataclass
class HeadLayout:
    """Attention head geometry used by hooks and the routing adapter."""

    num_layers: int
    num_attention_heads: int      # query heads
    num_key_value_heads: int      # kv groups (GQA)
    head_dim: int
    hidden_size: int

    @property
    def q_dim(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def group_size(self) -> int:
        """Query heads per kv group."""
        return self.num_attention_heads // self.num_key_value_heads


def _torch_dtype(name: str):
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def looks_instruction_tuned(name_or_path: str, config) -> bool:
    """Heuristic: flag instruction/chat checkpoints by NAME marker.

    Note: a baked-in ``chat_template`` is *not* used as evidence here — Qwen2.5
    *base* tokenizers ship a chat template too, so it produces false positives.
    Base-vs-instruct is decided by the name marker plus the structural head/layer
    checks in :func:`verify_base_checkpoint`.
    """
    lowered = str(name_or_path).lower()
    return any(marker in lowered for marker in _INSTRUCT_MARKERS)


def has_chat_template(obj) -> bool:
    """Whether a tokenizer/config carries a chat template (informational only)."""
    return bool(getattr(obj, "chat_template", None))


def head_layout_from_config(config) -> HeadLayout:
    n_heads = config.num_attention_heads
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // n_heads
    return HeadLayout(
        num_layers=config.num_hidden_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=getattr(config, "num_key_value_heads", n_heads),
        head_dim=head_dim,
        hidden_size=config.hidden_size,
    )


def verify_base_checkpoint(model_cfg: ModelConfig, config, tokenizer=None) -> HeadLayout:
    """Assert the checkpoint is the expected base model and return its layout.

    Raises :class:`CheckpointVerificationError` on any mismatch so a pilot can
    hard-fail before wasting GPU time.
    """
    name = getattr(config, "_name_or_path", model_cfg.name)

    if model_cfg.forbid_instruct:
        if looks_instruction_tuned(name, config):
            raise CheckpointVerificationError(
                f"Checkpoint '{name}' appears instruction/chat-tuned "
                "(name marker present); the proposal requires the base checkpoint."
            )
        # A chat template alone is NOT authoritative (Qwen2.5 base has one too),
        # so it is a warning, not a hard failure.
        if has_chat_template(config) or (tokenizer is not None and has_chat_template(tokenizer)):
            print(
                f"  [warn] '{name}' carries a chat template but its name is not an "
                "instruct marker; treating as base (expected for Qwen2.5 base)."
            )

    layout = head_layout_from_config(config)

    checks = [
        ("num_layers", model_cfg.expect_num_layers, layout.num_layers),
        ("num_attention_heads", model_cfg.expect_num_attention_heads, layout.num_attention_heads),
        ("num_key_value_heads", model_cfg.expect_num_key_value_heads, layout.num_key_value_heads),
    ]
    mismatches = [
        f"{field}: expected {exp}, got {got}"
        for field, exp, got in checks
        if exp is not None and exp != got
    ]
    if mismatches:
        raise CheckpointVerificationError(
            "Structural mismatch with the expected base model: " + "; ".join(mismatches)
        )

    return layout


def build_quantization_config(mode: str, quant_cfg: QuantConfig):
    """Return a ``BitsAndBytesConfig`` for ``4bit``/``8bit``, or ``None``."""
    if mode in ("bf16", "none", None):
        return None

    from transformers import BitsAndBytesConfig

    if mode == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    if mode == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_cfg.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=_torch_dtype(quant_cfg.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=quant_cfg.bnb_4bit_use_double_quant,
        )
    raise ValueError(f"Unknown quantization mode: {mode!r}")


def load_tokenizer(model_cfg: ModelConfig):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        model_cfg.name,
        revision=model_cfg.revision,
        trust_remote_code=model_cfg.trust_remote_code,
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(
    model_cfg: ModelConfig,
    quant_cfg: QuantConfig,
    quant_mode: str = "bf16",
    device_map: Optional[str] = "auto",
):
    """Load the causal-LM, quantized per ``quant_mode``."""
    import torch
    from transformers import AutoModelForCausalLM

    bnb = build_quantization_config(quant_mode, quant_cfg)
    kwargs = dict(
        revision=model_cfg.revision,
        trust_remote_code=model_cfg.trust_remote_code,
        device_map=device_map,
    )
    if bnb is not None:
        kwargs["quantization_config"] = bnb
    else:
        kwargs["torch_dtype"] = _torch_dtype(model_cfg.dtype)

    model = AutoModelForCausalLM.from_pretrained(model_cfg.name, **kwargs)
    model.config.use_cache = False  # incompatible with grad-checkpointing / patching
    return model
