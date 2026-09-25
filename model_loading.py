"""Single shared entry point for loading the base model.

Every component (tracing, training, memory pilot) must load through
`load_base_model` so that head indices, module paths and the checkpoint
revision are identical everywhere.

Owner: Balasubramanian (P0-2)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen2.5-1.5B"
# Pinned revision. Do not change without team sign-off: all Stage 1 scores
# are only comparable across runs if the weights are identical.
MODEL_REVISION = "8faed761d45a263340a0528343f099c05c9a4323"

Precision = Literal["fp16", "bf16", "8bit", "4bit"]


@dataclass(frozen=True)
class ExpectedArch:
    """Structural facts Stage 1 head indexing depends on.

    num_query_heads x head_dim must equal hidden_size, otherwise the
    1536 -> 12 x 128 reshape used for per-head capture is invalid.
    """

    num_hidden_layers: int = 28
    hidden_size: int = 1536
    num_attention_heads: int = 12   # query heads
    num_key_value_heads: int = 2    # GQA groups
    head_dim: int = 128

    def __post_init__(self) -> None:
        assert self.num_attention_heads * self.head_dim == self.hidden_size


@dataclass
class LoadConfig:
    model_id: str = MODEL_ID
    revision: str = MODEL_REVISION
    precision: Precision = "fp16"
    device_map: str | dict | None = "auto"
    attn_implementation: str = "sdpa"
    forbid_instruct: bool = True
    expected: ExpectedArch = field(default_factory=ExpectedArch)


def _quant_kwargs(precision: Precision) -> dict:
    """Build load kwargs for a precision mode.

    fp16 is the default for tracing. bf16 is NOT natively supported on
    Turing (sm_75) and is emulated, so prefer fp16 on the 2080 Ti nodes.
    4bit/8bit require bitsandbytes and compute capability >= 7.5.
    """
    if precision == "fp16":
        return {"torch_dtype": torch.float16}
    if precision == "bf16":
        return {"torch_dtype": torch.bfloat16}

    from transformers import BitsAndBytesConfig

    if precision == "8bit":
        return {
            "quantization_config": BitsAndBytesConfig(load_in_8bit=True),
        }
    if precision == "4bit":
        return {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            ),
        }
    raise ValueError(f"unknown precision: {precision!r}")


def check_device_capability(precision: Precision) -> None:
    """Fail loudly on a node that cannot run the requested precision.

    The Ada gnode* pool is heterogeneous: RTX 2080 Ti (sm_75) alongside
    GTX 1080 Ti (sm_61). bitsandbytes NF4 needs sm_75+, so a job that
    lands on a 1080 Ti will fail or silently degrade. Constrain the
    SLURM job as well -- this is the second line of defence.
    """
    if not torch.cuda.is_available():
        log.warning("no CUDA device visible; skipping capability check")
        return

    major, minor = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name()
    log.info("device=%s capability=sm_%d%d", name, major, minor)

    cap = (major, minor)
    if precision in ("4bit", "8bit") and cap < (7, 5):
        raise RuntimeError(
            f"{precision} needs compute capability >= 7.5, got sm_{major}{minor} "
            f"on {name}. Constrain the SLURM job to 2080 Ti nodes."
        )
    if precision == "bf16" and cap < (8, 0):
        log.warning(
            "bf16 is emulated below sm_80 (got sm_%d%d on %s); prefer fp16",
            major, minor, name,
        )


def load_base_model(cfg: LoadConfig | None = None):
    """Load the pinned base checkpoint. Returns (model, tokenizer)."""
    cfg = cfg or LoadConfig()
    check_device_capability(cfg.precision)

    log.info("loading %s @ %s (%s)", cfg.model_id, cfg.revision[:8], cfg.precision)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, revision=cfg.revision)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        revision=cfg.revision,
        device_map=cfg.device_map,
        attn_implementation=cfg.attn_implementation,
        **_quant_kwargs(cfg.precision),
    )
    model.eval()
    return model, tokenizer


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------

class VerificationError(AssertionError):
    """Raised when the loaded model does not match expectations."""


def _attn_module(model, layer_idx: int):
    """Resolve one layer's attention module.

    Stage 1 indexes heads through these exact paths, so if a transformers
    version bump renames anything we want a hard failure here rather than
    silently wrong scores downstream.
    """
    try:
        return model.model.layers[layer_idx].self_attn
    except (AttributeError, IndexError) as exc:
        raise VerificationError(
            f"cannot resolve model.model.layers[{layer_idx}].self_attn -- "
            "module layout changed; hooks.py and Stage 1 patching both "
            "depend on this path"
        ) from exc


def verify_architecture(model, expected: ExpectedArch) -> list[str]:
    """Check config fields against expectations. Returns list of notes."""
    notes: list[str] = []
    c = model.config

    checks = {
        "num_hidden_layers": expected.num_hidden_layers,
        "hidden_size": expected.hidden_size,
        "num_attention_heads": expected.num_attention_heads,
        "num_key_value_heads": expected.num_key_value_heads,
    }
    for attr, want in checks.items():
        got = getattr(c, attr, None)
        if got != want:
            raise VerificationError(f"config.{attr}: expected {want}, got {got}")
        notes.append(f"config.{attr} = {got}")

    head_dim = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
    if head_dim != expected.head_dim:
        raise VerificationError(
            f"head_dim: expected {expected.head_dim}, got {head_dim}"
        )
    notes.append(f"head_dim = {head_dim}")
    return notes


def verify_module_paths(model, expected: ExpectedArch) -> list[str]:
    """Assert the projection shapes Stage 1's head indexing relies on.

    q_proj output rows and o_proj input columns are both head-aligned in
    blocks of head_dim, which is what makes a per-head mask meaningful
    there. k_proj/v_proj are KV-group aligned (2 groups), so a per-head
    score cannot be applied to them -- asserted here so that assumption
    is checked rather than remembered.
    """
    notes: list[str] = []
    n_layers = expected.num_hidden_layers
    hidden = expected.hidden_size
    kv_width = expected.num_key_value_heads * expected.head_dim  # 256

    for i in (0, n_layers // 2, n_layers - 1):
        attn = _attn_module(model, i)
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            if not hasattr(attn, name):
                raise VerificationError(f"layer {i}: missing self_attn.{name}")

        if attn.q_proj.out_features != hidden:
            raise VerificationError(
                f"layer {i}: q_proj.out_features = {attn.q_proj.out_features}, "
                f"expected {hidden} (12 query heads x 128)"
            )
        if attn.o_proj.in_features != hidden:
            raise VerificationError(
                f"layer {i}: o_proj.in_features = {attn.o_proj.in_features}, "
                f"expected {hidden} -- this is the tensor per-head capture "
                "hooks reshape to [B, T, 12, 128]"
            )
        for name in ("k_proj", "v_proj"):
            proj = getattr(attn, name)
            if proj.out_features != kv_width:
                raise VerificationError(
                    f"layer {i}: {name}.out_features = {proj.out_features}, "
                    f"expected {kv_width} ({expected.num_key_value_heads} KV "
                    "groups x 128)"
                )
        notes.append(f"layer {i}: projection shapes OK")

    # The reshape Stage 1 depends on must be exact.
    if hidden % expected.num_attention_heads != 0:
        raise VerificationError("hidden_size not divisible by query-head count")
    notes.append(
        f"per-head slice width = {hidden // expected.num_attention_heads}"
    )
    return notes


def verify_revision(model, expected_revision: str) -> list[str]:
    """Best-effort check that the loaded weights are the pinned revision."""
    notes: list[str] = []
    resolved = getattr(model.config, "_commit_hash", None)
    if resolved is None:
        notes.append("WARNING: could not read resolved commit hash from config")
        return notes
    if resolved != expected_revision:
        raise VerificationError(
            f"loaded revision {resolved} != pinned {expected_revision}"
        )
    notes.append(f"revision {resolved[:12]} matches pin")
    return notes


# Substrings that indicate an instruction-tuned / chat checkpoint.
_INSTRUCT_MARKERS = ("instruct", "chat", "-it", "sft", "dpo", "rlhf")


def verify_not_instruct(model_id: str, tokenizer) -> list[str]:
    """Reject instruct checkpoints. Proposal requires the base model.

    Note: Qwen2.5 *base* ships a chat template, so template presence alone
    is a false positive (see Jawed's memory pilot report, note 4). We treat
    it as a warning and rely on the model id for the hard check.
    """
    notes: list[str] = []
    lowered = model_id.lower()
    hit = next((m for m in _INSTRUCT_MARKERS if m in lowered), None)
    if hit is not None:
        raise VerificationError(
            f"model id {model_id!r} contains {hit!r}; proposal requires the "
            "base (non-instruction-tuned) checkpoint"
        )
    notes.append(f"model id {model_id} shows no instruct markers")

    if getattr(tokenizer, "chat_template", None):
        notes.append(
            "note: tokenizer ships a chat template -- expected for Qwen2.5 "
            "base, not evidence of instruction tuning"
        )
    return notes


def verify_base_checkpoint(model, tokenizer, cfg: LoadConfig) -> list[str]:
    """Run all P0-2 checks. Raises VerificationError on the first failure."""
    notes: list[str] = []
    notes += verify_not_instruct(cfg.model_id, tokenizer)
    notes += verify_revision(model, cfg.revision)
    notes += verify_architecture(model, cfg.expected)
    notes += verify_module_paths(model, cfg.expected)
    return notes
