"""GQA-aware attention-head hooking.

Stage 1 (causal localization) needs to read and, later, patch activations at the
granularity of individual attention heads. This module provides the plumbing and
a smoke test that the hooks fire and reshape correctly **under quantization**,
which is the 🟡 de-risking item for Phase 0.

Reshape convention (Qwen2 / GQA):
    q_proj output : [B, T, num_attention_heads  * head_dim] -> [B, T, Hq,  d]
    k_proj/v_proj : [B, T, num_key_value_heads * head_dim] -> [B, T, Hkv, d]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .model_loading import HeadLayout

_PROJ_HEADS = {"q_proj": "q", "k_proj": "kv", "v_proj": "kv"}


def find_attention_modules(model) -> List[Tuple[int, object]]:
    """Return ``(layer_index, self_attn_module)`` for every decoder layer.

    Works for the standard ``model.model.layers[i].self_attn`` layout used by
    Qwen2 and most HF decoder-only models.
    """
    layers = model.model.layers
    out = []
    for idx, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
        if attn is None:
            raise AttributeError(f"Layer {idx} has no self-attention submodule")
        out.append((idx, attn))
    return out


def _heads_for(projection: str, layout: HeadLayout) -> int:
    kind = _PROJ_HEADS.get(projection)
    if kind == "q":
        return layout.num_attention_heads
    if kind == "kv":
        return layout.num_key_value_heads
    raise ValueError(f"Unsupported projection for head capture: {projection!r}")


@dataclass
class AttentionHeadCapture:
    """Register forward hooks that reshape a projection's output per head.

    Parameters
    ----------
    model : the loaded causal-LM.
    layout : :class:`HeadLayout` describing the GQA geometry.
    projection : which linear to capture (``"q_proj"`` by default, matching the
        proposal's primary configuration).
    """

    model: object
    layout: HeadLayout
    projection: str = "q_proj"
    activations: Dict[int, "object"] = field(default_factory=dict)
    _handles: List[object] = field(default_factory=list)

    def _make_hook(self, layer_idx: int):
        n_heads = _heads_for(self.projection, self.layout)
        d = self.layout.head_dim

        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            b, t, _ = tensor.shape
            # [B, T, H*d] -> [B, T, H, d]; detach so we never hold the graph.
            self.activations[layer_idx] = tensor.detach().view(b, t, n_heads, d)

        return hook

    def __enter__(self) -> "AttentionHeadCapture":
        for idx, attn in find_attention_modules(self.model):
            proj = getattr(attn, self.projection)
            self._handles.append(proj.register_forward_hook(self._make_hook(idx)))
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        return False

    def expected_shape(self, batch: int, seq_len: int) -> Tuple[int, int, int, int]:
        return (batch, seq_len, _heads_for(self.projection, self.layout), self.layout.head_dim)
