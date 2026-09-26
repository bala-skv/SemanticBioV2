"""
patching.py  -  Stage 1 activation-patching harness

Owner:      Daniel Ashish Abraham
Task:       Stage 1  (gate: stage1_sanity_checks.py must PASS before heatmap)
Depends on: phase0/model_loading.py  (pinned fp16 loader, Bala P0-2)
            phase0/hooks.py          (verified module layout, Daniel P0-6)

Design contract  (Bala handoff 2026-09-26)
------------------------------------------
  Hook target  : o_proj INPUT  (pre-hook, not output; not q_proj)
  Reshape      : 1536 -> 12 x 128  (head-aligned; verified at layers 0/14/27)
  patch_pos    : index of the LAST PROMPT TOKEN (from score_sequence)
  Capture dtype: fp16  (NF4 noise would smear bimodal scores)

Three-run protocol
------------------
  stored           = capture_activations(model, hi_ids, patch_pos)
  logits_corrupted = patched_logits(model, en_ids, patch_pos, None, None, None)
  logits_patched   = patched_logits(model, en_ids, patch_pos, L, H, stored)
  # Recovery computed by sequence_scoring.recovery(), not here.

Architecture constants  (verified gnode074 by Bala verify_model.py)
--------------------------------------------------------------------
  28 layers, 12 q-heads, head_dim 128, hidden 1536
  o_proj.in_features == q_proj.out_features == 1536 -> same reshape licensed
"""

from __future__ import annotations
from typing import Optional
import torch

NUM_LAYERS  = 28
NUM_HEADS   = 12
HEAD_DIM    = 128
HIDDEN_SIZE = NUM_HEADS * HEAD_DIM   # 1536
_CAPTURE_DTYPE = torch.float16       # fp16 trace; change only with team sign-off


class OProjCapture:
    """Context manager: records o_proj INPUT at patch_pos (read-only pre-hook).

    Captured per layer: Tensor [12, 128] fp16 CPU
    """

    def __init__(self, model, patch_pos: int) -> None:
        self.model     = model
        self.patch_pos = patch_pos
        self.handles:    list = []
        self.activations: dict[int, torch.Tensor] = {}

    def _make_hook(self, layer_idx: int):
        pos = self.patch_pos
        def _hook(_module, args):
            x = args[0]               # [B, T, 1536]
            self.activations[layer_idx] = (
                x[0, pos]
                .reshape(NUM_HEADS, HEAD_DIM)
                .detach().to(dtype=_CAPTURE_DTYPE).cpu()
            )
        return _hook

    def __enter__(self) -> "OProjCapture":
        for idx, layer in enumerate(self.model.model.layers):
            self.handles.append(
                layer.self_attn.o_proj.register_forward_pre_hook(self._make_hook(idx))
            )
        return self

    def __exit__(self, *_) -> bool:
        for h in self.handles: h.remove()
        self.handles.clear()
        return False


class OProjPatchSingle:
    """Substitutes ONE head's o_proj input at patch_pos in one layer."""

    def __init__(self, model, patch_pos: int, layer: int, head_idx: int,
                 stored: dict[int, torch.Tensor]) -> None:
        self.model    = model
        self.patch_pos = patch_pos
        self.layer    = layer
        self.head_idx = head_idx
        self.stored   = stored
        self.handles: list = []

    def __enter__(self) -> "OProjPatchSingle":
        src   = self.stored[self.layer][self.head_idx]   # [128] fp16 CPU
        pos   = self.patch_pos
        start = self.head_idx * HEAD_DIM
        end   = start + HEAD_DIM
        target = self.model.model.layers[self.layer]

        def _hook(_module, args):
            x = args[0].clone()
            x[0, pos, start:end] = src.to(device=x.device, dtype=x.dtype)
            return (x,)

        self.handles.append(
            target.self_attn.o_proj.register_forward_pre_hook(_hook)
        )
        return self

    def __exit__(self, *_) -> bool:
        for h in self.handles: h.remove()
        self.handles.clear()
        return False


class OProjPatchLayer:
    """Patches ALL 12 heads in one layer (layer-sweep sanity check 3)."""

    def __init__(self, model, patch_pos: int, layer: int,
                 stored: dict[int, torch.Tensor]) -> None:
        self.model    = model
        self.patch_pos = patch_pos
        self.layer    = layer
        self.stored   = stored
        self.handles: list = []

    def __enter__(self) -> "OProjPatchLayer":
        src    = self.stored[self.layer].reshape(HIDDEN_SIZE)  # [1536]
        pos    = self.patch_pos
        target = self.model.model.layers[self.layer]

        def _hook(_module, args):
            x = args[0].clone()
            x[0, pos, :] = src.to(device=x.device, dtype=x.dtype)
            return (x,)

        self.handles.append(
            target.self_attn.o_proj.register_forward_pre_hook(_hook)
        )
        return self

    def __exit__(self, *_) -> bool:
        for h in self.handles: h.remove()
        self.handles.clear()
        return False


class OProjPatchAll:
    """Patches ALL 336 heads across all 28 layers (sanity check 1: recovery~1.0)."""

    def __init__(self, model, patch_pos: int,
                 stored: dict[int, torch.Tensor]) -> None:
        self.model    = model
        self.patch_pos = patch_pos
        self.stored   = stored
        self.handles: list = []

    def _make_hook(self, layer_idx: int):
        src = self.stored[layer_idx].reshape(HIDDEN_SIZE)
        pos = self.patch_pos
        def _hook(_module, args):
            x = args[0].clone()
            x[0, pos, :] = src.to(device=x.device, dtype=x.dtype)
            return (x,)
        return _hook

    def __enter__(self) -> "OProjPatchAll":
        for idx, layer in enumerate(self.model.model.layers):
            self.handles.append(
                layer.self_attn.o_proj.register_forward_pre_hook(self._make_hook(idx))
            )
        return self

    def __exit__(self, *_) -> bool:
        for h in self.handles: h.remove()
        self.handles.clear()
        return False


@torch.no_grad()
def capture_activations(
    model,
    input_ids: torch.Tensor,
    patch_pos: int,
) -> dict[int, torch.Tensor]:
    """Run one forward pass; return o_proj INPUT activations at patch_pos.

    Returns
    -------
    dict  layer_idx -> Tensor [12, 128]  fp16  CPU
    """
    with OProjCapture(model, patch_pos) as cap:
        model(input_ids=input_ids, use_cache=False)
    if len(cap.activations) != NUM_LAYERS:
        raise RuntimeError(
            f"Expected {NUM_LAYERS} layers captured, got {len(cap.activations)}."
        )
    return cap.activations


@torch.no_grad()
def patched_logits(
    model,
    input_ids: torch.Tensor,
    patch_pos: int,
    layer: Optional[int],
    head_idx: Optional[int],
    stored: Optional[dict[int, torch.Tensor]],
) -> torch.Tensor:
    """Forward pass with optional single-head o_proj patching.

    Pass layer=None / head_idx=None / stored=None for an unpatched run.

    Returns logits [B, T, vocab_size].
    """
    if layer is None or head_idx is None or stored is None:
        return model(input_ids=input_ids, use_cache=False).logits
    with OProjPatchSingle(model, patch_pos, layer, head_idx, stored):
        return model(input_ids=input_ids, use_cache=False).logits
