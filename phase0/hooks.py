"""
hooks.py  —  Per-head hooking under quantization (GQA-correct shapes)

Task:       P0-6
Owner:      Daniel Ashish Abraham
Depends on: P0-1 (env), P0-2 (verified checkpoint + GQA layout)
Enables:    P0-7 (memory pilot), P0-9 (orchestration)

Definition of Done
------------------
Hook smoke test passes across 4-bit / 8-bit / bf16 with shape [B, T, H, d].
AttentionHeadCapture is the exact mechanism Stage 1 patches through.

Design notes
------------
Qwen2.5-1.5B is a Grouped-Query Attention (GQA) model:
  - 12 query heads  (num_attention_heads)
  -  2 key/value heads (num_key_value_heads)
  - head_dim = 128

Under 4-bit / 8-bit bitsandbytes quantization the q_proj weight is a
bnb.nn.Linear4bit / Linear8bitLt object, but the forward output tensor is
a plain torch.Tensor in the compute dtype (bfloat16 or float16).  The hook
therefore reads output shapes only and never touches the quantized weight
directly.

The captured shape for each layer is [B, T, num_attention_heads, head_dim].
Key/value heads are NOT captured here; they live in k_proj / v_proj and are
outside the scope of P0-6.  Stage 1 patching targets this same hook point.
"""

from __future__ import annotations

from typing import Any

import torch


# ---------------------------------------------------------------------------
# Core hook class
# ---------------------------------------------------------------------------

class AttentionHeadCapture:
    """Context manager that registers forward hooks on every layer's q_proj.

    Works identically under bf16 / 8-bit / 4-bit quantisation because it
    only inspects the output tensor, not the weight storage.

    Parameters
    ----------
    model :
        A loaded Qwen2ForCausalLM (or any model whose attention layers are
        accessible via ``model.model.layers[i].self_attn.q_proj``).
    num_heads : int
        Number of query heads (``num_attention_heads`` from the config).
    head_dim : int
        Dimension of each query head (``hidden_size // num_attention_heads``).
    capture_tensors : bool
        If True, store the full [B, T, H, d] float32 CPU tensor for each
        layer in ``self.activations``.  Use for offline analysis.  Leave
        False for lightweight smoke tests and tracing memory checks.
    """

    def __init__(
        self,
        model,
        num_heads: int,
        head_dim: int,
        capture_tensors: bool = False,
    ) -> None:
        self.model = model
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.capture_tensors = capture_tensors
        self.handles: list = []
        # layer_index -> list[int]  (the four-element shape)
        self.shapes: dict[int, list[int]] = {}
        # layer_index -> Tensor [B, T, H, d] float32 CPU  (only if capture_tensors)
        self.activations: dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _make_hook(self, layer_index: int):
        """Return a forward hook closure for the given layer index."""

        def _hook(_module, _inputs, output):
            if not isinstance(output, torch.Tensor):
                raise TypeError(
                    f"Layer {layer_index} q_proj returned {type(output).__name__} "
                    f"instead of Tensor — check model architecture."
                )
            if output.ndim != 3:
                raise AssertionError(
                    f"Layer {layer_index} q_proj output must be 3-D [B, T, H*d], "
                    f"got shape {tuple(output.shape)}."
                )

            batch, tokens, width = output.shape
            expected_width = self.num_heads * self.head_dim
            if width != expected_width:
                raise AssertionError(
                    f"Layer {layer_index}: expected q_proj output width "
                    f"{expected_width} (= {self.num_heads} heads x {self.head_dim} dim), "
                    f"got {width}.  Verify num_attention_heads and head_dim in config."
                )

            # GQA-correct reshape: split the head dimension
            heads = output.reshape(batch, tokens, self.num_heads, self.head_dim)
            self.shapes[layer_index] = list(heads.shape)

            if self.capture_tensors:
                # Always move to CPU + cast to float32 so downstream analysis
                # code never has to deal with quantised or half-precision tensors.
                self.activations[layer_index] = heads.detach().float().cpu()

        return _hook

    # ------------------------------------------------------------------
    # Context-manager interface
    # ------------------------------------------------------------------

    def __enter__(self) -> "AttentionHeadCapture":
        layers = self.model.model.layers
        for idx, layer in enumerate(layers):
            handle = layer.self_attn.q_proj.register_forward_hook(
                self._make_hook(idx)
            )
            self.handles.append(handle)
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        # Return False so any exception propagates normally.
        return False


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def random_input_ids(
    vocab_size: int,
    batch: int,
    seq_len: int,
    device: str = "cpu",
) -> torch.Tensor:
    """Return a random integer token tensor for smoke-test forward passes."""
    return torch.randint(
        0, int(vocab_size), (batch, seq_len), device=device, dtype=torch.long
    )


# ---------------------------------------------------------------------------
# Smoke-test runner  (used by hook_smoke_test.py and run_phase0.py)
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_hook_smoke(
    model,
    expected: dict[str, Any],
    seq_len: int = 32,
) -> dict[str, Any]:
    """Run a lightweight forward pass and verify hook shapes on all layers.

    Parameters
    ----------
    model :
        Loaded model (any precision).
    expected : dict
        Must contain ``num_hidden_layers``, ``num_attention_heads``,
        ``head_dim`` (taken from the verified checkpoint config).
    seq_len : int
        Token sequence length to use for the test input.

    Returns
    -------
    dict with keys:
        ``captured_layers``, ``expected_layers``, ``shape_each``,
        ``target``, ``handles_removed``.

    Raises
    ------
    AssertionError if any layer is missing or has the wrong shape.
    """
    num_layers = int(expected["num_hidden_layers"])
    num_heads  = int(expected["num_attention_heads"])
    head_dim   = int(expected["head_dim"])

        # Infer device from the model so this works on both CPU and GPU.
    _device = next(model.parameters()).device
    ids = random_input_ids(model.config.vocab_size, 1, seq_len, device=str(_device))

    with AttentionHeadCapture(model, num_heads, head_dim, capture_tensors=False) as cap:
        model(input_ids=ids, use_cache=False)

    expected_shape = [1, seq_len, num_heads, head_dim]

    # Every layer 0 ... num_layers-1 must be present.
    if sorted(cap.shapes) != list(range(num_layers)):
        raise AssertionError(
            f"Expected hooks on layers 0..{num_layers - 1}, "
            f"but captured layers: {sorted(cap.shapes)}"
        )

    # Every captured layer must have the correct [B, T, H, d] shape.
    bad = {
        layer: shape
        for layer, shape in cap.shapes.items()
        if shape != expected_shape
    }
    if bad:
        raise AssertionError(f"Incorrect hook shapes: {bad}")

    return {
        "captured_layers":  len(cap.shapes),
        "expected_layers":  num_layers,
        "shape_each":       expected_shape,
        "target":           "q_proj query-head blocks",
        "handles_removed":  len(cap.handles) == 0,
    }


@torch.no_grad()
def run_tracing_memory_check(
    model,
    expected: dict[str, Any],
    seq_len: int,
) -> dict[str, Any]:
    """Run a hooked forward pass with tensor capture and measure peak VRAM.

    This is the 'tracing' profile from the memory pilot:  forward pass with
    all 28 q_proj outputs captured to CPU float32.

    Returns
    -------
    dict with VRAM stats and capture size information.
    """
        _has_cuda = torch.cuda.is_available()
    if _has_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    _device = next(model.parameters()).device
    ids = random_input_ids(model.config.vocab_size, 1, seq_len, device=str(_device))

    with AttentionHeadCapture(
        model,
        int(expected["num_attention_heads"]),
        int(expected["head_dim"]),
        capture_tensors=True,
    ) as cap:
        model(input_ids=ids, use_cache=False)

        if _has_cuda:
        torch.cuda.synchronize()
        peak_allocated_gib = torch.cuda.max_memory_allocated() / 1024 ** 3
        peak_reserved_gib  = torch.cuda.max_memory_reserved()  / 1024 ** 3
    else:
        peak_allocated_gib = 0.0
        peak_reserved_gib  = 0.0

    total_cpu_bytes = sum(
        t.numel() * t.element_size() for t in cap.activations.values()
    )

    return {
        "seq_len":              seq_len,
        "captured_layers":      len(cap.activations),
        "capture_cpu_mib":      total_cpu_bytes / 1024 ** 2,
        "peak_allocated_gib":   peak_allocated_gib,
        "peak_reserved_gib":    peak_reserved_gib,
    }
