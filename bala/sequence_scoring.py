"""Sequence-level scoring for Stage 1.

Why not first-token logits
--------------------------
The v0 validator scored the first token of each target. On Devanagari that
does not work: Qwen's BPE splits Hindi words into byte fragments, and seven
distinct targets came back sharing first-token id 14925. A first-token
metric therefore cannot tell नीला from लाल -- it only detects "some
Devanagari byte follows", which is true of nearly any Hindi prompt.

What we do instead
------------------
Teacher-forced log-probability of the WHOLE target, read from a single
forward pass over prompt+target:

    logp(target | prompt) = sum_t log P(target_t | prompt, target_<t)

The patching metric is the difference between conditions:

    m = logp(hi_target | hi_prompt) - logp(en_target | en_prompt)

and Recovery = (m_patched - m_corrupted) / (m_clean - m_corrupted) as in
proposal 3.1.

Length asymmetry
----------------
Hindi targets average ~4.7 tokens, English ~1.0. Summed log-prob therefore
penalises Hindi by length. `reduction="mean"` divides by token count.
Pilot both and freeze one before the production run.

Owner: Balasubramanian
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

Reduction = Literal["sum", "mean"]


@dataclass
class ScoredSequence:
    logp: float                 # after reduction
    logp_sum: float
    n_target_tokens: int
    prompt_len: int             # tokens in prompt alone
    patch_position: int         # index of the last prompt token
    per_token: list[float]


@torch.no_grad()
def score_sequence(
    model,
    tokenizer,
    prompt: str,
    target: str,
    reduction: Reduction = "mean",
) -> ScoredSequence:
    """Teacher-forced log-prob of `target` continuing `prompt`.

    `patch_position` is the index of the final PROMPT token, which is where
    Stage 1 substitutes head activations. The metric is then read over the
    target positions that follow.
    """
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    full_ids = tokenizer.encode(prompt + target, add_special_tokens=False)

    n_prompt = len(prompt_ids)
    if full_ids[:n_prompt] != prompt_ids:
        # BPE merged across the prompt/target boundary. Fall back to
        # locating the divergence point so target positions stay correct.
        n_prompt = _common_prefix_len(prompt_ids, full_ids)

    n_target = len(full_ids) - n_prompt
    if n_target <= 0:
        raise ValueError(f"target {target!r} adds no tokens to prompt")

    ids = torch.tensor([full_ids], device=model.device)
    logits = model(ids).logits[0].float()
    logprobs = logits.log_softmax(-1)

    # Position i predicts token i+1.
    per_token = [
        logprobs[n_prompt + k - 1, full_ids[n_prompt + k]].item()
        for k in range(n_target)
    ]

    total = float(sum(per_token))
    reduced = total / n_target if reduction == "mean" else total

    return ScoredSequence(
        logp=reduced,
        logp_sum=total,
        n_target_tokens=n_target,
        prompt_len=n_prompt,
        patch_position=n_prompt - 1,
        per_token=per_token,
    )


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


@torch.no_grad()
def contrast_metric(
    model,
    tokenizer,
    hi_prompt: str,
    hi_target: str,
    en_prompt: str,
    en_target: str,
    reduction: Reduction = "mean",
) -> dict:
    """m = logp(hi) - logp(en). Positive means the run favours Hindi.

    Used three ways in Stage 1:
        m_clean     : both scored on the Hindi run
        m_corrupted : both scored on the English run
        m_patched   : English run with one head's Hindi activation
    """
    hi = score_sequence(model, tokenizer, hi_prompt, hi_target, reduction)
    en = score_sequence(model, tokenizer, en_prompt, en_target, reduction)
    return {
        "m": hi.logp - en.logp,
        "hi_logp": hi.logp,
        "en_logp": en.logp,
        "hi_n_tokens": hi.n_target_tokens,
        "en_n_tokens": en.n_target_tokens,
        "hi_patch_position": hi.patch_position,
        "en_patch_position": en.patch_position,
    }


def recovery(m_patched: float, m_clean: float, m_corrupted: float) -> float:
    """Normalized recovery score (proposal 3.1).

    0 = patching changed nothing, 1 = patching fully restored the Hindi
    behaviour. Values outside [0,1] are possible and are not errors --
    report them rather than clipping silently.
    """
    denom = m_clean - m_corrupted
    if abs(denom) < 1e-8:
        raise ValueError(
            "clean and corrupted runs are indistinguishable; this item "
            "provides no contrast and should be dropped"
        )
    return (m_patched - m_corrupted) / denom
