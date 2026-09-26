#!/usr/bin/env python3
"""
pilot_trace.py - Stage 1 Pilot Trace for Heatmap Generation

Iterates over 28 layers x 12 heads to generate recovery heatmaps for three conditions:
1. en <-> hi_latn
2. hi_deva <-> hi_latn
3. en <-> hi_deva

Since tracing_set_v1.json is not yet available, this uses a hardcoded test pair.
"""

from __future__ import annotations
import sys
import json
import torch
import torch.nn.functional as F
from pathlib import Path

_STAGE1_DIR = Path(__file__).resolve().parent
_REPO_ROOT  = _STAGE1_DIR.parent
for _p in (_STAGE1_DIR, _REPO_ROOT, _REPO_ROOT / "phase0"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from patching import capture_activations, patched_logits, NUM_LAYERS, NUM_HEADS
from model_loading import load_base_model, LoadConfig

def _score(model, input_ids: torch.Tensor, prompt_len: int) -> float:
    with torch.no_grad():
        logits = model(input_ids=input_ids, use_cache=False).logits
    T = input_ids.shape[1]
    lp = F.log_softmax(logits[0, prompt_len - 1 : T - 1], dim=-1)
    tgt = input_ids[0, prompt_len:]
    return lp[torch.arange(T - prompt_len), tgt].mean().item()

def _score_from_logits(logits: torch.Tensor, input_ids: torch.Tensor, prompt_len: int) -> float:
    T = input_ids.shape[1]
    lp = F.log_softmax(logits[0, prompt_len - 1 : T - 1], dim=-1)
    tgt = input_ids[0, prompt_len:]
    return lp[torch.arange(T - prompt_len), tgt.to(logits.device)].mean().item()

def _contrast(logp_native: float, logp_foreign: float) -> float:
    return logp_native - logp_foreign

def _recovery(m_clean: float, m_corrupted: float, m_patched: float) -> float:
    denom = m_clean - m_corrupted
    if abs(denom) < 1e-6:
        return 0.0
    return (m_patched - m_corrupted) / denom

def _make_ids(tok, prompt: str, target: str, device):
    p = tok(prompt, return_tensors="pt").input_ids.to(device)
    t = tok(target, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    return torch.cat([p, t], dim=1), p.shape[1]

CONDITIONS = {
    "en_hi-latn": {
        "src_prompt": "Aasman ka rang hai ", "src_target": "neela",
        "tgt_prompt": "The colour of the sky is ", "tgt_target": "blue"
    },
    "hi-deva_hi-latn": {
        "src_prompt": "Aasman ka rang hai ", "src_target": "neela",
        "tgt_prompt": "आसमान का रंग है ", "tgt_target": "नीला"
    },
    "en_hi-deva": {
        "src_prompt": "आसमान का रंग है ", "src_target": "नीला",
        "tgt_prompt": "The colour of the sky is ", "tgt_target": "blue"
    }
}

def run_trace(model, tok, device, cond_name: str, cond_data: dict):
    print(f"\n--- Running trace for {cond_name} ---")
    src_ids, src_plen = _make_ids(tok, cond_data["src_prompt"], cond_data["src_target"], device)
    tgt_ids, tgt_plen = _make_ids(tok, cond_data["tgt_prompt"], cond_data["tgt_target"], device)
    
    src_patch_pos = src_plen - 1
    tgt_patch_pos = tgt_plen - 1

    m_clean = _contrast(
        _score(model, src_ids, src_plen),
        _score(model, *_make_ids(tok, cond_data["src_prompt"], cond_data["tgt_target"], device)),
    )
    en_ids_n, en_plen_n = _make_ids(tok, cond_data["tgt_prompt"], cond_data["src_target"], device)
    en_ids_f, en_plen_f = _make_ids(tok, cond_data["tgt_prompt"], cond_data["tgt_target"], device)
    m_corrupted = _contrast(
        _score(model, en_ids_n, en_plen_n),
        _score(model, en_ids_f, en_plen_f),
    )

    print(f"  m_clean={m_clean:.4f}, m_corrupted={m_corrupted:.4f}")

    stored = capture_activations(model, src_ids, src_patch_pos)
    
    heatmap = []
    for L in range(NUM_LAYERS):
        layer_rec = []
        for H in range(NUM_HEADS):
            lg_n = patched_logits(model, en_ids_n, tgt_patch_pos, L, H, stored)
            lg_f = patched_logits(model, en_ids_f, tgt_patch_pos, L, H, stored)
            
            m_p = _contrast(
                _score_from_logits(lg_n, en_ids_n, en_plen_n),
                _score_from_logits(lg_f, en_ids_f, en_plen_f),
            )
            rec = _recovery(m_clean, m_corrupted, m_p)
            layer_rec.append(rec)
        heatmap.append(layer_rec)
        print(f"  L={L:02d} Max recovery in layer: {max(layer_rec):.4f}")
    
    return heatmap

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading model (fp16 trace precision) ...")
    cfg = LoadConfig(precision="fp16")
    model, tok = load_base_model(cfg)
    model.eval()
    
    results = {}
    for name, data in CONDITIONS.items():
        results[name] = run_trace(model, tok, device, name, data)
        
    out_file = Path("runs/pilot_trace_results.json")
    out_file.parent.mkdir(exist_ok=True)
    out_file.write_text(json.dumps(results, indent=2))
    print(f"\nTrace complete! Heatmaps saved to {out_file}")

if __name__ == "__main__":
    main()