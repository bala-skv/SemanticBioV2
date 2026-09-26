"""Pre-fetch the base checkpoint into the HF cache — run on a LOGIN node.

GPU compute nodes on most HP/SLURM clusters have **no outbound network**, and
the Qwen download is large and occasionally rate-limited. Fetching the weights
+ tokenizer once on a login node (where network works) means the SLURM job can
run fully offline via ``HF_HUB_OFFLINE=1``.

It downloads into ``HF_HOME`` (set this to roomy shared/scratch storage, not
your home quota) and verifies the checkpoint is the expected *base* model.

    export HF_HOME=/path/to/shared/hf_cache
    python cluster/prefetch_model.py --config configs/phase0.yaml
"""
import argparse
import os
import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from circuit_routing.config import load_config
from circuit_routing.model_loading import (
    load_tokenizer,
    looks_instruction_tuned,
    verify_base_checkpoint,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-download the base checkpoint on a login node.")
    parser.add_argument("--config", default="configs/phase0.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    name = cfg.model.name
    revision = cfg.model.revision

    print(f"HF_HOME        : {os.environ.get('HF_HOME', '(default ~/.cache/huggingface)')}")
    print(f"Checkpoint     : {name}")
    print(f"Revision (pin) : {revision or '(unpinned — will resolve to latest main)'}")
    if not revision:
        print("  WARNING: revision is null. Pin a commit SHA in configs/phase0.yaml "
              "for reproducibility before the real runs.")

    # Tokenizer (small) — also lets us reject an instruct checkpoint early.
    print("\n[1/2] downloading tokenizer ...")
    tok = load_tokenizer(cfg.model)
    if cfg.model.forbid_instruct and getattr(tok, "chat_template", None):
        # NOT a hard failure: Qwen2.5 *base* tokenizers ship a chat template too.
        # The authoritative base/instruct decision is the name marker + structural
        # checks in verify_base_checkpoint below.
        print("  [warn] tokenizer carries a chat template (expected for Qwen2.5 base); "
              "continuing — will verify via name marker + head/layer layout.")

    # Config + weights. Use snapshot_download so we cache the full repo without
    # needing a GPU to instantiate the model.
    print("\n[2/2] downloading model weights + config ...")
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig

    local_dir = snapshot_download(
        repo_id=name,
        revision=revision,
        allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors", "*.bin"],
    )
    print(f"  cached at: {local_dir}")

    config = AutoConfig.from_pretrained(name, revision=revision,
                                        trust_remote_code=cfg.model.trust_remote_code)
    if cfg.model.forbid_instruct and looks_instruction_tuned(name, config):
        print("  ERROR: config looks instruction/chat-tuned. Aborting.")
        return 1

    layout = verify_base_checkpoint(cfg.model, config, tok)
    print("\n[OK] base checkpoint verified and cached.")
    print(f"     layers={layout.num_layers}  q_heads={layout.num_attention_heads}  "
          f"kv_groups={layout.num_key_value_heads}  head_dim={layout.head_dim}")
    print("     The SLURM job can now run offline (HF_HUB_OFFLINE=1).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
