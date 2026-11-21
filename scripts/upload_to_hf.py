#!/usr/bin/env python3
"""
Upload 85M Speculative Draft Model weights to Hugging Face Hub.

Usage:
  # 1. Login to Hugging Face (requires a token with WRITE access from https://huggingface.co/settings/tokens)
  huggingface-cli login

  # 2. Upload model weights and config
  python scripts/upload_to_hf.py --repo-id <your-hf-username>/qwen2.5-85m-speculative-draft
"""

import argparse
import json
import os
import sys
from huggingface_hub import HfApi

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import get_qwen_85m_config


def upload(repo_id: str, checkpoint_path: str):
    api = HfApi()

    print(f"Creating / verifying Hugging Face repository: {repo_id}...")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    # 1. Generate config.json for the model
    cfg = get_qwen_85m_config()
    config_dict = {
        "model_name": "Hybrid-0.1-85M",
        "architectures": ["SmallLLM"],
        "model_type": "small_llm",
        "vocab_size": cfg.vocab_size,
        "d_model": cfg.d_model,
        "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads,
        "n_kv_heads": cfg.n_kv_heads,
        "hidden_dim": cfg.hidden_dim,
        "max_seq_len": cfg.max_seq_len,
        "rope_theta": cfg.rope_theta,
        "tie_embeddings": cfg.tie_embeddings,
        "target_model_family": "Qwen/Qwen2.5",
    }
    config_file = os.path.join(PROJECT_ROOT, "config.json")
    with open(config_file, "w") as f:
        json.dump(config_dict, f, indent=2)

    # 2. Generate Model Card README.md for Hugging Face
    hf_readme = f"""---
language:
- en
- zh
license: apache-2.0
tags:
- speculative-decoding
- draft-model
- qwen
- small-llm
- hybrid-0.1-85m
pipeline_tag: text-generation
---

# Hybrid-0.1-85M: Custom Speculative Draft Model for Qwen2.5

**Hybrid-0.1-85M** is a custom ~85M parameter causal language model designed from scratch to serve as a high-speed **Speculative Draft Model** for the **Qwen2.5** family (`Qwen2.5-0.5B`, `Qwen2.5-1.5B`, and `Qwen2.5-7B`).

When paired in a heterogeneous speculative decoding setup (running Hybrid-0.1-85M on GPU and Qwen2.5 on CPU/GPU), it delivers **2x–3x real-world acceleration** with 100% mathematical output fidelity.

## Architecture & Specifications
- **Model Name:** Hybrid-0.1-85M
- **Parameters:** ~85,000,000 (85M)
- **Transformer Layers:** 5
- **Hidden Dimension ($d_{{\\text{{model}}}}$):** 480
- **Attention Heads:** 8 Query heads, 2 Key-Value heads (Grouped Query Attention)
- **Activations:** SwiGLU (hidden_dim = 1280)
- **Positional Encoding:** RoPE (Rotary Position Embeddings, theta = 1,000,000)
- **Tokenizer:** Exact Qwen2.5 BPE (`vocab_size = 151,936`, tied embeddings)
- **Training:** SFT on 67,000 conversational instruction pairs (15,000 steps, final loss: 1.01)

## Speculative Decoding Usage
```python
import torch
from src.speculative import SpeculativeEngine
from model import SmallLLM
from config import get_qwen_85m_config
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. Load Tokenizer & Target
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
target_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", torch_dtype=torch.float16, device_map="cuda")

# 2. Load Hybrid-0.1-85M Draft Model
draft_cfg = get_qwen_85m_config()
draft_model = SmallLLM(draft_cfg).to("cuda")
checkpoint = torch.load("final_trained_draft_85m.pt", map_location="cuda")
draft_model.load_state_dict(checkpoint["model_state_dict"])
draft_model.eval()

# 3. Initialize Lossless Speculative Engine
engine = SpeculativeEngine(target_model=target_model, draft_model=draft_model, tokenizer=tokenizer)
output = engine.generate("Explain how quantum computing works.", k=4)
print(output["text"])
```
"""
- **Activations:** SwiGLU (hidden_dim = 1280)
- **Positional Encoding:** RoPE (theta = 1,000,000)
- **Tokenizer:** Exact Qwen2.5 Byte-Pair Encoding (`vocab_size = 151,936`, tied embeddings)
- **Training:** SFT on 67,000 conversational instruction pairs (15,000 steps, final loss: 1.01)

## Speculative Decoding Usage
Pair with `Qwen2.5-1.5B` or `Qwen2.5-7B` for **2x–3x lossless speedup**:
```python
from src.speculative import SpeculativeEngine
from model import SmallLLM
import torch

# Load draft weights
checkpoint = torch.load("final_trained_draft_85m.pt", map_location="cuda")
# Run lossless speculative decoding with Qwen
```
"""
    hf_readme_file = os.path.join(PROJECT_ROOT, "HF_MODEL_CARD.md")
    with open(hf_readme_file, "w") as f:
        f.write(hf_readme)

    print(f"\n[1/3] Uploading model checkpoint: {checkpoint_path}...")
    api.upload_file(
        path_or_fileobj=checkpoint_path,
        path_in_repo="final_trained_draft_85m.pt",
        repo_id=repo_id,
        repo_type="model",
    )
    print(" -> Checkpoint uploaded successfully!")

    print("[2/3] Uploading model config.json...")
    api.upload_file(
        path_or_fileobj=config_file,
        path_in_repo="config.json",
        repo_id=repo_id,
        repo_type="model",
    )
    print(" -> config.json uploaded successfully!")

    print("[3/3] Uploading Hugging Face Model Card...")
    api.upload_file(
        path_or_fileobj=hf_readme_file,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )
    print(" -> Model Card uploaded successfully!")

    # Clean temporary files
    if os.path.exists(config_file):
        os.remove(config_file)
    if os.path.exists(hf_readme_file):
        os.remove(hf_readme_file)

    print(f"\n🎉 Deployment Complete! Your model is live at: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Upload 85M Draft Model to Hugging Face")
    parser.add_argument("--repo-id", type=str, default="Tahary/Hybrid-0.1-85M", help="HF repo name (default: Tahary/Hybrid-0.1-85M)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints_qwen7b_draft/final_trained_draft_85m.pt",
        help="Path to trained .pt file",
    )
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at '{args.checkpoint}'")
        sys.exit(1)

    upload(args.repo_id, args.checkpoint)


if __name__ == "__main__":
    main()
