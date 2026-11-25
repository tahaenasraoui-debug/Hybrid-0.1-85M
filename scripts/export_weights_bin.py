"""
Export Hybrid-0.1-85M checkpoint to binary format for the C++ inference engine.
"""

import argparse
import os
import struct
import sys
from typing import Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from config import ModelConfig, get_qwen_85m_config


def export_checkpoint_to_bin(
    checkpoint_path: str,
    output_bin_path: str,
    config: Optional[ModelConfig] = None,
):
    print(f"Loading checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    if config is not None:
        cfg = config
    elif isinstance(ckpt, dict) and isinstance(ckpt.get("config"), ModelConfig):
        cfg = ckpt["config"]
    else:
        cfg = get_qwen_85m_config()

    os.makedirs(os.path.dirname(os.path.abspath(output_bin_path)), exist_ok=True)

    with open(output_bin_path, "wb") as f:
        # Header (12 uint32/float32 values = 48 bytes)
        # Magic: 'HYB1' = 0x31425948
        magic = 0x31425948
        header = struct.pack(
            "<IIIIIIIIffII",
            magic,
            cfg.vocab_size,     # 151936
            cfg.d_model,        # 480
            cfg.n_layers,       # 5
            cfg.n_heads,        # 8
            cfg.n_kv_heads,     # 2
            cfg.head_dim,       # 60
            cfg.hidden_dim,     # 1280
            cfg.rope_theta,     # 1000000.0
            cfg.norm_eps,       # 1e-6
            cfg.max_seq_len,    # 2048
            1 if cfg.tie_embeddings else 0,
        )
        f.write(header)

        # 1. tok_embeddings (vocab_size, d_model)
        tok_emb = state_dict["tok_embeddings.weight"].float().contiguous()
        f.write(tok_emb.numpy().tobytes())
        print(f"Wrote tok_embeddings: {tok_emb.shape} ({tok_emb.numel() * 4 / (1024*1024):.2f} MB)")

        # 2. Per-layer weights
        for layer_idx in range(cfg.n_layers):
            pfx = f"layers.{layer_idx}."
            attn_norm = state_dict[f"{pfx}attn_norm.weight"].float().contiguous()
            q_proj = state_dict[f"{pfx}attn.q_proj.weight"].float().contiguous()
            k_proj = state_dict[f"{pfx}attn.k_proj.weight"].float().contiguous()
            v_proj = state_dict[f"{pfx}attn.v_proj.weight"].float().contiguous()
            o_proj = state_dict[f"{pfx}attn.o_proj.weight"].float().contiguous()

            ffn_norm = state_dict[f"{pfx}ffn_norm.weight"].float().contiguous()
            w1 = state_dict[f"{pfx}ffn.w1.weight"].float().contiguous()  # gate_proj
            w2 = state_dict[f"{pfx}ffn.w2.weight"].float().contiguous()  # down_proj
            w3 = state_dict[f"{pfx}ffn.w3.weight"].float().contiguous()  # up_proj

            for tensor in [attn_norm, q_proj, k_proj, v_proj, o_proj, ffn_norm, w1, w2, w3]:
                f.write(tensor.numpy().tobytes())

            print(f"Wrote Layer {layer_idx} tensors")

        # 3. Final norm
        norm = state_dict["norm.weight"].float().contiguous()
        f.write(norm.numpy().tobytes())
        print(f"Wrote final norm: {norm.shape}")

        # 4. Untied LM head (when tie_embeddings == False)
        if not cfg.tie_embeddings:
            lm_head = state_dict["output.weight"].float().contiguous()
            f.write(lm_head.numpy().tobytes())
            print(f"Wrote untied lm_head: {lm_head.shape}")

    file_size_mb = os.path.getsize(output_bin_path) / (1024 * 1024)
    print(f"Successfully exported {output_bin_path}: {file_size_mb:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Export Hybrid-0.1-85M PyTorch checkpoint to C++ binary format.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints_qwen7b_draft/final_trained_draft_85m.pt",
        help="Path to the PyTorch .pt checkpoint (default: checkpoints_qwen7b_draft/final_trained_draft_85m.pt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="csrc/hybrid_85m_fp32.bin",
        help="Output binary path (default: csrc/hybrid_85m_fp32.bin)",
    )
    args = parser.parse_args()

    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path):
        project_ckpt = os.path.join(PROJECT_ROOT, ckpt_path)
        if os.path.exists(project_ckpt):
            ckpt_path = project_ckpt
        else:
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    out_path = args.output
    if not os.path.isabs(out_path) and not os.path.exists(os.path.dirname(os.path.abspath(out_path))):
        out_path = os.path.join(PROJECT_ROOT, out_path)

    export_checkpoint_to_bin(ckpt_path, out_path)


if __name__ == "__main__":
    main()
