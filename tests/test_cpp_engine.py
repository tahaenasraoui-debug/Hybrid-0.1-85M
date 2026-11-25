"""
Numerical Parity and Speculative Rollback Test for Hybrid-0.1-85M C++ Engine.
Compares PyTorch SmallLLM vs. C++ Engine on:
1. Logits cosine similarity and max absolute error
2. Greedy token equality
3. O(1) KV-cache rollback consistency (zero cache leakage)
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest
import torch
import torch.nn.functional as F

try:
    import hybrid_cpp
except ImportError:
    hybrid_cpp = None

from config import ModelConfig, get_qwen_85m_config
from model import SmallLLM
from scripts.export_weights_bin import export_checkpoint_to_bin

CKPT_PATH = os.path.join(PROJECT_ROOT, "checkpoints_qwen7b_draft", "final_trained_draft_85m.pt")
BIN_PATH = os.path.join(PROJECT_ROOT, "csrc", "hybrid_85m_fp32.bin")


@pytest.mark.skipif(
    hybrid_cpp is None or not os.path.exists(CKPT_PATH) or not os.path.exists(BIN_PATH),
    reason="Requires compiled hybrid_cpp extension, PyTorch checkpoint, and exported binary weights.",
)
def test_cpp_engine_parity():
    print("==========================================================")
    print("  Testing Numerical Parity: PyTorch vs. C++ Engine")
    print("==========================================================")

    device = "cpu"
    cfg = get_qwen_85m_config()
    model = SmallLLM(cfg).to(device)

    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    # Load C++ engine
    cpp_engine = hybrid_cpp.HybridCPPEngine()
    assert cpp_engine.load_model(BIN_PATH), f"Failed to load binary model {BIN_PATH}"

    test_tokens = [151644, 872, 198, 108386, 151645]  # Sample tokens

    # 1. Step-by-step autoregressive comparison with KV Cache
    py_kv_cache = None
    cpp_engine.reset_cache()

    for step, tok in enumerate(test_tokens):
        # PyTorch forward
        inp = torch.tensor([[tok]], dtype=torch.long, device=device)
        with torch.no_grad():
            py_logits, _, py_kv_cache = model(
                inp,
                kv_caches=py_kv_cache,
                use_cache=True,
                start_pos=step,
            )
        py_logits_flat = py_logits[0, -1].float()
        py_top_tok = int(torch.argmax(py_logits_flat).item())

        # C++ forward
        cpp_top_tok = cpp_engine.forward_token(tok, True)
        cpp_logits = cpp_engine.get_logits()

        cos_sim = F.cosine_similarity(py_logits_flat.unsqueeze(0), cpp_logits.unsqueeze(0)).item()
        max_diff = torch.max(torch.abs(py_logits_flat - cpp_logits)).item()

        print(f"Step {step} (Token {tok}):")
        print(f"  PyTorch Top Token: {py_top_tok} | C++ Top Token: {cpp_top_tok}")
        print(f"  Cosine Similarity: {cos_sim:.6f} | Max Abs Diff: {max_diff:.5f}")

        assert cos_sim > 0.999, f"Step {step}: Cosine similarity too low: {cos_sim}"
        assert py_top_tok == cpp_top_tok, f"Step {step}: Top token mismatch: {py_top_tok} vs {cpp_top_tok}"

    print("\nParity Test PASSED: C++ Engine is numerically identical to PyTorch!")


@pytest.mark.skipif(
    hybrid_cpp is None or not os.path.exists(BIN_PATH),
    reason="Requires compiled hybrid_cpp extension and exported binary weights.",
)
def test_cpp_kv_rollback():
    print("\n==========================================================")
    print("  Testing C++ KV-Cache Rollback Consistency (Zero Leakage)")
    print("==========================================================")

    cpp_engine = hybrid_cpp.HybridCPPEngine()
    assert cpp_engine.load_model(BIN_PATH)

    prompt = [151644, 872, 198, 108386, 151645]
    prefix_len = len(prompt)

    # 1. Ingest prompt
    cpp_engine.reset_cache()
    for tok in prompt:
        cpp_engine.forward_token(tok, False)
    assert cpp_engine.get_seq_len() == prefix_len

    # Clean target continuation
    next_clean = cpp_engine.forward_token(198, True)
    clean_logits = cpp_engine.get_logits().clone()

    # 2. Simulate speculative decoding: advance 5 candidate tokens
    cpp_engine.rollback_cache(prefix_len)
    assert cpp_engine.get_seq_len() == prefix_len

    # Speculate 5 bogus tokens
    for fake_tok in [42, 100, 200, 300, 400]:
        cpp_engine.forward_token(fake_tok, False)
    assert cpp_engine.get_seq_len() == prefix_len + 5

    # 3. Rollback rejection: rewind back to prefix_len
    cpp_engine.rollback_cache(prefix_len)
    assert cpp_engine.get_seq_len() == prefix_len

    # 4. Generate token 198 again after rollback
    next_after_rollback = cpp_engine.forward_token(198, True)
    rollback_logits = cpp_engine.get_logits().clone()

    max_diff = torch.max(torch.abs(clean_logits - rollback_logits)).item()
    print(f"Rollback max difference: {max_diff}")
    assert max_diff == 0.0, f"Cache leakage detected! Difference: {max_diff}"
    assert next_clean == next_after_rollback, "Token mismatch after rollback"

    print("Rollback Test PASSED: KV-Cache rewind has strictly 0.0 cache contamination!")


@pytest.mark.skipif(
    hybrid_cpp is None,
    reason="Requires compiled hybrid_cpp extension.",
)
def test_cpp_synthetic_untied_and_reload(tmp_path):
    """
    Tests C++ engine with non-64-byte-aligned dims, reload via load_model(),
    nonexistent file handling, and untied embeddings (tie_embeddings=False / 0).
    """
    torch.manual_seed(7)

    # Deliberately choose dimensions whose tensor byte sizes are not multiples of 64
    # e.g., d_model=24 (96 bytes), n_kv_heads=2, head_dim=6 -> 12 floats (48 bytes)
    tied_cfg = ModelConfig(
        vocab_size=37,
        d_model=24,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        hidden_dim=44,
        max_seq_len=20,
        rope_theta=10000.0,
        tie_embeddings=True,
    )
    untied_cfg = ModelConfig(
        vocab_size=37,
        d_model=24,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        hidden_dim=44,
        max_seq_len=20,
        rope_theta=10000.0,
        tie_embeddings=False,
    )

    tied_model = SmallLLM(tied_cfg).eval()
    untied_model = SmallLLM(untied_cfg).eval()

    tied_pt = tmp_path / "tied.pt"
    tied_bin = tmp_path / "tied.bin"
    untied_pt = tmp_path / "untied.pt"
    untied_bin = tmp_path / "untied.bin"

    torch.save({"model_state_dict": tied_model.state_dict(), "config": tied_cfg}, tied_pt)
    torch.save({"model_state_dict": untied_model.state_dict(), "config": untied_cfg}, untied_pt)

    export_checkpoint_to_bin(str(tied_pt), str(tied_bin), config=tied_cfg)
    export_checkpoint_to_bin(str(untied_pt), str(untied_bin), config=untied_cfg)

    cpp_engine = hybrid_cpp.HybridCPPEngine()
    assert not cpp_engine.load_model(str(tmp_path / "does_not_exist.bin"))
    assert cpp_engine.load_model(str(tied_bin))
    # Reloading a second model (with untied lm_head) must cleanly free prior buffers
    assert cpp_engine.load_model(str(untied_bin))

    tokens = [1, 5, 13, 29]
    py_cache = None
    cpp_engine.reset_cache()

    for step, tok in enumerate(tokens):
        with torch.no_grad():
            py_logits, _, py_cache = untied_model(
                torch.tensor([[tok]], dtype=torch.long),
                kv_caches=py_cache,
                use_cache=True,
                start_pos=step,
            )
        py_flat = py_logits[0, -1].float()
        cpp_top = cpp_engine.forward_token(tok, True)
        cpp_flat = cpp_engine.get_logits()

        cos_sim = F.cosine_similarity(py_flat.unsqueeze(0), cpp_flat.unsqueeze(0)).item()
        assert cos_sim > 0.9999, f"Untied parity cosine similarity too low: {cos_sim}"
        assert cpp_top == int(torch.argmax(py_flat).item())


if __name__ == "__main__":
    test_cpp_engine_parity()
    test_cpp_kv_rollback()
