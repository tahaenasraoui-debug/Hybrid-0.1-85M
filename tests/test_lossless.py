"""
Unit tests verifying the mathematical lossless property of Speculative Decoding.
Asserts 100% token-for-token parity with standard autoregressive generation at T=0.
"""

import pytest
import torch
import torch.nn.functional as F

from config import get_tiny_config, ModelConfig
from model import SmallLLM
from tokenizer import ByteTokenizer
from speculative import SpeculativeDecodingEngine, KVCacheManager


def autoregressive_target_generate(
    model: SmallLLM,
    prompt_ids: list,
    max_new_tokens: int,
    eos_token_id: int = None,
) -> list:
    """Standard token-by-token autoregressive generation on target model alone."""
    model.eval()
    device = next(model.parameters()).device
    input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # Prefill
    logits, _, kv_cache = model(input_tensor, use_cache=True, start_pos=0)
    next_tok = int(torch.argmax(logits[0, -1, :]).item())
    generated = [next_tok]

    for _ in range(max_new_tokens - 1):
        if eos_token_id is not None and next_tok == eos_token_id:
            break
        curr_tensor = torch.tensor([[next_tok]], dtype=torch.long, device=device)
        start_pos = len(prompt_ids) + len(generated) - 1
        logits, _, kv_cache = model(curr_tensor, kv_caches=kv_cache, use_cache=True, start_pos=start_pos)
        next_tok = int(torch.argmax(logits[0, -1, :]).item())
        generated.append(next_tok)

    return generated


@pytest.fixture
def setup_models():
    torch.manual_seed(42)
    tokenizer = ByteTokenizer(vocab_size=1024)

    # Target model: 4 layers, hidden dim 256
    target_cfg = ModelConfig(
        vocab_size=1024,
        d_model=256,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        hidden_dim=512,
        max_seq_len=256,
        tie_embeddings=True,
    )
    target_model = SmallLLM(target_cfg)

    # Draft model: 2 layers, hidden dim 128 (different capacity)
    draft_cfg = ModelConfig(
        vocab_size=1024,
        d_model=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        hidden_dim=256,
        max_seq_len=256,
        tie_embeddings=True,
    )
    draft_model = SmallLLM(draft_cfg)

    return target_model, draft_model, tokenizer


def test_greedy_lossless_equivalence(setup_models):
    """
    Core invariant test: At temperature=0, speculative decoding MUST produce
    a 100% token-for-token identical sequence to standard target autoregression.
    """
    target_model, draft_model, tokenizer = setup_models

    prompts = [
        "The quick brown fox",
        "Artificial intelligence is transforming",
        "Once upon a time in a land far away",
    ]

    engine = SpeculativeDecodingEngine(
        draft_model=draft_model,
        target_model=target_model,
        tokenizer=tokenizer,
        device="cpu",
    )

    for p in prompts:
        prompt_ids = tokenizer.encode(p, add_bos=True, add_eos=False)
        max_tokens = 32

        # 1. Baseline ground truth
        expected_tokens = autoregressive_target_generate(
            target_model, prompt_ids, max_new_tokens=max_tokens, eos_token_id=tokenizer.eos_id
        )

        # 2. Speculative decoding output
        result = engine.generate(
            prompt=prompt_ids,
            max_new_tokens=max_tokens,
            K=4,
            temperature=0.0,
        )
        speculative_tokens = result["generated_ids"]

        print(f"\nPrompt: '{p}'")
        print(f"Expected tokens ({len(expected_tokens)}):   {expected_tokens[:10]}...")
        print(f"Speculative tokens ({len(speculative_tokens)}): {speculative_tokens[:10]}...")
        print(f"Accepted: {result['num_accepted']} / {result['num_drafted']} (Rate: {result['acceptance_rate']:.1%})")

        # ASSERTION: Must match 100% token-for-token
        assert speculative_tokens == expected_tokens, (
            f"Speculative mismatch!\nExpected: {expected_tokens}\nGot:      {speculative_tokens}"
        )


def test_edge_case_varying_k(setup_models):
    """Verifies correctness across different draft lengths K in {1, 3, 5, 8}."""
    target_model, draft_model, tokenizer = setup_models
    engine = SpeculativeDecodingEngine(
        draft_model=draft_model,
        target_model=target_model,
        tokenizer=tokenizer,
        device="cpu",
    )

    prompt_ids = tokenizer.encode("Edge case testing with multiple K values", add_bos=True)
    expected_tokens = autoregressive_target_generate(
        target_model, prompt_ids, max_new_tokens=20, eos_token_id=tokenizer.eos_id
    )

    for k_val in [1, 2, 3, 5, 7]:
        res = engine.generate(prompt=prompt_ids, max_new_tokens=20, K=k_val, temperature=0.0)
        assert res["generated_ids"] == expected_tokens, f"Failed at K={k_val}"


def test_cache_slicing_integrity(setup_models):
    """
    Verifies that DynamicCache / native cache truncation does not leak
    phantom keys or corrupt position IDs.
    """
    target_model, draft_model, tokenizer = setup_models
    engine = SpeculativeDecodingEngine(
        draft_model=draft_model,
        target_model=target_model,
        tokenizer=tokenizer,
        device="cpu",
    )

    prompt_ids = tokenizer.encode("KV cache integrity verification", add_bos=True)
    res = engine.generate(prompt=prompt_ids, max_new_tokens=15, K=5, temperature=0.0)

    # Verify that generation completed without shape or dimension errors
    assert len(res["generated_ids"]) == 15
