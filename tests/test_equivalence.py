"""
Mandatory Verification Suite for Lossless Speculative Decoding.

Tests:
1. Greedy Equivalence Test: At temperature=0.0, asserts that speculative_generate()
   produces 100% token-identical sequences to standard autoregressive target generation
   across diverse prompts and lookahead values k in {2, 3, 4, 5}.
2. KV Cache Integrity Test: Asserts that target logits after a stateful KV-cache rollback
   match a fresh forward pass on the truncated sequence within atol=1e-4.
"""

import pytest
import torch
import torch.nn.functional as F

from config import ModelConfig
from model import SmallLLM
from tokenizer import ByteTokenizer
from src.speculative import speculative_generate, SpeculativeEngine
from speculative.cache_manager import KVCacheManager


def plain_autoregressive_generate(
    model: SmallLLM,
    prompt_ids: list,
    max_new_tokens: int,
    eos_token_id: int = None,
) -> list:
    """Ground truth token-by-token autoregressive generation on target model alone."""
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
        max_seq_len=512,
        tie_embeddings=True,
    )
    target_model = SmallLLM(target_cfg)

    # Draft model: 2 layers, hidden dim 128
    draft_cfg = ModelConfig(
        vocab_size=1024,
        d_model=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        hidden_dim=256,
        max_seq_len=512,
        tie_embeddings=True,
    )
    draft_model = SmallLLM(draft_cfg)

    return target_model, draft_model, tokenizer


def test_greedy_lossless_equivalence_multi_prompts(setup_models):
    """
    Asserts 100% token-identical outputs between speculative_generate
    and plain autoregressive target generation at temperature=0.0 across 50 prompts.
    """
    target_model, draft_model, tokenizer = setup_models

    prompts = [
        f"Prompt variation {i}: The quick brown fox jumps over the lazy dog in scenario {i * 7}"
        for i in range(50)
    ]

    for i, p in enumerate(prompts):
        prompt_ids = tokenizer.encode(p, add_bos=True, add_eos=False)
        max_tokens = 24
        k_val = 2 + (i % 5)  # Test across k in {2, 3, 4, 5, 6}

        # 1. Ground truth autoregressive generation
        ground_truth_ids = plain_autoregressive_generate(
            target_model, prompt_ids, max_new_tokens=max_tokens
        )

        # 2. Speculative decoding generation
        spec_result = speculative_generate(
            target_model=target_model,
            draft_model=draft_model,
            tokenizer=tokenizer,
            prompt=prompt_ids,
            max_new_tokens=max_tokens,
            k=k_val,
            temperature=0.0,
            device="cpu",
        )
        spec_ids = spec_result["generated_ids"]

        # Strict token-for-token equivalence
        assert spec_ids == ground_truth_ids, (
            f"Equivalence failed on prompt {i} with k={k_val}!\n"
            f"Expected: {ground_truth_ids}\n"
            f"Got:      {spec_ids}"
        )


def test_kv_cache_rollback_integrity(setup_models):
    """
    Asserts that target logits after a KV-cache rollback match a fresh forward pass
    on the truncated sequence within atol=1e-4.
    """
    target_model, _, tokenizer = setup_models
    target_model.eval()

    prompt = "Artificial Intelligence systems require strict cache consistency."
    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    input_tensor = torch.tensor([prompt_ids], dtype=torch.long)

    # 1. Forward prompt through model with KV-cache
    logits_prefill, _, kv_cache = target_model(input_tensor, use_cache=True, start_pos=0)
    prefix_len = len(prompt_ids)

    # 2. Forward 5 candidate tokens to simulate drafting
    draft_tokens = [12, 45, 78, 91, 103]
    draft_tensor = torch.tensor([draft_tokens], dtype=torch.long)
    logits_draft, _, kv_cache = target_model(
        draft_tensor, kv_caches=kv_cache, use_cache=True, start_pos=prefix_len
    )
    full_len = KVCacheManager.get_seq_len(kv_cache)
    assert full_len == prefix_len + len(draft_tokens)

    # 3. Simulate rollback: reject after 2 tokens, rollback cache to prefix_len + 2
    retained_draft_len = 2
    target_truncated_len = prefix_len + retained_draft_len
    rolled_back_cache = KVCacheManager.truncate_cache(kv_cache, target_truncated_len)
    assert KVCacheManager.get_seq_len(rolled_back_cache) == target_truncated_len

    # 4. Forward next token using the rolled-back cache
    next_test_token = 205
    test_tensor = torch.tensor([[next_test_token]], dtype=torch.long)
    cached_logits, _, _ = target_model(
        test_tensor, kv_caches=rolled_back_cache, use_cache=True, start_pos=target_truncated_len
    )

    # 5. Run a fresh forward pass from scratch on the exact truncated sequence + next_test_token
    full_ground_truth_seq = prompt_ids + draft_tokens[:retained_draft_len] + [next_test_token]
    fresh_tensor = torch.tensor([full_ground_truth_seq], dtype=torch.long)
    fresh_logits, _, _ = target_model(fresh_tensor, use_cache=False)

    # 6. Compare logits at the final position
    cached_last_logit = cached_logits[0, -1, :]
    fresh_last_logit = fresh_logits[0, -1, :]

    diff = torch.max(torch.abs(cached_last_logit - fresh_last_logit)).item()
    assert torch.allclose(cached_last_logit, fresh_last_logit, atol=1e-4), (
        f"KV cache rollback integrity failed! Max absolute diff: {diff:.6f} > 1e-4"
    )
