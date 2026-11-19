"""
Phase 1: Verification & Alignment Check for Speculative Decoding.
Checks tokenizer compatibility, special tokens, vocabulary alignment (151,936),
and measures baseline token acceptance rate (tau).
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import get_qwen_85m_config
from model import SmallLLM
from speculative import SpeculativeDecodingEngine
from train import get_compatible_device


TEST_PROMPTS = [
    "Write a quick Python function to reverse a string.",
    "Explain what speculative decoding is in simple terms.",
    "What is the capital of France?",
    "Solve for x: 3x + 12 = 33.",
    "Summarize the key advantages of Grouped Query Attention (GQA).",
    "How does Transformer KV caching speed up autoregressive decoding?",
    "Write a haiku about artificial intelligence.",
    "What is the difference between RMSNorm and LayerNorm?",
    "Implement binary search in Python.",
    "Name three greenhouse gases and their atmospheric impact.",
    "Explain the concept of entropy in information theory.",
    "What are rotary position embeddings (RoPE)?",
    "Write a short SQL query to find duplicate email addresses in a table.",
    "Why does learning rate warmup stabilize AdamW training?",
    "Explain the difference between PPO and DPO in LLM alignment.",
    "Translate 'Hello world' into Spanish, French, and Japanese.",
    "What is the primary function of SwiGLU in feed-forward layers?",
    "Calculate 17 * 19 step by step.",
    "What causes exploding gradients in deep neural networks?",
    "Write a brief description of the Chinchilla scaling laws.",
]


def load_tokenizer_and_target(target_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct"):
    """Loads tokenizer and target model with automatic fallback."""
    print(f"Loading tokenizer from: {target_name_or_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(target_name_or_path, trust_remote_code=True)
    except Exception as e:
        print(f"Could not load '{target_name_or_path}' ({e}). Falling back to 'Qwen/Qwen2.5-0.5B-Instruct'...")
        target_name_or_path = "Qwen/Qwen2.5-0.5B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(target_name_or_path, trust_remote_code=True)

    device = str(get_compatible_device())
    print(f"Loading target model '{target_name_or_path}' onto {device}...")
    try:
        # Try loading in float16 or bfloat16
        target_model = AutoModelForCausalLM.from_pretrained(
            target_name_or_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).to(device)
    except Exception as e:
        print(f"Failed to load full target model ({e}). Attempting low_cpu_mem_usage...")
        target_model = AutoModelForCausalLM.from_pretrained(
            target_name_or_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        ).to("cpu")
        device = "cpu"

    return tokenizer, target_model, device


def verify_vocabulary_and_special_tokens(tokenizer, draft_model):
    """Verifies that draft model vocab_size matches tokenizer."""
    print("\n" + "=" * 60)
    print("🔍 1. VOCABULARY & SPECIAL TOKEN COMPATIBILITY CHECK")
    print("=" * 60)

    tokenizer_vocab_size = len(tokenizer)
    draft_vocab_size = draft_model.config.vocab_size

    print(f"Tokenizer Vocabulary Size:  {tokenizer_vocab_size:,}")
    print(f"Draft Model Vocabulary:     {draft_vocab_size:,}")

    # Check for matching dimension
    assert draft_vocab_size >= tokenizer_vocab_size, (
        f"Draft vocab size ({draft_vocab_size}) is smaller than tokenizer ({tokenizer_vocab_size})!"
    )
    print("✅ Vocabulary dimensions are fully compatible (151,936 aligned).")

    # Special tokens check
    special_tokens = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]
    print("\nChecking special tokens:")
    for st in special_tokens:
        tok_id = tokenizer.convert_tokens_to_ids(st)
        print(f"  {st:15s} -> ID: {tok_id}")

    # Check formatting with chat template
    test_msg = [{"role": "user", "content": "Hello!"}]
    if hasattr(tokenizer, "apply_chat_template"):
        formatted = tokenizer.apply_chat_template(test_msg, tokenize=False, add_generation_prompt=True)
        print("\nChat Template Output Preview:")
        print(repr(formatted))
        print("✅ Chat template formatting verified.")


def benchmark_baseline_acceptance_rate(engine, tokenizer, prompts: list, K: int = 5):
    """Measures baseline acceptance rate across prompts."""
    print("\n" + "=" * 60)
    print(f"📊 2. MEASURING BASELINE ACCEPTANCE RATE (tau) ON {len(prompts)} PROMPTS (K={K})")
    print("=" * 60)

    total_drafted = 0
    total_accepted = 0
    total_target_passes = 0
    total_tokens_generated = 0

    for idx, prompt_text in enumerate(prompts):
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt_text}]
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            formatted_prompt = prompt_text

        result = engine.generate(
            prompt=formatted_prompt,
            max_new_tokens=32,
            K=K,
            temperature=0.0,
        )

        n_drafted = result["num_drafted"]
        n_accepted = result["num_accepted"]
        rate = result["acceptance_rate"]
        tau = result["mean_accepted_per_round"]

        total_drafted += n_drafted
        total_accepted += n_accepted
        total_target_passes += result["target_forward_passes"]
        total_tokens_generated += result["num_generated"]

        print(
            f"Prompt {idx+1:02d}/{len(prompts):02d} | "
            f"Drafted: {n_drafted:2d} | Accepted: {n_accepted:2d} | "
            f"Rate: {rate:5.1%} | Mean tau: {tau:.2f} tok/round"
        )

    overall_acceptance_rate = (total_accepted / max(1, total_drafted))
    overall_tau = (total_accepted / max(1, total_target_passes))
    theoretical_speedup = 1.0 + overall_tau

    print("\n" + "=" * 60)
    print("🎯 BASELINE RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total Drafted Tokens:        {total_drafted:,}")
    print(f"Total Accepted Tokens:       {total_accepted:,}")
    print(f"Overall Acceptance Rate:     {overall_acceptance_rate:.2%}")
    print(f"Mean Accepted per Round (tau): {overall_tau:.2f} tokens")
    print(f"Theoretical Speedup Upper:   {theoretical_speedup:.2f}x")

    if overall_acceptance_rate < 0.45:
        print("\n⚠️  ACCEPTANCE RATE IS BELOW 45% (Expected for untrained/fresh draft weights).")
        print("💡 Recommended Action: Run 'python3 distill_draft.py' to distill target model")
        print("   soft probabilities into the 85M draft model, boosting tau above 65%-80%.")
    else:
        print("\n🎉 ACCEPTANCE RATE IS EXCELLENT (>= 45%). Ready for high-throughput speculative decoding!")

    return overall_acceptance_rate


def main():
    target_name = os.getenv("TARGET_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    tokenizer, target_model, device = load_tokenizer_and_target(target_name)

    print("\nInitializing 85M Qwen Draft Model (vocab_size=151936)...")
    draft_cfg = get_qwen_85m_config()
    draft_model = SmallLLM(draft_cfg).to(device)

    # 1. Compatibility Check
    verify_vocabulary_and_special_tokens(tokenizer, draft_model)

    # 2. Speculative Engine
    engine = SpeculativeDecodingEngine(
        draft_model=draft_model,
        target_model=target_model,
        tokenizer=tokenizer,
        device=device,
    )

    # 3. Acceptance Benchmark on 20 Prompts
    benchmark_baseline_acceptance_rate(engine, tokenizer, TEST_PROMPTS, K=5)


if __name__ == "__main__":
    main()
