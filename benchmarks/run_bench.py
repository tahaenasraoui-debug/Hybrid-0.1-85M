#!/usr/bin/env python3
"""
Benchmarking Suite for Lossless Speculative Decoding vs. Autoregressive Baseline.

Measures:
1. Plain Autoregressive Baseline (tokens/sec)
2. Speculative Decoding (tokens/sec)
3. Net Speedup (x, honestly reported)
4. Draft Lookahead Length (k in {2, 3, 4, 5, 6})
5. Empirical Token Acceptance Rate (alpha) and Mean Accepted Tokens per Step (tau)

Outputs a clean Markdown table suitable for README.md.
"""

import argparse
import os
import sys
import time
from typing import Any, Dict, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from config import get_qwen_85m_config
from model import SmallLLM
from src.speculative import SpeculativeEngine
from train import get_compatible_device


# Diverse benchmark prompts covering coding, reasoning, and instruction following
BENCHMARK_PROMPTS = [
    "Write a Python function to compute the Fibonacci sequence using memoization.",
    "Explain the difference between synchronous and asynchronous execution in computer science.",
    "Calculate the derivative of f(x) = 3*x^3 + 5*x^2 - 12*x + 7.",
    "Summarize the core principles of the Theory of Relativity in simple terms.",
    "Write an SQL query to find the top 5 highest-paid employees in each department.",
    "Explain how a Transformer model uses Grouped Query Attention to save memory.",
    "What are the main causes and effects of ocean acidification?",
    "Write a recursive function to reverse a singly linked list in C++.",
]


def run_plain_autoregressive_baseline(
    target_model, tokenizer, prompts: List[str], max_new_tokens: int = 64, device: str = "cuda"
) -> Dict[str, float]:
    """Measures standard autoregressive baseline generation speed."""
    print("\n--- Running Plain Autoregressive Baseline ---")
    total_tokens = 0
    total_time = 0.0

    target_model.eval()

    for i, p in enumerate(prompts):
        messages = [{"role": "user", "content": p}]
        try:
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = tokenizer(formatted, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]

        torch.cuda.synchronize() if device == "cuda" else None
        t0 = time.perf_counter()

        with torch.no_grad():
            out = target_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        torch.cuda.synchronize() if device == "cuda" else None
        t1 = time.perf_counter()

        gen_tokens = out.shape[1] - input_len
        total_tokens += gen_tokens
        total_time += (t1 - t0)
        print(f" Prompt {i+1}/{len(prompts)}: {gen_tokens} tokens in {t1 - t0:.2f}s ({gen_tokens / max(1e-4, t1 - t0):.1f} tok/s)")

    baseline_speed = total_tokens / max(1e-4, total_time)
    print(f"Baseline Throughput: {baseline_speed:.2f} tokens/sec")
    return {"tokens_per_sec": baseline_speed, "total_tokens": total_tokens, "total_time": total_time}


def run_speculative_benchmarks(
    target_model,
    draft_model,
    tokenizer,
    prompts: List[str],
    k_values: List[int] = [2, 3, 4, 5, 6],
    max_new_tokens: int = 64,
    device: str = "cuda",
) -> List[Dict[str, Any]]:
    """Runs speculative decoding benchmarks across draft lookahead k values."""
    engine = SpeculativeEngine(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        device=device,
    )

    results = []

    for k in k_values:
        print(f"\n--- Testing Speculative Decoding with Lookahead k={k} ---")
        total_tokens = 0
        total_drafted = 0
        total_accepted = 0
        total_time = 0.0
        total_rounds = 0

        for i, p in enumerate(prompts):
            messages = [{"role": "user", "content": p}]
            try:
                formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            torch.cuda.synchronize() if device == "cuda" else None
            t0 = time.perf_counter()

            res = engine.generate(
                prompt=formatted,
                max_new_tokens=max_new_tokens,
                k=k,
                temperature=0.0,
            )

            torch.cuda.synchronize() if device == "cuda" else None
            t1 = time.perf_counter()

            gen_tokens = res["num_generated"]
            total_tokens += gen_tokens
            total_drafted += res["num_drafted"]
            total_accepted += res["num_accepted"]
            total_rounds += res["num_speculative_rounds"]
            total_time += (t1 - t0)
            print(f" k={k} | Prompt {i+1}/{len(prompts)}: {gen_tokens} tokens in {t1 - t0:.2f}s ({gen_tokens / max(1e-4, t1 - t0):.1f} tok/s, accepted: {res['num_accepted']}/{res['num_drafted']})", flush=True)

        speed = total_tokens / max(1e-4, total_time)
        alpha = (total_accepted / max(1, total_drafted)) * 100.0
        tau = total_accepted / max(1, total_rounds)

        print(f" k={k} Results: {speed:.2f} tok/s | Acceptance: {alpha:.1f}% | Tau: {tau:.2f} tok/step")
        results.append({
            "k": k,
            "tokens_per_sec": speed,
            "acceptance_rate": alpha,
            "mean_accepted_tau": tau,
            "total_tokens": total_tokens,
            "total_time": total_time,
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints_overnight/final_trained_draft_85m.pt")
    parser.add_argument("--max_tokens", type=int, default=64)
    args = parser.parse_args()

    device = str(get_compatible_device())
    print(f"============================================================")
    print(f"  Speculative Decoding Empirical Benchmarks")
    print(f"  Target: {args.target} | Device: {device}")
    print(f"============================================================")

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.target, trust_remote_code=True, local_files_only=True)

    # 2. Load Target Model (in 4-bit on CUDA to ensure resident VRAM < 2.8GB)
    print("Loading Target Model...")
    if device == "cuda":
        try:
            bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            target_model = AutoModelForCausalLM.from_pretrained(
                args.target, quantization_config=bnb_config, device_map="auto", local_files_only=True
            )
        except Exception:
            target_model = AutoModelForCausalLM.from_pretrained(
                args.target, torch_dtype=torch.float16, local_files_only=True
            ).to(device)
    else:
        target_model = AutoModelForCausalLM.from_pretrained(
            args.target, torch_dtype=torch.float32, local_files_only=True
        ).to(device)
    target_model.eval()

    # 3. Load Custom 85M Draft Model
    print("Loading Custom 85M Draft Model...")
    draft_cfg = get_qwen_85m_config()
    draft_model = SmallLLM(draft_cfg).to(device)

    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path):
        ckpt_path = "./checkpoints_overnight/latest_draft.pt"

    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        draft_model.load_state_dict(ckpt["model_state_dict"])
    else:
        print("Note: Checkpoint not found; using initialized weights for architecture benchmarking.")
    draft_model.eval()

    # 4. Measure Baseline
    baseline = run_plain_autoregressive_baseline(
        target_model, tokenizer, BENCHMARK_PROMPTS, max_new_tokens=args.max_tokens, device=device
    )

    # 5. Measure Speculative Across k in {2, 3, 4, 5, 6}
    spec_results = run_speculative_benchmarks(
        target_model,
        draft_model,
        tokenizer,
        BENCHMARK_PROMPTS,
        k_values=[2, 3, 4, 5, 6],
        max_new_tokens=args.max_tokens,
        device=device,
    )

    # 6. Format Markdown Table
    print("\n" + "=" * 70)
    print("              EMPIRICAL BENCHMARK RESULTS")
    print("=" * 70)

    md_table = []
    md_table.append("| Generation Mode | Draft Length ($k$) | Throughput (tok/s) | Net Speedup | Acceptance Rate ($\\alpha$) | Mean Accepted ($\\tau$) |")
    md_table.append("| :--- | :---: | :---: | :---: | :---: | :---: |")

    b_speed = baseline["tokens_per_sec"]
    md_table.append(f"| **Plain Baseline (Target Only)** | — | **{b_speed:.2f}** | 1.00× | — | — |")

    for r in spec_results:
        k = r["k"]
        s = r["tokens_per_sec"]
        speedup = s / max(1e-4, b_speed)
        alpha = r["acceptance_rate"]
        tau = r["mean_accepted_tau"]
        md_table.append(
            f"| **Speculative Decoding** | {k} | **{s:.2f}** | **{speedup:.2f}×** | {alpha:.1f}% | {tau:.2f} tok/step |"
        )

    table_str = "\n".join(md_table)
    print("\n" + table_str + "\n")

    # Save to results/benchmark_table.md
    os.makedirs("results", exist_ok=True)
    with open("results/benchmark_table.md", "w") as f:
        f.write(table_str + "\n")
    print("Saved benchmark table to results/benchmark_table.md")


if __name__ == "__main__":
    main()
