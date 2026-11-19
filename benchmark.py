"""
Phase 3: Benchmarking & Profiling Suite for Speculative Decoding.
Evaluates:
1. Baseline: Standard Autoregressive decoding on Target Model.
2. Static Speculative Decoding: Fixed K in {3, 5, 7}.
3. Adaptive Speculative Decoding: Confidence-gated dynamic K.
Generates Markdown results table and Matplotlib charts to results/benchmark_plot.png.
"""

import os
import time
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import get_qwen_85m_config
from model import SmallLLM
from speculative import SpeculativeDecodingEngine
from train import get_compatible_device


# Evaluation datasets across 3 distinct domains
BENCHMARK_PROMPTS = {
    "Code (HumanEval-style)": [
        "Write a Python function to check if a binary tree is symmetric.",
        "Implement merge sort in Python with detailed comments.",
        "Write a function to find the longest common prefix among an array of strings.",
        "Implement a LRU Cache in Python using OrderedDict.",
        "Write a regular expression to validate IPv4 addresses.",
        "Write a function that returns the prime factors of an integer.",
        "Implement Dijkstra's algorithm for shortest path in Python.",
        "Write a function to detect cycle in a directed graph.",
    ],
    "Math (GSM8K-style)": [
        "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins with four every day. She sells the remainder at the market for $2 each. How much in dollars does she make every day?",
        "A train travels 180 miles in 3 hours. If it increases its speed by 20 mph, how long will it take to travel 320 miles?",
        "If 5 machines take 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
        "A bookstore sells hardcover books for $25 and paperbacks for $12. If they sold 40 books for a total of $675, how many paperbacks were sold?",
        "Solve for x: 4(x - 3) + 7 = 2(x + 5) - 3.",
        "What is the sum of all integers from 1 to 100 inclusive?",
        "A circle has a circumference of 31.4 cm. What is its area (use pi = 3.14)?",
        "If a car depreciates by 15% each year, what will be the value of a $30,000 car after 2 years?",
    ],
    "Instruction Following (Dolly-style)": [
        "Explain the key differences between renewable and non-renewable energy sources.",
        "What was the significance of the Rosetta Stone in understanding ancient history?",
        "Provide three actionable tips for improving sleep quality based on scientific consensus.",
        "Summarize the plot of Romeo and Juliet in three bullet points.",
        "Explain how vaccines prepare the human immune system to fight infections.",
        "What are the main functions of the United Nations Security Council?",
        "Compare classical conditioning with operant conditioning with everyday examples.",
        "Describe the water cycle and its importance to Earth's ecosystems.",
    ],
}


def run_autoregressive_baseline(target_model, tokenizer, prompt_text: str, max_new_tokens: int = 64, device="cpu"):
    """Runs standard single-token autoregressive generation on target model."""
    if hasattr(tokenizer, "apply_chat_template"):
        formatted = tokenizer.apply_chat_template([{"role": "user", "content": prompt_text}], tokenize=False, add_generation_prompt=True)
    else:
        formatted = prompt_text

    input_ids = tokenizer.encode(formatted)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    t0 = time.perf_counter()
    with torch.no_grad():
        # Prefill
        out = target_model(input_ids=input_tensor, use_cache=True)
        cache = out.past_key_values
        next_tok = int(torch.argmax(out.logits[0, -1, :]).item())
        gen_tokens = [next_tok]

        for _ in range(max_new_tokens - 1):
            if next_tok == getattr(tokenizer, "eos_token_id", None):
                break
            tok_tensor = torch.tensor([[next_tok]], dtype=torch.long, device=device)
            out = target_model(input_ids=tok_tensor, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            next_tok = int(torch.argmax(out.logits[0, -1, :]).item())
            gen_tokens.append(next_tok)

    dt_ms = (time.perf_counter() - t0) * 1000.0
    tok_per_sec = (len(gen_tokens) / (dt_ms / 1000.0)) if dt_ms > 0 else 0.0
    ms_per_tok = (dt_ms / len(gen_tokens)) if len(gen_tokens) > 0 else 0.0

    return {
        "num_tokens": len(gen_tokens),
        "wall_time_ms": dt_ms,
        "tokens_per_sec": tok_per_sec,
        "ms_per_tok": ms_per_tok,
    }


def evaluate_benchmark(
    target_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    draft_checkpoint_path: str = None,
    output_dir: str = "./results",
    max_new_tokens: int = 64,
):
    os.makedirs(output_dir, exist_ok=True)
    device = str(get_compatible_device())
    print(f"Loading benchmark suite on device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(target_name, trust_remote_code=True)
    target_model = AutoModelForCausalLM.from_pretrained(
        target_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)

    # Load 85M draft model
    draft_cfg = get_qwen_85m_config()
    draft_model = SmallLLM(draft_cfg).to(device)
    if draft_checkpoint_path and os.path.exists(draft_checkpoint_path):
        print(f"Loading distilled draft checkpoint from: {draft_checkpoint_path}")
        ckpt = torch.load(draft_checkpoint_path, map_location=device)
        draft_model.load_state_dict(ckpt["model_state_dict"])

    engine = SpeculativeDecodingEngine(
        draft_model=draft_model,
        target_model=target_model,
        tokenizer=tokenizer,
        device=device,
    )

    modes = [
        {"name": "Baseline (Target Alone)", "type": "baseline"},
        {"name": "Speculative (K=3)", "type": "spec", "K": 3, "conf": None},
        {"name": "Speculative (K=5)", "type": "spec", "K": 5, "conf": None},
        {"name": "Speculative (K=7)", "type": "spec", "K": 7, "conf": None},
        {"name": "Adaptive Speculative (Conf=0.6)", "type": "spec", "K": 5, "conf": 0.60},
    ]

    domain_results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for domain, prompts in BENCHMARK_PROMPTS.items():
        print("\n" + "=" * 65)
        print(f"🚀 BENCHMARKING DOMAIN: {domain} ({len(prompts)} prompts)")
        print("=" * 65)
        domain_results[domain] = {}

        for mode in modes:
            m_name = mode["name"]
            print(f"\nEvaluating configuration: {m_name}...")
            total_time_ms = 0.0
            total_tokens = 0
            total_drafted = 0
            total_accepted = 0

            for p_idx, prompt in enumerate(prompts):
                if hasattr(tokenizer, "apply_chat_template"):
                    formatted_p = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
                else:
                    formatted_p = prompt

                if mode["type"] == "baseline":
                    res = run_autoregressive_baseline(target_model, tokenizer, prompt, max_new_tokens=max_new_tokens, device=device)
                    total_time_ms += res["wall_time_ms"]
                    total_tokens += res["num_tokens"]
                else:
                    res = engine.generate(
                        prompt=formatted_p,
                        max_new_tokens=max_new_tokens,
                        K=mode["K"],
                        confidence_threshold=mode["conf"],
                        temperature=0.0,
                    )
                    total_time_ms += res["wall_time_ms"]
                    total_tokens += res["num_generated"]
                    total_drafted += res["num_drafted"]
                    total_accepted += res["num_accepted"]

            avg_tok_per_sec = (total_tokens / (total_time_ms / 1000.0)) if total_time_ms > 0 else 0.0
            ms_per_tok = (total_time_ms / total_tokens) if total_tokens > 0 else 0.0
            acc_rate = (total_accepted / max(1, total_drafted)) if total_drafted > 0 else 0.0

            domain_results[domain][m_name] = {
                "tokens_per_sec": avg_tok_per_sec,
                "ms_per_tok": ms_per_tok,
                "acceptance_rate": acc_rate,
            }

            print(f"  -> {avg_tok_per_sec:.1f} tok/s | {ms_per_tok:.1f} ms/tok | Acceptance Rate: {acc_rate:.1%}")

    # Generate Markdown Table
    print("\n" + "=" * 80)
    print("📊 SPECULATIVE DECODING BENCHMARK RESULTS")
    print("=" * 80)
    md_table = "| Domain | Configuration | Throughput (tok/s) | Latency (ms/tok) | Acceptance Rate (%) | Speedup |\n"
    md_table += "| :--- | :--- | :---: | :---: | :---: | :---: |\n"

    speedup_plot_data = {}

    for domain, configs in domain_results.items():
        baseline_tok_s = configs["Baseline (Target Alone)"]["tokens_per_sec"]
        speedup_plot_data[domain] = {}

        for cfg_name, stats in configs.items():
            speedup = (stats["tokens_per_sec"] / max(1e-5, baseline_tok_s))
            acc_str = f"{stats['acceptance_rate']:.1%}" if "Speculative" in cfg_name else "-"
            speedup_str = f"**{speedup:.2f}x**" if "Speculative" in cfg_name else "1.00x (ref)"
            md_table += f"| {domain} | {cfg_name} | {stats['tokens_per_sec']:.1f} | {stats['ms_per_tok']:.1f} | {acc_str} | {speedup_str} |\n"
            speedup_plot_data[domain][cfg_name] = speedup

    print(md_table)

    # Plot results with Matplotlib
    plot_path = os.path.join(output_dir, "benchmark_plot.png")
    generate_benchmark_plots(domain_results, plot_path)
    print(f"\n🖼️ Benchmark plots generated and saved to: {plot_path}")

    return domain_results


def generate_benchmark_plots(domain_results: dict, output_path: str):
    """Creates side-by-side comparison bar charts for Throughput and Speedup."""
    domains = list(domain_results.keys())
    configs = list(domain_results[domains[0]].keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    x = np.arange(len(domains))
    width = 0.16

    colors = ["#4A5568", "#3182CE", "#63B3ED", "#4FD1C5", "#38A169"]

    # Chart 1: Throughput (tokens/s)
    for i, cfg_name in enumerate(configs):
        vals = [domain_results[d][cfg_name]["tokens_per_sec"] for d in domains]
        ax1.bar(x + (i - len(configs) / 2) * width + width / 2, vals, width, label=cfg_name, color=colors[i % len(colors)])

    ax1.set_ylabel("Throughput (Tokens / Second)", fontsize=11, fontweight="bold")
    ax1.set_title("Decoding Throughput by Domain", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([d.split("(")[0].strip() for d in domains], fontsize=10)
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Chart 2: Speedup vs Baseline
    for i, cfg_name in enumerate(configs):
        if "Baseline" in cfg_name:
            continue
        vals = [
            domain_results[d][cfg_name]["tokens_per_sec"] / max(1e-5, domain_results[d]["Baseline (Target Alone)"]["tokens_per_sec"])
            for d in domains
        ]
        ax2.bar(x + (i - (len(configs) - 1) / 2) * width, vals, width, label=cfg_name, color=colors[i % len(colors)])

    ax2.axhline(1.0, color="gray", linestyle=":", linewidth=1.5, label="Baseline (1.0x)")
    ax2.set_ylabel("Speedup Multiplier (x)", fontsize=11, fontweight="bold")
    ax2.set_title("Speculative Speedup Multiplier vs Target Alone", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([d.split("(")[0].strip() for d in domains], fontsize=10)
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


if __name__ == "__main__":
    evaluate_benchmark()
