"""
MLSys Benchmark Suite: PyTorch Eager vs. C++ Engine for Hybrid-0.1-85M.
Measures:
1. Inter-token Draft Latency (ms/tok)
2. KV-Cache Rollback Overhead (ns vs us)
3. Speculative Drafting Cycles across Horizon k
4. Memory Allocation & Footprint
"""

import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import hybrid_cpp
from config import get_qwen_85m_config
from model import SmallLLM
from speculative.cache_manager import KVCacheManager


def benchmark_systems():
    print("================================================================================")
    print("   MLSys System Benchmark: Hybrid-0.1-85M (PyTorch Eager vs. C++ Engine)")
    print("================================================================================")

    device = "cpu"
    cfg = get_qwen_85m_config()
    model = SmallLLM(cfg).to(device)
    ckpt_path = os.path.join(PROJECT_ROOT, "checkpoints_qwen7b_draft/final_trained_draft_85m.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    cpp_engine = hybrid_cpp.HybridCPPEngine()
    bin_path = os.path.join(PROJECT_ROOT, "csrc/hybrid_85m_fp32.bin")
    cpp_engine.load_model(bin_path)

    prompt = [151644, 872, 198, 108386, 151645, 198, 151644, 77091, 198]
    prompt_len = len(prompt)

    # ---------------------------------------------------------
    # 1. Autoregressive Draft Latency Comparison
    # ---------------------------------------------------------
    print("\n--- 1. Autoregressive Single-Token Draft Latency ---")
    STEPS = 20

    # PyTorch Warmup & Benchmark
    py_cache = None
    for tok in prompt:
        inp = torch.tensor([[tok]], dtype=torch.long, device=device)
        with torch.no_grad():
            _, _, py_cache = model(inp, kv_caches=py_cache, use_cache=True, start_pos=KVCacheManager.get_seq_len(py_cache))

    cur_tok = 198
    py_latencies = []
    for _ in range(STEPS):
        inp = torch.tensor([[cur_tok]], dtype=torch.long, device=device)
        t0 = time.perf_counter()
        with torch.no_grad():
            logits, _, py_cache = model(inp, kv_caches=py_cache, use_cache=True, start_pos=KVCacheManager.get_seq_len(py_cache))
            cur_tok = int(torch.argmax(logits[0, -1]).item())
        t1 = time.perf_counter()
        py_latencies.append((t1 - t0) * 1000.0)

    py_mean_ms = sum(py_latencies) / len(py_latencies)
    py_tok_per_sec = 1000.0 / py_mean_ms

    # C++ Warmup & Benchmark
    cpp_engine.reset_cache()
    for tok in prompt:
        cpp_engine.forward_token(tok, False)

    cur_tok = 198
    cpp_latencies = []
    for _ in range(STEPS):
        t0 = time.perf_counter()
        cur_tok = cpp_engine.forward_token(cur_tok, False)
        t1 = time.perf_counter()
        cpp_latencies.append((t1 - t0) * 1000.0)

    cpp_mean_ms = sum(cpp_latencies) / len(cpp_latencies)
    cpp_tok_per_sec = 1000.0 / cpp_mean_ms

    speedup = py_mean_ms / cpp_mean_ms

    print(f"PyTorch Eager (CPU):    {py_mean_ms:6.2f} ms/token ({py_tok_per_sec:5.1f} tok/s)")
    print(f"C++ Engine (AVX2/OMP):  {cpp_mean_ms:6.2f} ms/token ({cpp_tok_per_sec:5.1f} tok/s)")
    print(f"Speedup Factor:         {speedup:6.2f}x faster")

    # ---------------------------------------------------------
    # 2. KV-Cache Rollback Overhead Comparison
    # ---------------------------------------------------------
    print("\n--- 2. KV-Cache Rollback Overhead (Leviathan Rejection Step) ---")
    TRIALS = 5000

    # PyTorch cache slicing
    # Setup cache with prompt + 5 tokens
    py_test_cache = None
    for tok in prompt + [1, 2, 3, 4, 5]:
        inp = torch.tensor([[tok]], dtype=torch.long, device=device)
        with torch.no_grad():
            _, _, py_test_cache = model(inp, kv_caches=py_test_cache, use_cache=True, start_pos=KVCacheManager.get_seq_len(py_test_cache))

    t0 = time.perf_counter()
    for _ in range(TRIALS):
        _ = KVCacheManager.truncate_cache(py_test_cache, prompt_len)
    t1 = time.perf_counter()
    py_rb_us = ((t1 - t0) / TRIALS) * 1e6

    # C++ Engine pointer rollback
    cpp_engine.reset_cache()
    for tok in prompt + [1, 2, 3, 4, 5]:
        cpp_engine.forward_token(tok, False)

    t0 = time.perf_counter()
    for _ in range(TRIALS):
        cpp_engine.rollback_cache(prompt_len)
    t1 = time.perf_counter()
    cpp_rb_ns = ((t1 - t0) / TRIALS) * 1e9

    print(f"PyTorch Cache Rollback (slicing):  {py_rb_us:8.3f} us ({py_rb_us*1000:8.1f} ns)")
    print(f"C++ Zero-Copy Rollback (pointer):  {cpp_rb_ns/1000:8.3f} us ({cpp_rb_ns:8.1f} ns)")
    print(f"Rollback Latency Reduction:        {py_rb_us / (cpp_rb_ns / 1000):8.1f}x reduction")

    # ---------------------------------------------------------
    # 3. Speculative Drafting Horizon Benchmarks
    # ---------------------------------------------------------
    print("\n--- 3. Speculative Cycles (Draft k Tokens + Rollback on Rejection) ---")
    print(f"{'Horizon k':<10} | {'PyTorch Draft (ms)':<20} | {'C++ Draft (ms)':<16} | {'C++ Rollback (us)':<18} | {'Draft Speedup':<14}")
    print("-" * 88)

    horizons = [1, 2, 3, 4, 5, 6, 8]
    summary_data = []

    for k in horizons:
        # Measure PyTorch draft cycle
        py_draft_times = []
        for _ in range(5):
            # Ingest prompt
            c_cache = None
            for tok in prompt:
                inp = torch.tensor([[tok]], dtype=torch.long, device=device)
                with torch.no_grad():
                    _, _, c_cache = model(inp, kv_caches=c_cache, use_cache=True, start_pos=KVCacheManager.get_seq_len(c_cache))
            c_tok = 198
            t0 = time.perf_counter()
            for _ in range(k):
                inp = torch.tensor([[c_tok]], dtype=torch.long, device=device)
                with torch.no_grad():
                    logits_step, _, c_cache = model(
                        inp, kv_caches=c_cache, use_cache=True, start_pos=KVCacheManager.get_seq_len(c_cache)
                    )
                    c_tok = int(torch.argmax(logits_step[0, -1]).item())
            t1 = time.perf_counter()
            py_draft_times.append((t1 - t0) * 1000.0)

        py_k_ms = sum(py_draft_times) / len(py_draft_times)

        # Measure C++ draft cycle via C++ internal timer
        cpp_res = cpp_engine.benchmark_draft_cycle(prompt, k, 10)
        cpp_k_ms = cpp_res["mean_draft_ms"]
        cpp_rb_us = cpp_res["mean_rollback_us"]
        k_speedup = py_k_ms / cpp_k_ms

        summary_data.append((k, py_k_ms, cpp_k_ms, cpp_rb_us, k_speedup))
        print(f"{k:<10} | {py_k_ms:<20.2f} | {cpp_k_ms:<16.2f} | {cpp_rb_us:<18.4f} | {k_speedup:<14.2f}x")

    print("\n================================================================================")
    print("Benchmark completed successfully.")


if __name__ == "__main__":
    benchmark_systems()
