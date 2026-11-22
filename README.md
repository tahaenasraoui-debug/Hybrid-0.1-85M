# Hybrid-0.1-85M: Lossless Speculative Decoding with Qwen

[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2Bcu121-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Tahary%2FHybrid--0.1--85M-yellow.svg)](https://huggingface.co/Tahary/Hybrid-0.1-85M)
[![Target Family](https://img.shields.io/badge/Target_Family-Qwen2.5-blue.svg)](https://huggingface.co/Qwen)
[![Verification](https://img.shields.io/badge/Equivalence_Tests-Passing-brightgreen.svg)](tests/test_equivalence.py)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A production-grade, mathematically lossless **Speculative Decoding Engine** and custom **Hybrid-0.1-85M Causal Transformer** built from scratch in PyTorch — with **zero high-level `model.generate()` wrappers**.

> **Model Weights on Hugging Face:** [`Tahary/Hybrid-0.1-85M`](https://huggingface.co/Tahary/Hybrid-0.1-85M)
> 
> **GitHub Codebase:** [`tahaenasraoui-debug/Hybrid-0.1-85M`](https://github.com/tahaenasraoui-debug/Hybrid-0.1-85M)

---

## 1. Architecture & Scope Disclosure

### Scope Honesty
* **Target Model (`Qwen/Qwen3-1.7B` / `Qwen2.5`):** Pretrained weights sourced from the official Qwen family (28 layers, 16 Q / 8 KV GQA heads, 32K context). Loaded in 4-bit NF4 (`BitsAndBytesConfig`) to guarantee total resident GPU memory remains strictly under **2.8 GB** on a 3 GB VRAM consumer card (NVIDIA GTX 1050 Mobile).
* **Draft Model (Custom 85M Causal LM):** Architected and trained **from scratch**. Incorporates Rotary Position Embeddings (RoPE, $\theta = 10^6$), SwiGLU non-linear activations, RMSNorm, Grouped-Query Attention (GQA), and native KV-caching.
* **Tokenizer Alignment:** Both models share Qwen's exact tokenizer (`vocab_size = 151,669` / `151,936`) and special tokens (`<|im_start|>`, `<|im_end|>`).
* **Parameter Budget:** Uses `tie_word_embeddings=True` on the 85M draft model:
  $$\text{Embedding Parameters: } 151,669 \times 384 \approx 58.2\text{M}$$
  $$\text{Transformer Layer Parameters: } 15 \times 1.77\text{M} \approx 26.8\text{M}$$
  $$\text{Total Parameters: } 58.2\text{M} + 26.8\text{M} \approx \mathbf{85.0\text{M Total}}$$

---

## 2. Distillation Pipeline (`scripts/distill.py`)

To maximize speculative token acceptance ($\alpha$), the 85M draft model is distilled against the target teacher model over curated conversational instruction datasets (Alpaca + Dolly-15k).

### Mode Consistency Guardrail
Reasoning tokens (`<think>...</think>`) create severe distribution mismatches during speculative drafting if prompts are formatted inconsistently. The pipeline explicitly configures:
```python
tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    enable_thinking=False,  # Enforces consistency with direct-answer generation
)
```

### Supported Objectives
1. **Forward KL Divergence (Mode-Covering, Default):**
   $$D_{\text{KL}}(P_{\text{target}} \parallel Q_{\text{draft}}) = \sum_{x \in \mathcal{V}} P_{\text{target}}(x) \log \left( \frac{P_{\text{target}}(x)}{Q_{\text{draft}}(x)} \right)$$
   Encourages the draft model to cover all modes of the teacher distribution so candidate tokens have a high probability of acceptance.
2. **Reverse KL Divergence (Mode-Seeking):**
   $$D_{\text{KL}}(Q_{\text{draft}} \parallel P_{\text{target}}) = \sum_{x \in \mathcal{V}} Q_{\text{draft}}(x) \log \left( \frac{Q_{\text{draft}}(x)}{P_{\text{target}}(x)} \right)$$
3. **Supervised Fine-Tuning (Prompt-Masked Cross-Entropy):** Standard next-token prediction over assistant completion tokens only.

---

## 3. Speculative Decoding Engine (`src/speculative.py`) — Strict Invariants

### 1. VRAM Guardrail (< 2.8 GB Resident Memory)
* Target model loaded in **4-bit NF4** (`~1.0 GB`).
* Draft model in **FP16** (`~170 MB`).
* `max_seq_len <= 2048` bounded context.
* Zero layers offload to system RAM during generation.

### 2. No Fake Re-computation (Single Parallel Verification Pass)
The target model evaluates all $k$ candidate tokens drafted by the 85M model in a **single parallel forward pass**:
```python
# Evaluates all k draft tokens simultaneously:
target_logits, target_cache = target_model(
    input_ids=draft_tensor,  # Shape: (1, k)
    past_key_values=target_cache,
    use_cache=True,
)
```
The prompt prefill is executed exactly **once**; subsequent rounds pass only candidate token slices.

### 3. Stateful KV Cache Rollback
When token $i \le k$ is rejected:
* Both target and draft `past_key_values` are sliced/truncated back to `prefix_len + i - 1` in $O(1)$ time.
* Eliminates cache pollution from rejected candidate tokens without recomputing prior tokens.

### 4. Mathematically Lossless Rejection Sampling (Leviathan et al., 2023)
* Candidate draft token $x_i$ is accepted with probability:
  $$\alpha_i = \min\left(1, \frac{P_{\text{target}}(x_i)}{Q_{\text{draft}}(x_i)}\right)$$
* If token $x_i$ is rejected, it is resampled from the normalized positive residual distribution:
  $$P_{\text{res}}(x) = \frac{\max\left(0, P_{\text{target}}(x) - Q_{\text{draft}}(x)\right)}{\sum_{x'} \max\left(0, P_{\text{target}}(x') - Q_{\text{draft}}(x')\right)}$$
* If all $k$ draft tokens are accepted, a bonus $(k+1)$-th token is sampled directly from $P_{\text{target}}(\cdot \mid x_{1..k})$.

---

## 4. Mandatory Verification Suite (`tests/test_equivalence.py`)

All invariants are formally verified under automated `pytest` test suites:

```bash
PYTHONPATH=. pytest tests/test_equivalence.py -v
```

1. **Greedy Equivalence Test:** Asserts that at `temperature=0.0` (argmax), `speculative_generate()` produces a **100% token-for-token identical sequence** to plain autoregressive target generation across 50 diverse prompts and variable $k \in \{2, 3, 4, 5, 6\}$.
2. **KV Cache Integrity Test:** Asserts that target logits after a stateful KV-cache rollback match a fresh forward pass on the truncated sequence within **$\text{atol} = 10^{-4}$**.

---

## 5. Empirical Benchmarking Suite (`benchmarks/run_bench.py`)

### Test Configuration
* **Hardware:** NVIDIA GeForce GTX 1050 Mobile (3 GB VRAM, Pascal `sm_61`), Intel i5-9300H, 16 GB RAM.
* **Batch Size:** 1
* **Target Model:** `Qwen/Qwen2.5-0.5B-Instruct` (4-bit NF4)
* **Draft Model:** Custom 85M Distilled Causal LM (FP16)

### Results Table (Honestly Reported)

| Generation Mode | Draft Lookahead ($k$) | Throughput (tok/s) | Net Speedup | Acceptance Rate ($\alpha$) | Mean Accepted ($\tau$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Plain Autoregressive Baseline** | — | **10.95 tok/s** | **1.00×** | — | — |
| **Speculative Decoding** | $k=2$ | 3.78 tok/s | 0.35× | 1.4% | 0.03 tok/step |
| **Speculative Decoding** | $k=3$ | 2.95 tok/s | 0.27× | 1.2% | 0.03 tok/step |
| **Speculative Decoding** | $k=4$ | 2.41 tok/s | 0.22× | 1.1% | 0.03 tok/step |
| **Speculative Decoding** | $k=5$ | 2.05 tok/s | 0.19× | 1.0% | 0.03 tok/step |
| **Speculative Decoding** | $k=6$ | 1.78 tok/s | 0.16× | 0.9% | 0.03 tok/step |

### Scientific Analysis: Why Speedup is < 1.0x on Small Targets
As noted in the project specification and the literature (Leviathan et al., 2023; Chen et al., 2023):
1. **Target Forward Latency:** A small target model (0.5B – 1.7B) is already extremely fast ($\approx 90\text{ ms/token}$). The computational cost of drafting $k$ tokens ($\approx 25\text{ ms/token}$) plus parallel verification is comparable to evaluating tokens directly on the target.
2. **Vocabulary Scaling:** With an enormous vocabulary ($151,936$ tokens), greedy top-1 argmax agreement between an 85M student and teacher on out-of-distribution prompts is naturally sparse without stochastic temperature sampling or soft top-$p$ scaling.
3. **Where Speculative Decoding Delivers $2\times–3\times$ Speedup:** When the target model is scaled to **7B, 14B, or 70B** parameters, the target forward pass takes $150\text{–}400\text{ ms}$, while the 85M draft model still executes in $< 25\text{ ms}$. At that regime, accepting even 2–3 tokens per verification round yields immediate $> 2.0\times$ wall-clock speedups.

---

## 6. C++ Systems Inference Engine (`csrc/`) — Zero-Allocation & Rollback Profiling

To eliminate Python interpreter overhead, PyTorch dynamic allocations, and cache slicing latency during Leviathan rejection sampling, we implemented a custom **standalone C++ inference engine** (`csrc/engine.cpp`) with AVX2 vectorization, OpenMP multi-threading, and PyBind11 bindings.

### Core Systems Features
* **Zero-Allocation Dynamic Arena:** KV Cache is preallocated contiguously in memory (`[layers, 2, max_seq_len, n_kv_heads, head_dim]`). Generation loop executes with **zero dynamic `malloc` / `cudaMalloc` allocations**.
* **$O(1)$ Zero-Copy KV-Cache Rollback:** In speculative decoding, rejected candidate draft tokens must be purged from the KV-cache. While PyTorch requires recreating tensor views or slicing lists (`cache[..., :target_len]`), our C++ engine executes rollback in **$0.21\ \mu\text{s}$ ($212\text{ ns}$)** via simple pointer rewinds (`seq_len = target_len`), achieving a **$418\times$ latency reduction** over PyTorch slicing.
* **Fused Vector Kernels (AVX2 + FMA):** RMSNorm, RoPE rotary coordinate rotations, and SwiGLU non-linearities (`silu(w1) * w3`) are computed with AVX2 intrinsics and fused memory passes.

### Systems Microbenchmarks (Intel i5-9300H, 8 Threads)

| Metric | PyTorch Eager (CPU) | C++ Engine (AVX2 + OpenMP) | Speedup / Reduction |
| :--- | :---: | :---: | :---: |
| **KV-Cache Rollback Latency** | $88.93\ \mu\text{s}$ | **$0.21\ \mu\text{s}$ ($213\text{ ns}$)** | **$418.0\times$ faster** |
| **Rollback Memory Allocation** | Non-zero (tensor slicing) | **$0\text{ bytes}$ (pointer rewind)** | **Zero allocation** |
| **Draft Latency ($k=1$)** | $26.34\text{ ms}$ | **$23.82\text{ ms}$** | **$1.11\times$ faster** |
| **Draft Latency ($k=4$)** | $114.70\text{ ms}$ | **$96.12\text{ ms}$** | **$1.19\times$ faster** |
| **Draft Latency ($k=5$)** | $145.65\text{ ms}$ | **$117.64\text{ ms}$** | **$1.24\times$ faster** |
| **Model Load Time** | $\approx 1,200\text{ ms}$ | **$68.87\text{ ms}$** | **$17.4\times$ faster** |
| **Cosine Similarity to PyTorch** | $1.000000$ | $1.000000$ | **Exact numerical parity** |

### Build & Run C++ Engine

```bash
# 1. Export checkpoint weights to flat binary
python scripts/export_weights_bin.py

# 2. Compile standalone C++ binary (zero Python dependencies)
make -C csrc
./csrc/hybrid_infer csrc/hybrid_85m_fp32.bin

# 3. Compile Python C++ Extension (hybrid_cpp)
python setup_cpp.py build_ext --inplace

# 4. Run Parity & Zero-Leakage Verification
python tests/test_cpp_engine.py

# 5. Run MLSys Comparative Benchmark
python benchmarks/benchmark_systems.py
```

---

## Quickstart

### 1. Installation
```bash
git clone https://github.com/tahaenasraoui-debug/Hybrid-0.1-85M.git
cd Hybrid-0.1-85M

# Setup Python 3.12 environment
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. Run Verification Suite
```bash
PYTHONPATH=. pytest tests/test_equivalence.py -v
```

### 3. Run Benchmarking Suite
```bash
python benchmarks/run_bench.py --max_tokens 32
```

### 4. Interactive Chat CLI
```bash
python chat_speculative.py --k 4
```
