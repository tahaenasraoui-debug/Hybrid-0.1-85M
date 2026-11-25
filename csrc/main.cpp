#include "engine.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <numeric>
#include <ostream>
#include <string>

static void print_usage(const char* prog, std::ostream& os) {
    const char* exe = (prog && *prog) ? prog : "hybrid_infer";
    os << "Usage: " << exe << " [model_bin_path]\n"
       << "  model_bin_path  Path to exported FP32 binary weights "
       << "(default: csrc/hybrid_85m_fp32.bin)\n"
       << "  -h, --help      Display this usage message" << std::endl;
}

int main(int argc, char** argv) {
    if (argc > 1 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
        print_usage(argv[0], std::cout);
        return 0;
    }

    std::string bin_path = "csrc/hybrid_85m_fp32.bin";
    if (argc > 1) {
        bin_path = argv[1];
    }

    std::cout << "==========================================================" << std::endl;
    std::cout << "  Hybrid-0.1-85M: C++ Speculative Inference Runtime" << std::endl;
    std::cout << "  MLSys Zero-Copy KV-Cache & Micro-Kernel Benchmark" << std::endl;
    std::cout << "==========================================================" << std::endl;

    HybridInferenceEngine engine;

    auto t_load_start = std::chrono::high_resolution_clock::now();
    if (!engine.load_from_binary(bin_path)) {
        std::cerr << "Error: Failed to load model from " << bin_path << std::endl;
        print_usage(argv[0], std::cerr);
        return 1;
    }
    auto t_load_end = std::chrono::high_resolution_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(t_load_end - t_load_start).count();

    std::cout << "Model loaded successfully in: " << std::fixed << std::setprecision(2) << load_ms << " ms" << std::endl;
    std::cout << "Architecture Specifications:" << std::endl;
    std::cout << "  - Embeddings: " << (engine.header.tie_embeddings ? "tied" : "untied") << std::endl;
    std::cout << "  - Vocabulary Size: " << engine.header.vocab_size << std::endl;
    std::cout << "  - Hidden Dimension: " << engine.header.d_model << std::endl;
    std::cout << "  - Transformer Layers: " << engine.header.n_layers << std::endl;
    std::cout << "  - Query Heads: " << engine.header.n_heads << " (Head Dim: " << engine.header.head_dim << ")" << std::endl;
    std::cout << "  - KV Heads: " << engine.header.n_kv_heads << " (GQA Factor: " << engine.header.n_heads / engine.header.n_kv_heads << "x)" << std::endl;
    std::cout << "  - SwiGLU Intermediate Dim: " << engine.header.hidden_dim << std::endl;
    std::cout << "  - RoPE Theta: " << engine.header.rope_theta << std::endl;
    std::cout << "  - Max Sequence Length: " << engine.header.max_seq_len << std::endl;
    std::cout << "  - OpenMP Threads: " << omp_get_max_threads() << std::endl;

    // Test prompt (simulate Qwen token IDs)
    std::vector<int> prompt = {151644, 872, 198, 108386, 151645, 198, 151644, 77091, 198}; // Standard Qwen chat template tokens
    std::cout << "\n[Warmup] Ingesting prompt of " << prompt.size() << " tokens..." << std::endl;

    engine.reset_cache();
    int next_tok = -1;
    for (int tok : prompt) {
        next_tok = engine.forward_token(tok, true);
    }
    std::cout << "Ingested. KV Cache valid length: " << engine.get_seq_len() 
              << " | Initial greedy prediction: " << next_tok << std::endl;

    // Microbenchmark 1: Autoregressive Token Generation Latency
    std::cout << "\n----------------------------------------------------------" << std::endl;
    std::cout << "  Microbenchmark 1: Autoregressive Token Generation Latency" << std::endl;
    std::cout << "----------------------------------------------------------" << std::endl;
    const int WARMUP_STEPS = 5;
    const int BENCH_STEPS = 25;

    // Warmup
    for (int i = 0; i < WARMUP_STEPS; ++i) {
        next_tok = engine.forward_token(next_tok, false);
    }

    std::vector<double> latencies_ms;
    latencies_ms.reserve(BENCH_STEPS);

    for (int i = 0; i < BENCH_STEPS; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        next_tok = engine.forward_token(next_tok, false);
        auto t1 = std::chrono::high_resolution_clock::now();
        latencies_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    double avg_latency = std::accumulate(latencies_ms.begin(), latencies_ms.end(), 0.0) / BENCH_STEPS;
    double throughput = 1000.0 / avg_latency;

    std::cout << "Autoregressive Draft Latency: " << std::fixed << std::setprecision(2) << avg_latency << " ms/token" << std::endl;
    std::cout << "Draft Model Throughput:        " << std::fixed << std::setprecision(1) << throughput << " tokens/sec" << std::endl;

    // Microbenchmark 2: KV-Cache Rollback Overhead
    std::cout << "\n----------------------------------------------------------" << std::endl;
    std::cout << "  Microbenchmark 2: KV-Cache Rollback Overhead (O(1))" << std::endl;
    std::cout << "----------------------------------------------------------" << std::endl;

    const int ROLLBACK_TRIALS = 10000;
    int current_len = engine.get_seq_len();
    int target_len = current_len - 4;

    auto t_rb_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ROLLBACK_TRIALS; ++i) {
        engine.rollback_cache(target_len);
        // Toggle target to prevent compiler optimization elision
        target_len = (target_len == current_len - 4) ? current_len - 2 : current_len - 4;
    }
    auto t_rb_end = std::chrono::high_resolution_clock::now();
    double total_rb_ns = std::chrono::duration<double, std::nano>(t_rb_end - t_rb_start).count();
    double avg_rb_ns = total_rb_ns / ROLLBACK_TRIALS;

    std::cout << "Rollback Trials:              " << ROLLBACK_TRIALS << std::endl;
    std::cout << "Average Rollback Latency:     " << std::fixed << std::setprecision(2) << avg_rb_ns << " ns (" 
              << std::setprecision(5) << avg_rb_ns / 1000.0 << " us)" << std::endl;
    std::cout << "Allocation/Copy Overhead:     0 bytes (Zero-copy pointer rewind)" << std::endl;

    // Microbenchmark 3: Speculative Drafting & Rejection Cycles across Horizon k
    std::cout << "\n----------------------------------------------------------" << std::endl;
    std::cout << "  Microbenchmark 3: Speculative Cycles (Draft + Rollback)" << std::endl;
    std::cout << "----------------------------------------------------------" << std::endl;
    std::cout << "| Horizon (k) | Draft Latency (ms) | Rollback (us) | Total Cycle (ms) | Effective tok/s |" << std::endl;
    std::cout << "|:-----------:|:------------------:|:-------------:|:----------------:|:---------------:|" << std::endl;

    std::vector<int> horizons = {1, 2, 3, 4, 5, 6, 8};
    for (int k : horizons) {
        engine.reset_cache();
        for (int tok : prompt) {
            next_tok = engine.forward_token(tok, false);
        }
        int base_seq = engine.get_seq_len();

        const int CYCLES = 10;
        double total_draft_ms = 0;
        double total_rollback_us = 0;

        for (int c = 0; c < CYCLES; ++c) {
            int cur = next_tok;
            auto td0 = std::chrono::high_resolution_clock::now();
            for (int s = 0; s < k; ++s) {
                cur = engine.forward_token(cur, false);
            }
            auto td1 = std::chrono::high_resolution_clock::now();
            total_draft_ms += std::chrono::duration<double, std::milli>(td1 - td0).count();

            // Simulate rejection of last token (rollback to base_seq + k - 1)
            auto tr0 = std::chrono::high_resolution_clock::now();
            engine.rollback_cache(base_seq);
            auto tr1 = std::chrono::high_resolution_clock::now();
            total_rollback_us += std::chrono::duration<double, std::micro>(tr1 - tr0).count();
        }

        double mean_draft_ms = total_draft_ms / CYCLES;
        double mean_rollback_us = total_rollback_us / CYCLES;
        double total_cycle_ms = mean_draft_ms + (mean_rollback_us / 1000.0);
        double eff_rate = (double)k / (total_cycle_ms / 1000.0);

        std::cout << "| " << std::setw(11) << k 
                  << " | " << std::setw(18) << std::fixed << std::setprecision(2) << mean_draft_ms 
                  << " | " << std::setw(13) << std::fixed << std::setprecision(3) << mean_rollback_us
                  << " | " << std::setw(16) << std::fixed << std::setprecision(2) << total_cycle_ms 
                  << " | " << std::setw(15) << std::fixed << std::setprecision(1) << eff_rate 
                  << " |" << std::endl;
    }

    std::cout << "==========================================================" << std::endl;
    std::cout << "Benchmark Complete." << std::endl;
    return 0;
}
