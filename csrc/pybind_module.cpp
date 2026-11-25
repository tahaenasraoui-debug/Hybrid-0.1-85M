#include <torch/extension.h>
#include "engine.h"
#include <memory>
#include <chrono>

class PyHybridEngine {
private:
    std::unique_ptr<HybridInferenceEngine> engine;

public:
    PyHybridEngine() {
        engine = std::make_unique<HybridInferenceEngine>();
    }

    bool load_model(const std::string& bin_path) {
        return engine->load_from_binary(bin_path);
    }

    void reset_cache() {
        engine->reset_cache();
    }

    void rollback_cache(int target_len) {
        engine->rollback_cache(target_len);
    }

    int get_seq_len() const {
        return engine->get_seq_len();
    }

    int forward_token(int token_id, bool compute_full_logits = false) {
        return engine->forward_token(token_id, compute_full_logits);
    }

    torch::Tensor get_logits() {
        if (!engine->loaded) {
            throw std::runtime_error("Model not loaded");
        }
        int vocab_size = engine->header.vocab_size;
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        torch::Tensor t = torch::empty({vocab_size}, options);
        std::memcpy(t.data_ptr<float>(), engine->get_logits(), vocab_size * sizeof(float));
        return t;
    }

    std::vector<int> speculate_k(const std::vector<int>& prefix_tokens, int k) {
        return engine->speculate_k(prefix_tokens, k);
    }

    // MLSys Microbenchmark helper
    py::dict benchmark_draft_cycle(const std::vector<int>& prompt, int k, int num_trials) {
        engine->reset_cache();
        int cur_tok = -1;
        for (int tok : prompt) {
            cur_tok = engine->forward_token(tok, false);
        }
        int base_seq = engine->get_seq_len();

        double total_draft_us = 0.0;
        double total_rollback_ns = 0.0;

        for (int trial = 0; trial < num_trials; ++trial) {
            int t = cur_tok;
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int step = 0; step < k; ++step) {
                t = engine->forward_token(t, false);
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            total_draft_us += std::chrono::duration<double, std::micro>(t1 - t0).count();

            auto tr0 = std::chrono::high_resolution_clock::now();
            engine->rollback_cache(base_seq);
            auto tr1 = std::chrono::high_resolution_clock::now();
            total_rollback_ns += std::chrono::duration<double, std::nano>(tr1 - tr0).count();
        }

        double mean_draft_ms = (total_draft_us / num_trials) / 1000.0;
        double mean_rollback_ns = total_rollback_ns / num_trials;

        py::dict res;
        res["horizon_k"] = k;
        res["mean_draft_ms"] = mean_draft_ms;
        res["latency_per_token_ms"] = mean_draft_ms / k;
        res["throughput_tok_per_sec"] = 1000.0 / (mean_draft_ms / k);
        res["mean_rollback_ns"] = mean_rollback_ns;
        res["mean_rollback_us"] = mean_rollback_ns / 1000.0;
        return res;
    }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<PyHybridEngine>(m, "HybridCPPEngine")
        .def(py::init<>())
        .def("load_model", &PyHybridEngine::load_model)
        .def("reset_cache", &PyHybridEngine::reset_cache)
        .def("rollback_cache", &PyHybridEngine::rollback_cache)
        .def("get_seq_len", &PyHybridEngine::get_seq_len)
        .def("forward_token", &PyHybridEngine::forward_token, 
             py::arg("token_id"), py::arg("compute_full_logits") = false)
        .def("get_logits", &PyHybridEngine::get_logits)
        .def("speculate_k", &PyHybridEngine::speculate_k)
        .def("benchmark_draft_cycle", &PyHybridEngine::benchmark_draft_cycle);
}
