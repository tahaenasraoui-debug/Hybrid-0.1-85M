#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#ifdef _WIN32
#include <malloc.h>
#endif

#if (defined(__AVX2__) && defined(__FMA__)) || (defined(_MSC_VER) && defined(__AVX2__))
#include <immintrin.h>
#define HYBRID_USE_AVX2 1
#endif

#ifdef _OPENMP
#include <omp.h>
#ifdef _MSC_VER
#define HYBRID_OMP_PRAGMA(x) __pragma(x)
#else
#define HYBRID_OMP_PRAGMA(x) _Pragma(#x)
#endif
#else
#define HYBRID_OMP_PRAGMA(x)
inline int omp_get_max_threads() { return 1; }
#endif

#if defined(_OPENMP) && _OPENMP >= 201307
#define HYBRID_OMP_SIMD() HYBRID_OMP_PRAGMA(omp simd)
#else
#define HYBRID_OMP_SIMD()
#endif

inline float* alloc_aligned_floats(size_t num_floats) {
    if (num_floats > (std::numeric_limits<size_t>::max() - 63) / sizeof(float)) {
        return nullptr;
    }
    size_t raw_bytes = num_floats * sizeof(float);
    if (raw_bytes == 0) {
        raw_bytes = 64;
    }
    size_t aligned_bytes = ((raw_bytes + 63) / 64) * 64;
#ifdef _WIN32
    void* ptr = _aligned_malloc(aligned_bytes, 64);
#else
    void* ptr = std::aligned_alloc(64, aligned_bytes);
#endif
    if (ptr) {
        std::memset(ptr, 0, aligned_bytes);
    }
    return static_cast<float*>(ptr);
}

inline void free_aligned_floats(void* ptr) {
    if (!ptr) {
        return;
    }
#ifdef _WIN32
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

struct ModelHeader {
    uint32_t magic;          // 0x31425948 ('HYB1')
    uint32_t vocab_size;     // 151936
    uint32_t d_model;        // 480
    uint32_t n_layers;       // 5
    uint32_t n_heads;        // 8
    uint32_t n_kv_heads;     // 2
    uint32_t head_dim;       // 60
    uint32_t hidden_dim;     // 1280
    float rope_theta;        // 1000000.0f
    float norm_eps;          // 1e-6f
    uint32_t max_seq_len;    // 2048
    uint32_t tie_embeddings; // 1 = tied, 0 = untied (lm_head follows final_norm)
};

struct LayerWeights {
    float* attn_norm = nullptr; // [d_model]
    float* q_proj = nullptr;    // [n_heads * head_dim, d_model] = [480, 480]
    float* k_proj = nullptr;    // [n_kv_heads * head_dim, d_model] = [120, 480]
    float* v_proj = nullptr;    // [n_kv_heads * head_dim, d_model] = [120, 480]
    float* o_proj = nullptr;    // [d_model, n_heads * head_dim] = [480, 480]
    float* ffn_norm = nullptr;  // [d_model]
    float* w1 = nullptr;        // [hidden_dim, d_model] = [1280, 480] (gate)
    float* w2 = nullptr;        // [d_model, hidden_dim] = [480, 1280] (down)
    float* w3 = nullptr;        // [hidden_dim, d_model] = [1280, 480] (up)
};

// Zero-allocation, contiguous linear KV Cache for fast rollback
struct KVCache {
    int max_seq_len = 0;
    int n_layers = 0;
    int n_kv_heads = 0;
    int head_dim = 0;
    int seq_len = 0; // Current sequence length

    // Flat memory layout: [n_layers, 2 (k,v), max_seq_len, n_kv_heads, head_dim]
    float* data = nullptr;

    bool init(int layers, int kv_heads, int h_dim, int max_len) {
        free_mem();
        n_layers = layers;
        n_kv_heads = kv_heads;
        head_dim = h_dim;
        max_seq_len = max_len;
        seq_len = 0;
        size_t total_floats = (size_t)n_layers * 2 * max_seq_len * n_kv_heads * head_dim;
        data = alloc_aligned_floats(total_floats);
        return data != nullptr;
    }

    void free_mem() {
        if (data) {
            free_aligned_floats(data);
            data = nullptr;
        }
        seq_len = 0;
    }

    inline float* get_k_ptr(int layer, int pos) {
        size_t layer_stride = (size_t)2 * max_seq_len * n_kv_heads * head_dim;
        return data + layer * layer_stride + (size_t)pos * n_kv_heads * head_dim;
    }

    inline float* get_v_ptr(int layer, int pos) {
        size_t layer_stride = (size_t)2 * max_seq_len * n_kv_heads * head_dim;
        size_t v_offset = (size_t)max_seq_len * n_kv_heads * head_dim;
        return data + layer * layer_stride + v_offset + (size_t)pos * n_kv_heads * head_dim;
    }

    // O(1) rollback for speculative decoding: zero copies, zero allocations
    inline void rollback(int target_len) {
        if (target_len >= 0 && target_len <= max_seq_len) {
            seq_len = target_len;
        }
    }
};

class HybridInferenceEngine {
public:
    ModelHeader header;
    std::vector<LayerWeights> layers;
    float* tok_embeddings; // [vocab_size, d_model]
    float* lm_head;        // [vocab_size, d_model] when tie_embeddings == 0
    float* final_norm;     // [d_model]

    // Precomputed RoPE tables: [max_seq_len, head_dim]
    float* rope_cos;
    float* rope_sin;

    // KV Cache
    KVCache kv_cache;

    // Preallocated activation scratch buffers
    float* act_x;
    float* act_norm_x;
    float* act_q;
    float* act_k;
    float* act_v;
    float* act_attn_out;
    float* act_o_out;
    float* act_w1;
    float* act_w3;
    float* act_ffn_out;
    float* act_logits;
    float* act_scores; // [max_seq_len]

    bool loaded;

    HybridInferenceEngine();
    ~HybridInferenceEngine();

    void free_buffers();
    bool load_from_binary(const std::string& bin_path);
    void reset_cache();
    void rollback_cache(int target_len);
    int get_seq_len() const { return kv_cache.seq_len; }

    // Forward pass for a single token; updates KV cache at current seq_len
    // If compute_full_logits is false, it only finds the argmax token rapidly
    int forward_token(int token_id, bool compute_full_logits = true);

    // Get logits for current token (size: vocab_size)
    const float* get_logits() const { return act_logits; }

    // Speculate k candidate tokens from prompt or current state
    std::vector<int> speculate_k(const std::vector<int>& prefix_tokens, int k);

    // Fast vector math helpers
    static inline float vec_dot(const float* a, const float* b, int n) {
#ifdef HYBRID_USE_AVX2
        __m256 sum256 = _mm256_setzero_ps();
        int i = 0;
        for (; i <= n - 8; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            sum256 = _mm256_fmadd_ps(va, vb, sum256);
        }
        alignas(32) float tmp[8];
        _mm256_store_ps(tmp, sum256);
        float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
        for (; i < n; ++i) {
            sum += a[i] * b[i];
        }
        return sum;
#else
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            sum += a[i] * b[i];
        }
        return sum;
#endif
    }

    static inline void gemv(float* y, const float* W, const float* x, int out_dim, int in_dim) {
        HYBRID_OMP_PRAGMA(omp parallel for schedule(static))
        for (int i = 0; i < out_dim; ++i) {
            y[i] = vec_dot(W + (size_t)i * in_dim, x, in_dim);
        }
    }

    static inline void rmsnorm(float* out, const float* x, const float* weight, int dim, float eps) {
#ifdef HYBRID_USE_AVX2
        __m256 sum256 = _mm256_setzero_ps();
        int i = 0;
        for (; i <= dim - 8; i += 8) {
            __m256 vx = _mm256_loadu_ps(x + i);
            sum256 = _mm256_fmadd_ps(vx, vx, sum256);
        }
        alignas(32) float tmp[8];
        _mm256_store_ps(tmp, sum256);
        float sum_sq = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
        for (; i < dim; ++i) {
            sum_sq += x[i] * x[i];
        }
        float scale = 1.0f / std::sqrt((sum_sq / (float)dim) + eps);

        __m256 vscale = _mm256_set1_ps(scale);
        for (i = 0; i <= dim - 8; i += 8) {
            __m256 vx = _mm256_loadu_ps(x + i);
            __m256 vw = _mm256_loadu_ps(weight + i);
            __m256 vo = _mm256_mul_ps(_mm256_mul_ps(vx, vscale), vw);
            _mm256_storeu_ps(out + i, vo);
        }
        for (; i < dim; ++i) {
            out[i] = x[i] * scale * weight[i];
        }
#else
        float sum_sq = 0.0f;
        for (int i = 0; i < dim; ++i) {
            sum_sq += x[i] * x[i];
        }
        float scale = 1.0f / std::sqrt((sum_sq / (float)dim) + eps);
        for (int i = 0; i < dim; ++i) {
            out[i] = x[i] * scale * weight[i];
        }
#endif
    }

    static inline float silu(float x) {
        return x / (1.0f + std::exp(-x));
    }
};
