#include "engine.h"
#include <fstream>
#include <iostream>
#include <limits>

namespace {

inline void free_and_null(float*& ptr) {
    if (ptr) {
        std::free(ptr);
        ptr = nullptr;
    }
}

float* alloc_and_read(std::ifstream& file, size_t num_floats) {
    float* ptr = alloc_aligned_floats(num_floats);
    if (!ptr) {
        return nullptr;
    }
    if (num_floats > static_cast<size_t>(std::numeric_limits<std::streamsize>::max()) / sizeof(float)) {
        std::free(ptr);
        return nullptr;
    }
    std::streamsize bytes = static_cast<std::streamsize>(num_floats * sizeof(float));
    if (!file.read(reinterpret_cast<char*>(ptr), bytes)) {
        std::free(ptr);
        return nullptr;
    }
    return ptr;
}

} // namespace

HybridInferenceEngine::HybridInferenceEngine()
    : header{},
      tok_embeddings(nullptr),
      lm_head(nullptr),
      final_norm(nullptr),
      rope_cos(nullptr),
      rope_sin(nullptr),
      act_x(nullptr),
      act_norm_x(nullptr),
      act_q(nullptr),
      act_k(nullptr),
      act_v(nullptr),
      act_attn_out(nullptr),
      act_o_out(nullptr),
      act_w1(nullptr),
      act_w3(nullptr),
      act_ffn_out(nullptr),
      act_logits(nullptr),
      act_scores(nullptr),
      loaded(false) {}

HybridInferenceEngine::~HybridInferenceEngine() {
    free_buffers();
}

void HybridInferenceEngine::free_buffers() {
    loaded = false;
    free_and_null(tok_embeddings);
    free_and_null(lm_head);
    free_and_null(final_norm);
    free_and_null(rope_cos);
    free_and_null(rope_sin);

    for (auto& l : layers) {
        free_and_null(l.attn_norm);
        free_and_null(l.q_proj);
        free_and_null(l.k_proj);
        free_and_null(l.v_proj);
        free_and_null(l.o_proj);
        free_and_null(l.ffn_norm);
        free_and_null(l.w1);
        free_and_null(l.w2);
        free_and_null(l.w3);
    }
    layers.clear();

    kv_cache.free_mem();

    free_and_null(act_x);
    free_and_null(act_norm_x);
    free_and_null(act_q);
    free_and_null(act_k);
    free_and_null(act_v);
    free_and_null(act_attn_out);
    free_and_null(act_o_out);
    free_and_null(act_w1);
    free_and_null(act_w3);
    free_and_null(act_ffn_out);
    free_and_null(act_logits);
    free_and_null(act_scores);
}

bool HybridInferenceEngine::load_from_binary(const std::string& bin_path) {
    free_buffers();

    std::ifstream file(bin_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open binary model file: " << bin_path << std::endl;
        return false;
    }

    if (!file.read(reinterpret_cast<char*>(&header), sizeof(ModelHeader))) {
        std::cerr << "Failed to read model header from: " << bin_path << std::endl;
        return false;
    }
    if (header.magic != 0x31425948) { // 'HYB1'
        std::cerr << "Invalid magic bytes in model header: " << std::hex << header.magic << std::dec << std::endl;
        return false;
    }
    if (header.vocab_size == 0 || header.d_model == 0 || header.n_layers == 0 ||
        header.n_heads == 0 || header.n_kv_heads == 0 || header.head_dim == 0 ||
        header.hidden_dim == 0 || header.max_seq_len == 0 ||
        (header.n_heads % header.n_kv_heads) != 0 || (header.head_dim % 2) != 0) {
        std::cerr << "Invalid model dimensions in header" << std::endl;
        return false;
    }

    // 1. Token embeddings
    size_t embed_floats = (size_t)header.vocab_size * header.d_model;
    tok_embeddings = alloc_and_read(file, embed_floats);
    if (!tok_embeddings) {
        std::cerr << "Failed to load tok_embeddings" << std::endl;
        free_buffers();
        return false;
    }

    // 2. Layers
    layers.resize(header.n_layers);
    for (uint32_t l = 0; l < header.n_layers; ++l) {
        LayerWeights& lw = layers[l];

        size_t q_size = (size_t)(header.n_heads * header.head_dim) * header.d_model;
        size_t kv_size = (size_t)(header.n_kv_heads * header.head_dim) * header.d_model;
        size_t o_size = (size_t)header.d_model * (header.n_heads * header.head_dim);
        size_t w1_size = (size_t)header.hidden_dim * header.d_model;
        size_t w2_size = (size_t)header.d_model * header.hidden_dim;
        size_t w3_size = (size_t)header.hidden_dim * header.d_model;

        lw.attn_norm = alloc_and_read(file, header.d_model);
        lw.q_proj = alloc_and_read(file, q_size);
        lw.k_proj = alloc_and_read(file, kv_size);
        lw.v_proj = alloc_and_read(file, kv_size);
        lw.o_proj = alloc_and_read(file, o_size);
        lw.ffn_norm = alloc_and_read(file, header.d_model);
        lw.w1 = alloc_and_read(file, w1_size);
        lw.w2 = alloc_and_read(file, w2_size);
        lw.w3 = alloc_and_read(file, w3_size);

        if (!lw.attn_norm || !lw.q_proj || !lw.k_proj || !lw.v_proj || !lw.o_proj ||
            !lw.ffn_norm || !lw.w1 || !lw.w2 || !lw.w3) {
            std::cerr << "Failed to load weights for layer " << l << std::endl;
            free_buffers();
            return false;
        }
    }

    // 3. Final norm
    final_norm = alloc_and_read(file, header.d_model);
    if (!final_norm) {
        std::cerr << "Failed to load final_norm" << std::endl;
        free_buffers();
        return false;
    }

    // 3b. Untied LM head (if tie_embeddings == 0)
    if (header.tie_embeddings == 0) {
        lm_head = alloc_and_read(file, embed_floats);
        if (!lm_head) {
            std::cerr << "Failed to load untied lm_head weights" << std::endl;
            free_buffers();
            return false;
        }
    }

    // 4. Precompute RoPE
    size_t rope_size = (size_t)header.max_seq_len * header.head_dim;
    rope_cos = alloc_aligned_floats(rope_size);
    rope_sin = alloc_aligned_floats(rope_size);
    if (!rope_cos || !rope_sin) {
        std::cerr << "Failed to allocate RoPE tables" << std::endl;
        free_buffers();
        return false;
    }

    int half_dim = (int)header.head_dim / 2;
    for (uint32_t pos = 0; pos < header.max_seq_len; ++pos) {
        for (int i = 0; i < half_dim; ++i) {
            float exponent = (float)(2 * i) / (float)header.head_dim;
            float freq = 1.0f / std::pow(header.rope_theta, exponent);
            float angle = (float)pos * freq;
            float c = std::cos(angle);
            float s = std::sin(angle);

            rope_cos[pos * header.head_dim + i] = c;
            rope_cos[pos * header.head_dim + i + half_dim] = c;
            rope_sin[pos * header.head_dim + i] = s;
            rope_sin[pos * header.head_dim + i + half_dim] = s;
        }
    }

    // 5. Initialize KV Cache
    if (!kv_cache.init((int)header.n_layers, (int)header.n_kv_heads, (int)header.head_dim, (int)header.max_seq_len)) {
        std::cerr << "Failed to allocate KV cache" << std::endl;
        free_buffers();
        return false;
    }

    // 6. Allocate scratch buffers
    act_x = alloc_aligned_floats(header.d_model);
    act_norm_x = alloc_aligned_floats(header.d_model);
    act_q = alloc_aligned_floats((size_t)header.n_heads * header.head_dim);
    act_k = alloc_aligned_floats((size_t)header.n_kv_heads * header.head_dim);
    act_v = alloc_aligned_floats((size_t)header.n_kv_heads * header.head_dim);
    act_attn_out = alloc_aligned_floats((size_t)header.n_heads * header.head_dim);
    act_o_out = alloc_aligned_floats(header.d_model);
    act_w1 = alloc_aligned_floats(header.hidden_dim);
    act_w3 = alloc_aligned_floats(header.hidden_dim);
    act_ffn_out = alloc_aligned_floats(header.d_model);
    act_logits = alloc_aligned_floats(header.vocab_size);
    act_scores = alloc_aligned_floats(header.max_seq_len);

    if (!act_x || !act_norm_x || !act_q || !act_k || !act_v || !act_attn_out ||
        !act_o_out || !act_w1 || !act_w3 || !act_ffn_out || !act_logits || !act_scores) {
        std::cerr << "Failed to allocate activation scratch buffers" << std::endl;
        free_buffers();
        return false;
    }

    loaded = true;
    return true;
}

void HybridInferenceEngine::reset_cache() {
    kv_cache.rollback(0);
}

void HybridInferenceEngine::rollback_cache(int target_len) {
    kv_cache.rollback(target_len);
}

int HybridInferenceEngine::forward_token(int token_id, bool compute_full_logits) {
    if (!loaded) return -1;
    if (token_id < 0 || token_id >= (int)header.vocab_size) return -1;
    if (kv_cache.seq_len >= (int)header.max_seq_len) return -1;

    int pos = kv_cache.seq_len;
    int d_model = header.d_model;
    int head_dim = header.head_dim;
    int n_heads = header.n_heads;
    int n_kv_heads = header.n_kv_heads;
    int hidden_dim = header.hidden_dim;

    // 1. Embedding lookup
    std::memcpy(act_x, tok_embeddings + (size_t)token_id * d_model, d_model * sizeof(float));

    // Pointer to RoPE cos/sin for current position
    const float* cur_cos = rope_cos + (size_t)pos * head_dim;
    const float* cur_sin = rope_sin + (size_t)pos * head_dim;

    int half_dim = head_dim / 2; // 30
    float inv_sqrt_head_dim = 1.0f / std::sqrt((float)head_dim);

    // 2. Transformer layers
    for (uint32_t l = 0; l < header.n_layers; ++l) {
        LayerWeights& lw = layers[l];

        // Attention RMSNorm
        rmsnorm(act_norm_x, act_x, lw.attn_norm, d_model, header.norm_eps);

        // Q, K, V Projections
        gemv(act_q, lw.q_proj, act_norm_x, n_heads * head_dim, d_model);
        gemv(act_k, lw.k_proj, act_norm_x, n_kv_heads * head_dim, d_model);
        gemv(act_v, lw.v_proj, act_norm_x, n_kv_heads * head_dim, d_model);

        // Apply RoPE to Query heads
        for (int h = 0; h < n_heads; ++h) {
            float* q_head = act_q + h * head_dim;
            for (int i = 0; i < half_dim; ++i) {
                float q0 = q_head[i];
                float q1 = q_head[i + half_dim];
                float c = cur_cos[i];
                float s = cur_sin[i];
                q_head[i] = q0 * c - q1 * s;
                q_head[i + half_dim] = q1 * c + q0 * s;
            }
        }

        // Apply RoPE to Key heads
        for (int kv_h = 0; kv_h < n_kv_heads; ++kv_h) {
            float* k_head = act_k + kv_h * head_dim;
            for (int i = 0; i < half_dim; ++i) {
                float k0 = k_head[i];
                float k1 = k_head[i + half_dim];
                float c = cur_cos[i];
                float s = cur_sin[i];
                k_head[i] = k0 * c - k1 * s;
                k_head[i + half_dim] = k1 * c + k0 * s;
            }
        }

        // Write K and V into KV Cache at pos
        float* k_dest = kv_cache.get_k_ptr(l, pos);
        float* v_dest = kv_cache.get_v_ptr(l, pos);
        std::memcpy(k_dest, act_k, n_kv_heads * head_dim * sizeof(float));
        std::memcpy(v_dest, act_v, n_kv_heads * head_dim * sizeof(float));

        // Grouped Query Attention (GQA)
        // 8 query heads share 2 KV heads (4 query heads per KV head)
        int kv_group_size = n_heads / n_kv_heads; // 4
        int total_kv_len = pos + 1;

        for (int h = 0; h < n_heads; ++h) {
            int kv_h = h / kv_group_size;
            float* q_head = act_q + h * head_dim;
            float* out_head = act_attn_out + h * head_dim;

            // Compute attention scores against all past cached keys
            float max_score = -std::numeric_limits<float>::infinity();
            for (int t = 0; t < total_kv_len; ++t) {
                const float* k_cached = kv_cache.get_k_ptr(l, t) + kv_h * head_dim;
                float score = vec_dot(q_head, k_cached, head_dim) * inv_sqrt_head_dim;
                act_scores[t] = score;
                if (score > max_score) max_score = score;
            }

            // Stable Softmax
            float sum_exp = 0.0f;
            for (int t = 0; t < total_kv_len; ++t) {
                float exp_val = std::exp(act_scores[t] - max_score);
                act_scores[t] = exp_val;
                sum_exp += exp_val;
            }
            float inv_sum = 1.0f / sum_exp;
            for (int t = 0; t < total_kv_len; ++t) {
                act_scores[t] *= inv_sum;
            }

            // Weighted sum of cached values
            std::memset(out_head, 0, head_dim * sizeof(float));
            for (int t = 0; t < total_kv_len; ++t) {
                const float* v_cached = kv_cache.get_v_ptr(l, t) + kv_h * head_dim;
                float weight = act_scores[t];
                HYBRID_OMP_PRAGMA(omp simd)
                for (int d = 0; d < head_dim; ++d) {
                    out_head[d] += weight * v_cached[d];
                }
            }
        }

        // Output projection
        gemv(act_o_out, lw.o_proj, act_attn_out, d_model, n_heads * head_dim);

        // Residual connection
        HYBRID_OMP_PRAGMA(omp simd)
        for (int i = 0; i < d_model; ++i) {
            act_x[i] += act_o_out[i];
        }

        // FFN RMSNorm
        rmsnorm(act_norm_x, act_x, lw.ffn_norm, d_model, header.norm_eps);

        // SwiGLU: w1 (gate) & w3 (up)
        gemv(act_w1, lw.w1, act_norm_x, hidden_dim, d_model);
        gemv(act_w3, lw.w3, act_norm_x, hidden_dim, d_model);

        // Fused SiLU * up
        HYBRID_OMP_PRAGMA(omp simd)
        for (int i = 0; i < hidden_dim; ++i) {
            act_w1[i] = silu(act_w1[i]) * act_w3[i];
        }

        // Down projection
        gemv(act_ffn_out, lw.w2, act_w1, d_model, hidden_dim);

        // Residual connection
        HYBRID_OMP_PRAGMA(omp simd)
        for (int i = 0; i < d_model; ++i) {
            act_x[i] += act_ffn_out[i];
        }
    }

    // Increment KV sequence length
    kv_cache.seq_len = pos + 1;

    // Final RMSNorm
    rmsnorm(act_norm_x, act_x, final_norm, d_model, header.norm_eps);

    const float* out_weights = (header.tie_embeddings != 0 || lm_head == nullptr) ? tok_embeddings : lm_head;
    int vocab = static_cast<int>(header.vocab_size);
    int best_token = 0;
    if (compute_full_logits) {
        gemv(act_logits, out_weights, act_norm_x, vocab, d_model);

        float max_logit = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < vocab; ++i) {
            if (act_logits[i] > max_logit) {
                max_logit = act_logits[i];
                best_token = i;
            }
        }
    } else {
        // Fast parallel argmax without storing full vocabulary logits
        float global_max = -std::numeric_limits<float>::infinity();
        int global_best = 0;

        HYBRID_OMP_PRAGMA(omp parallel)
        {
            float local_max = -std::numeric_limits<float>::infinity();
            int local_best = 0;

            HYBRID_OMP_PRAGMA(omp for schedule(static))
            for (int i = 0; i < vocab; ++i) {
                float val = vec_dot(out_weights + (size_t)i * d_model, act_norm_x, d_model);
                if (val > local_max) {
                    local_max = val;
                    local_best = i;
                }
            }

            HYBRID_OMP_PRAGMA(omp critical)
            {
                if (local_max > global_max) {
                    global_max = local_max;
                    global_best = local_best;
                }
            }
        }
        best_token = global_best;
    }

    return best_token;
}

std::vector<int> HybridInferenceEngine::speculate_k(const std::vector<int>& prefix_tokens, int k) {
    std::vector<int> drafted;
    if (k <= 0) return drafted;

    // Feed prefix prompt
    int next_tok = -1;
    for (int tok : prefix_tokens) {
        next_tok = forward_token(tok, false);
    }

    if (next_tok < 0) return drafted;
    drafted.push_back(next_tok);

    // Generate remaining k-1 speculative tokens
    for (int step = 1; step < k; ++step) {
        next_tok = forward_token(next_tok, false);
        if (next_tok < 0) break;
        drafted.push_back(next_tok);
    }

    return drafted;
}
