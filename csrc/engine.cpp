#include "engine.h"
#include <fstream>
#include <iostream>
#include <chrono>
#include <stdexcept>
#include <limits>

HybridInferenceEngine::HybridInferenceEngine()
    : tok_embeddings(nullptr),
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
    if (tok_embeddings) free(tok_embeddings);
    if (final_norm) free(final_norm);
    if (rope_cos) free(rope_cos);
    if (rope_sin) free(rope_sin);

    for (auto& l : layers) {
        if (l.attn_norm) free(l.attn_norm);
        if (l.q_proj) free(l.q_proj);
        if (l.k_proj) free(l.k_proj);
        if (l.v_proj) free(l.v_proj);
        if (l.o_proj) free(l.o_proj);
        if (l.ffn_norm) free(l.ffn_norm);
        if (l.w1) free(l.w1);
        if (l.w2) free(l.w2);
        if (l.w3) free(l.w3);
    }

    kv_cache.free_mem();

    if (act_x) free(act_x);
    if (act_norm_x) free(act_norm_x);
    if (act_q) free(act_q);
    if (act_k) free(act_k);
    if (act_v) free(act_v);
    if (act_attn_out) free(act_attn_out);
    if (act_o_out) free(act_o_out);
    if (act_w1) free(act_w1);
    if (act_w3) free(act_w3);
    if (act_ffn_out) free(act_ffn_out);
    if (act_logits) free(act_logits);
    if (act_scores) free(act_scores);
}

bool HybridInferenceEngine::load_from_binary(const std::string& bin_path) {
    std::ifstream file(bin_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open binary model file: " << bin_path << std::endl;
        return false;
    }

    file.read((char*)&header, sizeof(ModelHeader));
    if (header.magic != 0x31425948) { // 'HYB1'
        std::cerr << "Invalid magic bytes in model header: " << std::hex << header.magic << std::endl;
        return false;
    }

    // 1. Token embeddings
    size_t embed_floats = (size_t)header.vocab_size * header.d_model;
    tok_embeddings = (float*)aligned_alloc(64, embed_floats * sizeof(float));
    file.read((char*)tok_embeddings, embed_floats * sizeof(float));

    // 2. Layers
    layers.resize(header.n_layers);
    for (uint32_t l = 0; l < header.n_layers; ++l) {
        LayerWeights& lw = layers[l];

        lw.attn_norm = (float*)aligned_alloc(64, header.d_model * sizeof(float));
        file.read((char*)lw.attn_norm, header.d_model * sizeof(float));

        size_t q_size = (size_t)(header.n_heads * header.head_dim) * header.d_model;
        lw.q_proj = (float*)aligned_alloc(64, q_size * sizeof(float));
        file.read((char*)lw.q_proj, q_size * sizeof(float));

        size_t k_size = (size_t)(header.n_kv_heads * header.head_dim) * header.d_model;
        lw.k_proj = (float*)aligned_alloc(64, k_size * sizeof(float));
        file.read((char*)lw.k_proj, k_size * sizeof(float));

        size_t v_size = (size_t)(header.n_kv_heads * header.head_dim) * header.d_model;
        lw.v_proj = (float*)aligned_alloc(64, v_size * sizeof(float));
        file.read((char*)lw.v_proj, v_size * sizeof(float));

        size_t o_size = (size_t)header.d_model * (header.n_heads * header.head_dim);
        lw.o_proj = (float*)aligned_alloc(64, o_size * sizeof(float));
        file.read((char*)lw.o_proj, o_size * sizeof(float));

        lw.ffn_norm = (float*)aligned_alloc(64, header.d_model * sizeof(float));
        file.read((char*)lw.ffn_norm, header.d_model * sizeof(float));

        size_t w1_size = (size_t)header.hidden_dim * header.d_model;
        lw.w1 = (float*)aligned_alloc(64, w1_size * sizeof(float));
        file.read((char*)lw.w1, w1_size * sizeof(float));

        size_t w2_size = (size_t)header.d_model * header.hidden_dim;
        lw.w2 = (float*)aligned_alloc(64, w2_size * sizeof(float));
        file.read((char*)lw.w2, w2_size * sizeof(float));

        size_t w3_size = (size_t)header.hidden_dim * header.d_model;
        lw.w3 = (float*)aligned_alloc(64, w3_size * sizeof(float));
        file.read((char*)lw.w3, w3_size * sizeof(float));
    }

    // 3. Final norm
    final_norm = (float*)aligned_alloc(64, header.d_model * sizeof(float));
    file.read((char*)final_norm, header.d_model * sizeof(float));

    // 4. Precompute RoPE
    size_t rope_size = (size_t)header.max_seq_len * header.head_dim;
    rope_cos = (float*)aligned_alloc(64, rope_size * sizeof(float));
    rope_sin = (float*)aligned_alloc(64, rope_size * sizeof(float));

    int half_dim = header.head_dim / 2; // 30
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
    kv_cache.init(header.n_layers, header.n_kv_heads, header.head_dim, header.max_seq_len);

    // 6. Allocate scratch buffers
    act_x = (float*)aligned_alloc(64, header.d_model * sizeof(float));
    act_norm_x = (float*)aligned_alloc(64, header.d_model * sizeof(float));
    act_q = (float*)aligned_alloc(64, header.n_heads * header.head_dim * sizeof(float));
    act_k = (float*)aligned_alloc(64, header.n_kv_heads * header.head_dim * sizeof(float));
    act_v = (float*)aligned_alloc(64, header.n_kv_heads * header.head_dim * sizeof(float));
    act_attn_out = (float*)aligned_alloc(64, header.n_heads * header.head_dim * sizeof(float));
    act_o_out = (float*)aligned_alloc(64, header.d_model * sizeof(float));
    act_w1 = (float*)aligned_alloc(64, header.hidden_dim * sizeof(float));
    act_w3 = (float*)aligned_alloc(64, header.hidden_dim * sizeof(float));
    act_ffn_out = (float*)aligned_alloc(64, header.d_model * sizeof(float));
    act_logits = (float*)aligned_alloc(64, header.vocab_size * sizeof(float));
    act_scores = (float*)aligned_alloc(64, header.max_seq_len * sizeof(float));

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
                #pragma omp simd
                for (int d = 0; d < head_dim; ++d) {
                    out_head[d] += weight * v_cached[d];
                }
            }
        }

        // Output projection
        gemv(act_o_out, lw.o_proj, act_attn_out, d_model, n_heads * head_dim);

        // Residual connection
        #pragma omp simd
        for (int i = 0; i < d_model; ++i) {
            act_x[i] += act_o_out[i];
        }

        // FFN RMSNorm
        rmsnorm(act_norm_x, act_x, lw.ffn_norm, d_model, header.norm_eps);

        // SwiGLU: w1 (gate) & w3 (up)
        gemv(act_w1, lw.w1, act_norm_x, hidden_dim, d_model);
        gemv(act_w3, lw.w3, act_norm_x, hidden_dim, d_model);

        // Fused SiLU * up
        #pragma omp simd
        for (int i = 0; i < hidden_dim; ++i) {
            act_w1[i] = silu(act_w1[i]) * act_w3[i];
        }

        // Down projection
        gemv(act_ffn_out, lw.w2, act_w1, d_model, hidden_dim);

        // Residual connection
        #pragma omp simd
        for (int i = 0; i < d_model; ++i) {
            act_x[i] += act_ffn_out[i];
        }
    }

    // Increment KV sequence length
    kv_cache.seq_len = pos + 1;

    // Final RMSNorm
    rmsnorm(act_norm_x, act_x, final_norm, d_model, header.norm_eps);

    int best_token = 0;
    if (compute_full_logits) {
        // Tied embeddings: output.weight == tok_embeddings
        gemv(act_logits, tok_embeddings, act_norm_x, header.vocab_size, d_model);

        float max_logit = -std::numeric_limits<float>::infinity();
        for (uint32_t i = 0; i < header.vocab_size; ++i) {
            if (act_logits[i] > max_logit) {
                max_logit = act_logits[i];
                best_token = (int)i;
            }
        }
    } else {
        // Fast parallel argmax without storing full 151,936 logits
        float global_max = -std::numeric_limits<float>::infinity();
        int global_best = 0;

        #pragma omp parallel
        {
            float local_max = -std::numeric_limits<float>::infinity();
            int local_best = 0;

            #pragma omp for schedule(static)
            for (uint32_t i = 0; i < header.vocab_size; ++i) {
                float val = vec_dot(tok_embeddings + (size_t)i * d_model, act_norm_x, d_model);
                if (val > local_max) {
                    local_max = val;
                    local_best = (int)i;
                }
            }

            #pragma omp critical
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
