from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    More computationally efficient than LayerNorm by avoiding mean centering.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_rope_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precomputes cos and sin values for Rotary Position Embeddings (RoPE).
    dim: head_dim
    end: max_seq_len
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # (seq_len, dim // 2)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    # Duplicate along the last dimension so shape becomes (seq_len, dim)
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input."""
    half_dim = x.shape[-1] // 2
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Applies Rotary Position Embedding to x.
    x: (batch, n_heads, seq_len, head_dim)
    cos, sin: (seq_len, head_dim)
    """
    # Reshape cos/sin to (1, 1, seq_len, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(1).type_as(x)
    sin = sin.unsqueeze(0).unsqueeze(1).type_as(x)
    return (x * cos) + (rotate_half(x) * sin)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeats KV heads across query heads for Grouped Query Attention (GQA).
    x: (batch, n_kv_heads, seq_len, head_dim) -> (batch, n_kv_heads * n_rep, seq_len, head_dim)
    """
    if n_rep == 1:
        return x
    bs, n_kv_heads, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :].expand(bs, n_kv_heads, n_rep, seq_len, head_dim)
    return x.reshape(bs, n_kv_heads * n_rep, seq_len, head_dim)


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) layer with KV cache support.
    Reduces KV cache memory by using fewer KV heads than Query heads.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads

        self.q_proj = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=False)

        self.dropout = config.dropout

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = x.shape

        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (batch_size, n_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Update KV cache for incremental autoregressive generation
        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            k = torch.cat([prev_k, k], dim=2)
            v = torch.cat([prev_v, v], dim=2)

        new_kv_cache = (k, v) if use_cache else None

        # Repeat KV heads for GQA
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # Scaled Dot-Product Attention (PyTorch handles FlashAttention/Math kernels automatically)
        past_len = kv_cache[0].shape[2] if kv_cache is not None else 0
        total_kv_len = k.shape[2]

        if seq_len > 1:
            if past_len == 0:
                is_causal = True
                attn_mask = None
            else:
                is_causal = False
                # Build causal mask when decoding multiple draft tokens with an existing KV cache
                q_pos = torch.arange(seq_len, device=x.device).unsqueeze(1) + past_len
                k_pos = torch.arange(total_kv_len, device=x.device).unsqueeze(0)
                attn_mask = (k_pos <= q_pos)  # Shape: (seq_len, total_kv_len)
        else:
            is_causal = False
            attn_mask = None

        dropout_p = self.dropout if self.training else 0.0

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )

        # Reshape back: (batch_size, seq_len, d_model)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.o_proj(attn_out)

        return output, new_kv_cache


class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU feed-forward network as used in LLaMA / PaLM / Mistral.
    FFN(x) = (SiLU(x * W_gate) * (x * W_up)) * W_down
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.hidden_dim, bias=False)  # gate_proj
        self.w2 = nn.Linear(config.hidden_dim, config.d_model, bias=False)  # down_proj
        self.w3 = nn.Linear(config.d_model, config.hidden_dim, bias=False)  # up_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """Pre-norm Transformer decoder layer."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attn = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.ffn = SwiGLUFeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-norm Self-Attention with residual connection
        normed_x = self.attn_norm(x)
        attn_out, new_kv_cache = self.attn(
            normed_x, cos=cos, sin=sin, kv_cache=kv_cache, use_cache=use_cache
        )
        x = x + attn_out

        # Pre-norm Feed-Forward with residual connection
        x = x + self.ffn(self.ffn_norm(x))

        return x, new_kv_cache


class SmallLLM(nn.Module):
    """
    Complete Causal Decoder-Only Transformer Language Model (1B - 5B parameters).
    Features:
    - Root Mean Square Layer Normalization (RMSNorm)
    - Rotary Positional Embeddings (RoPE)
    - Grouped Query Attention (GQA)
    - SwiGLU Non-linear Activations
    - Native PyTorch Scaled Dot-Product Attention (FlashAttention compatible)
    - Autoregressive KV-Caching for inference
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying (optional)
        if config.tie_embeddings:
            self.output.weight = self.tok_embeddings.weight

        # Precompute RoPE cos and sin tables
        cos, sin = precompute_rope_freqs_cis(
            dim=config.head_dim,
            end=config.max_seq_len,
            theta=config.rope_theta,
        )
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_embed = sum(
            p.numel() for n, p in self.named_parameters() if "tok_embeddings" not in n and "output" not in n
        )
        return {
            "total": total,
            "trainable": trainable,
            "non_embedding": non_embed,
            "total_millions": total / 1e6,
            "total_billions": total / 1e9,
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_caches: Optional[list] = None,
        use_cache: bool = False,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[list]]:
        """
        input_ids: (batch_size, seq_len)
        targets: Optional (batch_size, seq_len) for computing cross entropy loss
        start_pos: start index for RoPE positioning during KV-cached generation
        """
        _, seq_len = input_ids.shape

        h = self.tok_embeddings(input_ids)

        # Slice RoPE frequencies for current positions
        cos = self.cos_cached[start_pos : start_pos + seq_len]
        sin = self.sin_cached[start_pos : start_pos + seq_len]

        new_kv_caches = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            h, new_cache = layer(
                h, cos=cos, sin=sin, kv_cache=layer_cache, use_cache=use_cache
            )
            if use_cache:
                new_kv_caches.append(new_cache)

        h = self.norm(h)
        logits = self.output(h)

        loss = None
        if targets is not None:
            # Flatten to compute CrossEntropyLoss: (batch * seq_len, vocab_size) vs (batch * seq_len)
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), targets.view(-1))

        return logits, loss, new_kv_caches
