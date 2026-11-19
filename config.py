from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for SmallLLM architecture."""
    vocab_size: int = 32000
    d_model: int = 2048
    n_layers: int = 22
    n_heads: int = 16
    n_kv_heads: int = 4  # Grouped Query Attention (GQA)
    hidden_dim: int = 5632  # SwiGLU hidden dimension
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    tie_embeddings: bool = False
    dropout: float = 0.0

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def estimate_parameter_count(self) -> int:
        """Calculates total parameter count according to the config."""
        embed = self.vocab_size * self.d_model
        head_dim = self.head_dim
        # Attention: Q + K + V + Output
        q_proj = self.d_model * (self.n_heads * head_dim)
        k_proj = self.d_model * (self.n_kv_heads * head_dim)
        v_proj = self.d_model * (self.n_kv_heads * head_dim)
        o_proj = (self.n_heads * head_dim) * self.d_model
        attn = q_proj + k_proj + v_proj + o_proj

        # SwiGLU: gate (w1) + down (w2) + up (w3)
        ffn = 3 * self.d_model * self.hidden_dim
        norms = 2 * self.d_model  # attn_norm + ffn_norm

        layer_params = attn + ffn + norms
        final_norm = self.d_model
        lm_head = 0 if self.tie_embeddings else (self.vocab_size * self.d_model)

        return embed + (self.n_layers * layer_params) + final_norm + lm_head


def get_1b_config() -> ModelConfig:
    """
    ~1.12 Billion parameters.
    Similar size and layout to TinyLlama / Llama-3.2 1B.
    """
    return ModelConfig(
        vocab_size=32000,
        d_model=2048,
        n_layers=22,
        n_heads=16,
        n_kv_heads=4,
        hidden_dim=5632,
        max_seq_len=2048,
        rope_theta=10000.0,
        tie_embeddings=False,
    )


def get_2b_config() -> ModelConfig:
    """
    ~2.01 Billion parameters.
    Balanced 2B model architecture (2048 hidden dim, 36 layers, GQA 4x).
    """
    return ModelConfig(
        vocab_size=32000,
        d_model=2048,
        n_layers=36,
        n_heads=16,
        n_kv_heads=4,
        hidden_dim=6800,
        max_seq_len=2048,
        rope_theta=10000.0,
        tie_embeddings=False,
    )


def get_3b_config() -> ModelConfig:
    """
    ~3.02 Billion parameters.
    Similar size and layout to Llama-3.2 3B / StableLM-3B.
    """
    return ModelConfig(
        vocab_size=32000,
        d_model=3072,
        n_layers=28,
        n_heads=24,
        n_kv_heads=8,
        hidden_dim=8192,
        max_seq_len=4096,
        rope_theta=10000.0,
        tie_embeddings=False,
    )


def get_5b_config() -> ModelConfig:
    """
    ~4.87 Billion parameters.
    Ideal sweet spot between 3B and 7B.
    """
    return ModelConfig(
        vocab_size=32000,
        d_model=4096,
        n_layers=26,
        n_heads=32,
        n_kv_heads=8,
        hidden_dim=11008,
        max_seq_len=4096,
        rope_theta=10000.0,
        tie_embeddings=False,
    )


def get_tiny_config() -> ModelConfig:
    """
    ~18.7 Million parameters.
    Designed for fast testing, code verification, and running on local GPUs (e.g. 2-4GB VRAM) or CPU.
    """
    return ModelConfig(
        vocab_size=4096,
        d_model=512,
        n_layers=6,
        n_heads=8,
        n_kv_heads=2,
        hidden_dim=1376,
        max_seq_len=512,
        rope_theta=10000.0,
        tie_embeddings=True,
    )


def get_gtx1050_config() -> ModelConfig:
    """
    ~87.5 Million parameters.
    The absolute sweet spot for a 3GB VRAM laptop GPU (GTX 1050 Mobile):
    - Full Transformer with 10 layers, GQA, SwiGLU, RMSNorm
    - Fits completely in 3GB VRAM including AdamW optimizer states and activations
    - Trains fast and can learn real English grammar, stories, and domain knowledge.
    """
    return ModelConfig(
        vocab_size=32000,
        d_model=768,
        n_layers=10,
        n_heads=12,
        n_kv_heads=4,
        hidden_dim=2048,
        max_seq_len=512,
        rope_theta=10000.0,
        tie_embeddings=True,
    )


def get_qwen_85m_config() -> ModelConfig:
    """
    ~85.0 Million parameters matching Qwen tokenizer vocabulary (vocab_size=151936).
    Designed specifically as a high-speed draft model for speculative decoding:
    - vocab_size=151936 (exact match for Qwen2.5 / Qwen3 family)
    - tied embeddings to keep parameters at ~85M
    - RoPE base 1,000,000 (Qwen standard)
    """
    return ModelConfig(
        vocab_size=151936,
        d_model=480,
        n_layers=5,
        n_heads=8,
        n_kv_heads=2,
        hidden_dim=1280,
        max_seq_len=2048,
        rope_theta=1000000.0,
        tie_embeddings=True,
    )


@dataclass
class TrainingConfig:
    """Hyperparameters for pre-training."""
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.1
    warmup_steps: int = 100
    max_steps: int = 2000
    clip_grad_norm: float = 1.0
    device: str = "cuda"
    precision: str = "float16"  # "bfloat16", "float16", or "float32"
    save_every: int = 500
    eval_every: int = 100
    checkpoint_dir: str = "./checkpoints"
