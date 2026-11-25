"""
Unit tests for stochastic speculative sampling, top-p filtering, engine guards,
max_seq_len clamping, deprecation shim compatibility, and distillation loss wiring.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from config import ModelConfig
from model import SmallLLM
import scripts.distill as distill_module
from speculative.engine import SpeculativeDecodingEngine
from speculative.rejection_sampler import RejectionSampler
from src.speculative import (
    SpeculativeEngine,
    apply_top_p_filtering,
    logits_to_probs,
    speculative_generate,
)
from tokenizer import ByteTokenizer


class _FixedLogitsLM(nn.Module):
    """Minimal causal LM returning fixed per-step logits with tuple KV-cache."""

    def __init__(self, logits_1d: torch.Tensor):
        super().__init__()
        self.register_buffer("_logits", logits_1d.clone())
        self.dummy = nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(max_seq_len=128, eos_token_id=None)

    def forward(self, input_ids: torch.Tensor, past_key_values=None, use_cache: bool = True):
        batch_size, seq_len = input_ids.shape
        past_len = past_key_values[0][0].shape[-2] if past_key_values is not None else 0
        total_len = past_len + seq_len
        logits = self._logits.view(1, 1, -1).expand(batch_size, seq_len, -1).clone()
        new_cache = None
        if use_cache:
            k = torch.zeros(batch_size, 1, total_len, 4, device=input_ids.device)
            v = torch.zeros(batch_size, 1, total_len, 4, device=input_ids.device)
            new_cache = ((k, v),)
        return SimpleNamespace(logits=logits, past_key_values=new_cache)


def test_apply_top_p_filtering_shapes_and_edge_cases():
    probs_1d = torch.tensor([0.45, 0.35, 0.10, 0.06, 0.04])

    # top_p >= 1.0 leaves distribution unchanged
    unchanged = apply_top_p_filtering(probs_1d, 1.0)
    assert torch.allclose(unchanged, probs_1d)

    # top_p <= 0.0 collapses onto argmax token
    greedy = apply_top_p_filtering(probs_1d, 0.0)
    assert torch.allclose(greedy, torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]))

    # top_p = 0.70 keeps tokens {0, 1} (cumsum 0.80) and zeros {2, 3, 4}
    filtered = apply_top_p_filtering(probs_1d, 0.70)
    expected = torch.tensor([0.45 / 0.80, 0.35 / 0.80, 0.0, 0.0, 0.0])
    assert torch.allclose(filtered, expected, atol=1e-6)
    assert torch.isclose(filtered.sum(), torch.tensor(1.0))

    # 2D batch support matches RejectionSampler.apply_top_p
    probs_2d = torch.stack([probs_1d, torch.tensor([0.10, 0.60, 0.20, 0.05, 0.05])], dim=0)
    out_src = apply_top_p_filtering(probs_2d, 0.70)
    out_sampler = RejectionSampler.apply_top_p(probs_2d, 0.70)
    assert torch.allclose(out_src, out_sampler, atol=1e-6)
    assert torch.allclose(out_src.sum(dim=-1), torch.ones(2))


def test_stochastic_rejection_sampler_exact_distribution():
    torch.manual_seed(1234)
    target_probs_base = torch.tensor([0.45, 0.35, 0.10, 0.06, 0.04])
    draft_probs_base = torch.tensor([0.15, 0.15, 0.40, 0.20, 0.10])
    top_p = 0.70

    target_probs = torch.stack([target_probs_base, target_probs_base], dim=0)
    draft_probs = draft_probs_base.unsqueeze(0)
    draft_nucleus = RejectionSampler.apply_top_p(draft_probs_base, top_p)
    expected_target = RejectionSampler.apply_top_p(target_probs_base, top_p)

    counts = torch.zeros(5, dtype=torch.long)
    trials = 4000
    for _ in range(trials):
        d_tok = int(torch.multinomial(draft_nucleus, num_samples=1).item())
        res = RejectionSampler.sample(
            draft_tokens=[d_tok],
            draft_probs=draft_probs,
            target_probs=target_probs,
            temperature=1.0,
            top_p=top_p,
        )
        counts[res.accepted_tokens[0]] += 1

    # Tokens outside target top_p nucleus must never be emitted
    assert counts[2].item() == 0
    assert counts[3].item() == 0
    assert counts[4].item() == 0

    empirical = counts.float() / trials
    assert torch.allclose(empirical, expected_target, atol=0.035), (
        f"Expected {expected_target.tolist()}, got {empirical.tolist()}"
    )


def test_speculative_engine_stochastic_top_p_distribution():
    torch.manual_seed(42)
    target_probs_base = torch.tensor([0.45, 0.35, 0.10, 0.06, 0.04])
    draft_probs_base = torch.tensor([0.15, 0.15, 0.40, 0.20, 0.10])
    target_logits = torch.log(target_probs_base)
    draft_logits = torch.log(draft_probs_base)

    target_model = _FixedLogitsLM(target_logits)
    draft_model = _FixedLogitsLM(draft_logits)
    tokenizer = ByteTokenizer(vocab_size=5)

    top_p = 0.70
    expected = logits_to_probs(target_logits, temperature=1.0, top_p=top_p)

    counts = torch.zeros(5, dtype=torch.long)
    trials = 2000
    for _ in range(trials):
        res = speculative_generate(
            target_model=target_model,
            draft_model=draft_model,
            tokenizer=tokenizer,
            prompt=[1, 2],
            max_new_tokens=1,
            k=2,
            temperature=1.0,
            top_p=top_p,
            device="cpu",
        )
        tok = res["generated_ids"][0]
        counts[tok] += 1

    assert counts[2].item() == 0
    assert counts[3].item() == 0
    assert counts[4].item() == 0
    empirical = counts.float() / trials
    assert torch.allclose(empirical, expected, atol=0.045), (
        f"Stochastic speculative distribution mismatch: {empirical.tolist()} vs {expected.tolist()}"
    )


def test_self_draft_stochastic_top_p_full_acceptance():
    torch.manual_seed(99)
    cfg = ModelConfig(
        vocab_size=64,
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        hidden_dim=64,
        max_seq_len=64,
        tie_embeddings=True,
    )
    model = SmallLLM(cfg).eval()
    tokenizer = ByteTokenizer(vocab_size=64)

    res = speculative_generate(
        target_model=model,
        draft_model=model,
        tokenizer=tokenizer,
        prompt=[1, 10, 20, 30],
        max_new_tokens=16,
        k=4,
        temperature=0.8,
        top_p=0.7,
        device="cpu",
    )
    assert res["num_generated"] == 16
    assert res["num_drafted"] > 0
    assert res["num_accepted"] == res["num_drafted"]
    assert res["acceptance_rate"] == 1.0


@pytest.mark.parametrize(
    "flag_attr",
    ["is_loaded_in_4bit", "is_loaded_in_8bit", "hf_device_map"],
)
def test_speculative_engine_skips_to_for_quantized_or_dispatched_model(flag_attr):
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=64,
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        hidden_dim=64,
        max_seq_len=32,
        tie_embeddings=True,
    )
    target_model = SmallLLM(cfg).eval()
    draft_model = SmallLLM(cfg).eval()
    tokenizer = ByteTokenizer(vocab_size=64)

    if flag_attr == "hf_device_map":
        target_model.hf_device_map = {"": "cpu"}
    else:
        setattr(target_model, flag_attr, True)

    def _forbid_to(*_args, **_kwargs):
        raise AssertionError(".to() must not be called on quantized or dispatched models")

    target_model.to = _forbid_to  # type: ignore[method-assign]

    engine = SpeculativeEngine(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        device="cpu",
    )
    out = engine.generate(prompt=[1, 5, 9], max_new_tokens=6, k=3, temperature=0.0)
    assert out["num_generated"] == 6


def test_speculative_engine_clamps_k_to_max_seq_len():
    torch.manual_seed(11)
    cfg = ModelConfig(
        vocab_size=64,
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        hidden_dim=64,
        max_seq_len=16,
        tie_embeddings=True,
    )
    target_model = SmallLLM(cfg).eval()
    draft_model = SmallLLM(cfg).eval()
    tokenizer = ByteTokenizer(vocab_size=64)

    for temp, top_p in [(0.0, 1.0), (0.8, 0.85)]:
        res = speculative_generate(
            target_model=target_model,
            draft_model=draft_model,
            tokenizer=tokenizer,
            prompt=[1] * 13,  # remaining_seq = 16 - 13 = 3 < k=8
            max_new_tokens=10,
            k=8,
            temperature=temp,
            top_p=top_p,
            device="cpu",
            max_seq_len=16,
        )
        assert 1 <= res["num_generated"] <= 10


def test_deprecated_engine_shim_warns_and_preserves_keys():
    torch.manual_seed(5)
    cfg = ModelConfig(
        vocab_size=64,
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        hidden_dim=64,
        max_seq_len=32,
        tie_embeddings=True,
    )
    target_model = SmallLLM(cfg).eval()
    draft_model = SmallLLM(cfg).eval()
    tokenizer = ByteTokenizer(vocab_size=64)

    with pytest.warns(DeprecationWarning, match="SpeculativeDecodingEngine is deprecated"):
        legacy_engine = SpeculativeDecodingEngine(
            draft_model=draft_model,
            target_model=target_model,
            tokenizer=tokenizer,
            device="cpu",
        )
    res = legacy_engine.generate(prompt=[1, 4, 8], max_new_tokens=6, K=3, temperature=0.0)
    assert res["generated_text"] == res["text"]
    assert res["mean_accepted_per_round"] == res["mean_accepted_per_step"]
    assert pytest.approx(res["wall_time_ms"]) == res["wall_time_s"] * 1000.0
    assert len(res["generated_ids"]) == 6


def test_compute_distillation_loss_modes_and_wiring(monkeypatch, tmp_path):
    torch.manual_seed(3)
    draft_logits = torch.randn(2, 4, 16, requires_grad=True)
    target_logits = torch.randn(2, 4, 16)

    l_fkl = distill_module.compute_distillation_loss(draft_logits, target_logits, loss_type="forward_kl")
    l_rkl = distill_module.compute_distillation_loss(draft_logits, target_logits, loss_type="reverse_kl")
    l_sft = distill_module.compute_distillation_loss(draft_logits, target_logits, loss_type="sft")

    for loss in (l_fkl, l_rkl, l_sft):
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        assert loss.requires_grad

    assert not torch.isclose(l_fkl, l_rkl)
    assert not torch.isclose(l_fkl, l_sft)

    with pytest.raises(ValueError, match="Unsupported loss_type"):
        distill_module.compute_distillation_loss(draft_logits, target_logits, loss_type="invalid")

    # Verify distill() forwards loss_type into compute_distillation_loss
    seen_loss_types = []
    real_compute = distill_module.compute_distillation_loss

    def _spy_compute(*args, **kwargs):
        seen_loss_types.append(kwargs.get("loss_type"))
        return real_compute(*args, **kwargs)

    class _DummyTeacher(nn.Module):
        def __init__(self, vocab_size: int):
            super().__init__()
            self.p = nn.Parameter(torch.zeros(1))
            self.vocab_size = vocab_size

        def forward(self, input_ids):
            b, s = input_ids.shape
            return SimpleNamespace(logits=torch.randn(b, s, self.vocab_size))

    class _DummyDraft(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.emb = nn.Embedding(cfg.vocab_size, 8)
            self.proj = nn.Linear(8, cfg.vocab_size)

        def forward(self, input_ids):
            return self.proj(self.emb(input_ids)), None, None

    class _DummyDataset(torch.utils.data.Dataset):
        def __init__(self, *_args, **_kwargs):
            pass

        def __len__(self):
            return 2

        def __getitem__(self, idx):
            return torch.tensor([1, 2, 3, 4], dtype=torch.long)

    tiny_cfg = ModelConfig(
        vocab_size=32,
        d_model=16,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        hidden_dim=32,
        max_seq_len=16,
        tie_embeddings=True,
    )
    monkeypatch.setattr(distill_module, "compute_distillation_loss", _spy_compute)
    monkeypatch.setattr(distill_module, "get_compatible_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(distill_module, "get_qwen_85m_config", lambda: tiny_cfg)
    monkeypatch.setattr(distill_module, "SmallLLM", _DummyDraft)
    monkeypatch.setattr(distill_module, "InstructionDataset", _DummyDataset)
    monkeypatch.setattr(
        distill_module.AutoTokenizer,
        "from_pretrained",
        staticmethod(lambda *_a, **_kw: SimpleNamespace(pad_token_id=0)),
    )
    monkeypatch.setattr(
        distill_module.AutoModelForCausalLM,
        "from_pretrained",
        staticmethod(lambda *_a, **_kw: _DummyTeacher(tiny_cfg.vocab_size)),
    )

    for chosen in ("reverse_kl", "sft"):
        distill_module.distill(
            target_name="dummy",
            output_dir=str(tmp_path / chosen),
            loss_type=chosen,
            max_steps=1,
            save_every=1,
        )

    assert seen_loss_types == ["reverse_kl", "sft"]
