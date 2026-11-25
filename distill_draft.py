"""
Phase 1b: Knowledge Distillation Pipeline for Speculative Decoding.
Fine-tunes the 85M draft model on soft probabilities from the target model
using forward KL-divergence to maximize the speculative acceptance rate (tau).
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import get_qwen_85m_config, TrainingConfig
from model import SmallLLM
from train import configure_optimizers, get_lr


class DistillationDataset(Dataset):
    """Tokenizes text sequences and formats them with the target chat template."""
    def __init__(self, texts: list, tokenizer: AutoTokenizer, max_seq_len: int = 512):
        self.samples = []
        for text in texts:
            if isinstance(text, str):
                messages = [{"role": "user", "content": text}]
                formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                tokens = tokenizer.encode(formatted, truncation=True, max_length=max_seq_len)
            else:
                tokens = text[:max_seq_len]

            if len(tokens) >= 8:
                self.samples.append(torch.tensor(tokens, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.samples[idx]


def collate_fn(batch, pad_token_id: int = 0):
    """Pads batch to maximum sequence length in batch."""
    max_len = max(len(x) for x in batch)
    padded_batch = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    for i, x in enumerate(batch):
        padded_batch[i, : len(x)] = x
    return padded_batch


def forward_kl_loss(
    draft_logits: torch.Tensor,
    target_logits: torch.Tensor,
    temperature: float = 1.5,
    top_k: int = 100,
) -> torch.Tensor:
    """
    Computes memory-efficient top-k forward KL-divergence:
    KL(Softmax(Z_target/T) || Softmax(Z_draft/T)).
    Focuses on top-k highest probability tokens of the target model.
    """
    # Flatten across batch and sequence dimensions: (N, vocab_size)
    target_flat = target_logits.view(-1, target_logits.size(-1))
    draft_flat = draft_logits.view(-1, draft_logits.size(-1))

    # Identify top-k target logits to save memory and avoid full 151,936 dense divergence
    top_target_logits, top_indices = torch.topk(target_flat, k=top_k, dim=-1)

    # Gather matching draft logits at top-k indices
    gathered_draft_logits = torch.gather(draft_flat, -1, top_indices)

    # Compute softmax over top-k
    p_target = F.softmax(top_target_logits / temperature, dim=-1)
    log_q_draft = F.log_softmax(gathered_draft_logits / temperature, dim=-1)

    # KL divergence: sum(P * (log P - log Q))
    log_p_target = F.log_softmax(top_target_logits / temperature, dim=-1)
    kl = (p_target * (log_p_target - log_q_draft)).sum(dim=-1).mean()

    return (temperature ** 2) * kl


def distill(
    draft_model: SmallLLM,
    target_model: nn.Module,
    tokenizer: AutoTokenizer,
    texts: list,
    output_checkpoint_path: str = "./checkpoints/distilled_draft_85m.pt",
    batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 3e-4,
    max_steps: int = 200,
    device: str = "cpu",
):
    """Runs the distillation loop."""
    os.makedirs(os.path.dirname(output_checkpoint_path), exist_ok=True)
    device = torch.device(device)
    print(f"Starting distillation on device: {device}")

    draft_model.to(device)
    target_model.to(device)
    draft_model.train()
    target_model.eval()

    # Freeze target teacher model completely
    for p in target_model.parameters():
        p.requires_grad = False

    dataset = DistillationDataset(texts, tokenizer, max_seq_len=256)
    pad_id = getattr(tokenizer, "pad_token_id", 0) or 0
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, pad_id))

    optimizer = configure_optimizers(draft_model, weight_decay=0.01, learning_rate=learning_rate)
    train_cfg = TrainingConfig(
        learning_rate=learning_rate,
        min_learning_rate=learning_rate * 0.1,
        warmup_steps=20,
        max_steps=max_steps,
    )

    step = 0
    data_iter = iter(loader)
    t0 = time.time()
    running_kl = 0.0

    print(f"Distilling {len(dataset):,} conversational samples over {max_steps} steps...")

    while step < max_steps:
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(gradient_accumulation_steps):
            try:
                tokens = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                tokens = next(data_iter)

            tokens = tokens.to(device)

            # Target forward pass (no gradients)
            with torch.no_grad():
                target_out = target_model(tokens)
                target_logits = target_out.logits

            # Draft forward pass (compute gradients)
            draft_logits, _, _ = draft_model(tokens)

            # Forward KL divergence loss
            loss = forward_kl_loss(draft_logits, target_logits, temperature=1.5, top_k=64)
            loss = loss / gradient_accumulation_steps
            loss.backward()

            accum_loss += loss.item()

        # Update learning rate
        lr = get_lr(step, train_cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        torch.nn.utils.clip_grad_norm_(draft_model.parameters(), 1.0)
        optimizer.step()

        running_kl += accum_loss
        step += 1

        if step % 10 == 0:
            dt = time.time() - t0
            t0 = time.time()
            avg_kl = running_kl / 10
            running_kl = 0.0
            print(f"Step {step:04d}/{max_steps} | Forward KL Loss: {avg_kl:.4f} | LR: {lr:.2e} | Time/10steps: {dt:.1f}s")

    print(f"\n--> Saving distilled draft model to: {output_checkpoint_path}")
    torch.save(
        {
            "step": step,
            "model_state_dict": draft_model.state_dict(),
            "config": draft_model.config,
        },
        output_checkpoint_path,
    )
    print("✅ Distillation complete! The draft model is now aligned with target soft probabilities.")


if __name__ == "__main__":
    from verify_compat import TEST_PROMPTS

    print("Running lightweight distillation alignment demo...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    target_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )

    draft_cfg = get_qwen_85m_config()
    draft_model = SmallLLM(draft_cfg)

    # Run quick 15-step demo alignment
    distill(
        draft_model=draft_model,
        target_model=target_model,
        tokenizer=tokenizer,
        texts=TEST_PROMPTS * 3,
        max_steps=15,
        device="cpu",
    )
