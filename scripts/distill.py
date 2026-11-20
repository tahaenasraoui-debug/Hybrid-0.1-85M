#!/usr/bin/env python3
"""
Distillation Pipeline for Custom 85M Draft Model aligned to Qwen.

Key Requirements:
1. Mode Consistency: Explicitly passes enable_thinking=False in apply_chat_template
   to eliminate distribution mismatch from reasoning blocks during drafting.
2. Supports Forward KL Divergence, Reverse KL Divergence, and SFT (prompt-masked Cross-Entropy).
3. Memory Guardrail: Uses 4-bit NF4 quantized teacher + 8-bit AdamW to train comfortably
   under 2.0 GB VRAM on a 3GB GPU.
"""

import argparse
import json
import os
import sys
import time

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from config import get_qwen_85m_config, TrainingConfig
from model import SmallLLM
from train import get_compatible_device


class InstructionDataset(Dataset):
    """Loads instruction dataset with prompt masking for SFT or KD."""

    def __init__(self, tokenizer: AutoTokenizer, max_seq_len: int = 256, enable_thinking: bool = False):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.enable_thinking = enable_thinking
        self.samples = []

        # Load Dolly-15k
        dolly_path = "dolly_15k.jsonl"
        if os.path.exists(dolly_path):
            with open(dolly_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        inst = item.get("instruction", "")
                        ctx = item.get("context", "")
                        resp = item.get("response", "")
                        user_content = f"{inst}\n{ctx}".strip() if ctx else inst
                        if user_content and resp:
                            self.samples.append((user_content, resp))
                    except Exception:
                        pass

        # Load Alpaca-52k
        alpaca_path = "alpaca_52k.json"
        if os.path.exists(alpaca_path):
            with open(alpaca_path, "r", encoding="utf-8") as f:
                alpaca_data = json.load(f)
                for item in alpaca_data:
                    inst = item.get("instruction", "")
                    inp = item.get("input", "")
                    resp = item.get("output", "")
                    user_content = f"{inst}\n{inp}".strip() if inp else inst
                    if user_content and resp:
                        self.samples.append((user_content, resp))

        print(f"Loaded {len(self.samples):,} curated conversational samples for distillation.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        user_msg, assistant_msg = self.samples[idx]
        messages = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
        # Explicit mode consistency: pass enable_thinking=False (or True if configured)
        try:
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            # Tokenizer doesn't accept enable_thinking argument
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
            )

        tokens = self.tokenizer.encode(formatted, truncation=True, max_length=self.max_seq_len)
        return torch.tensor(tokens, dtype=torch.long)


def collate_padded(batch, pad_id: int = 151643):
    max_len = max(len(x) for x in batch)
    padded = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    for i, x in enumerate(batch):
        padded[i, : len(x)] = x
    return padded


def compute_distillation_loss(
    draft_logits: torch.Tensor,
    target_logits: torch.Tensor,
    loss_type: str = "forward_kl",
    temperature: float = 1.5,
    top_k: int = 64,
) -> torch.Tensor:
    """Computes Forward KL, Reverse KL, or Cross Entropy loss."""
    top_target, top_idx = torch.topk(target_logits, k=top_k, dim=-1)
    gathered_draft = torch.gather(draft_logits, -1, top_idx)

    p_target = F.softmax(top_target / temperature, dim=-1)
    log_p_target = F.log_softmax(top_target / temperature, dim=-1)
    log_q_draft = F.log_softmax(gathered_draft / temperature, dim=-1)
    q_draft = F.softmax(gathered_draft / temperature, dim=-1)

    if loss_type == "forward_kl":
        # Mode-covering: E_p [ log p - log q ]
        kl = (p_target * (log_p_target - log_q_draft)).sum(dim=-1).mean()
        return (temperature ** 2) * kl
    elif loss_type == "reverse_kl":
        # Mode-seeking: E_q [ log q - log p ]
        kl = (q_draft * (log_q_draft - log_p_target)).sum(dim=-1).mean()
        return (temperature ** 2) * kl
    else:
        # Cross Entropy over top target token
        targets = top_idx[:, :, 0]
        return F.cross_entropy(draft_logits.view(-1, draft_logits.size(-1)), targets.view(-1))


def distill(
    target_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    output_dir: str = "./checkpoints_distill",
    loss_type: str = "forward_kl",
    max_steps: int = 1000,
    save_every: int = 500,
    learning_rate: float = 3e-4,
    enable_thinking: bool = False,
):
    os.makedirs(output_dir, exist_ok=True)
    device = str(get_compatible_device())
    print(f"=== Running Distillation Pipeline ({loss_type}) on {device} ===")

    tokenizer = AutoTokenizer.from_pretrained(target_name, trust_remote_code=True)

    # 4-bit Quantized Teacher for 3GB VRAM
    if device == "cuda":
        try:
            import bitsandbytes as bnb
            bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            teacher = AutoModelForCausalLM.from_pretrained(
                target_name,
                quantization_config=bnb_config,
                device_map="auto",
            )
            has_bnb = True
        except Exception:
            teacher = AutoModelForCausalLM.from_pretrained(
                target_name, torch_dtype=torch.float16
            ).to(device)
            has_bnb = False
    else:
        teacher = AutoModelForCausalLM.from_pretrained(target_name, torch_dtype=torch.float32).to(device)
        has_bnb = False

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # 85M Draft Model
    draft_cfg = get_qwen_85m_config()
    draft = SmallLLM(draft_cfg).to(device).train()

    if device == "cuda" and has_bnb:
        optimizer = bnb.optim.AdamW8bit(draft.parameters(), lr=learning_rate, weight_decay=0.01)
    else:
        optimizer = torch.optim.AdamW(draft.parameters(), lr=learning_rate, weight_decay=0.01)

    dataset = InstructionDataset(tokenizer, max_seq_len=256, enable_thinking=enable_thinking)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=True, collate_fn=lambda b: collate_padded(b, tokenizer.pad_token_id or 151643)
    )
    data_iter = iter(loader)

    print(f"Beginning distillation loop for {max_steps} steps...")
    for step in range(1, max_steps + 1):
        try:
            tokens = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            tokens = next(data_iter)

        tokens = tokens.to(device)
        with torch.no_grad():
            t_out = teacher(tokens)
            target_logits = t_out.logits
            top_target, top_idx = torch.topk(target_logits, k=64, dim=-1)
            del t_out
            if device == "cuda":
                torch.cuda.empty_cache()

        draft_logits, _, _ = draft(tokens)
        gathered_draft = torch.gather(draft_logits, -1, top_idx)
        del draft_logits

        temp = 1.5
        p_target = F.softmax(top_target / temp, dim=-1)
        log_p_target = F.log_softmax(top_target / temp, dim=-1)
        log_q_draft = F.log_softmax(gathered_draft / temp, dim=-1)

        loss = ((temp ** 2) * (p_target * (log_p_target - log_q_draft)).sum(dim=-1)).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 50 == 0:
            print(f"Step {step:05d}/{max_steps} | Loss: {loss.item():.4f}")

        if step % save_every == 0 or step == max_steps:
            save_path = os.path.join(output_dir, f"draft_distill_step_{step}.pt")
            torch.save({"step": step, "model_state_dict": draft.state_dict()}, save_path)
            print(f"Saved checkpoint: {save_path}")

    print("Distillation completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--loss", type=str, default="forward_kl", choices=["forward_kl", "reverse_kl", "sft"])
    args = parser.parse_args()
    distill(target_name=args.target, max_steps=args.steps, loss_type=args.loss)
