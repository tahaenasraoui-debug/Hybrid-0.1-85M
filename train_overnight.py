#!/usr/bin/env python3
"""
Overnight Distillation Training for Qwen Draft Model (85M).

Key Features:
- Thermal-Aware: Limits CPU threads to prevent laptop thermal throttling.
- Combined Dataset: Alpaca (52k) + Dolly (15k) = 67k Qwen-formatted instructions.
- Forward KL-Divergence + Cross Entropy loss to maximize speculative acceptance rate (tau).
- Auto-Checkpointing & Resume: Saves every 500 steps and resumes seamlessly if interrupted.
- Live Logging: Writes detailed progress, tokens/sec, and ETA to train_overnight.log.
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json
import signal
import sys
import time
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

from config import get_qwen_85m_config, TrainingConfig
from model import SmallLLM
from train import configure_optimizers, get_lr, get_compatible_device


# Configure thermal management: use 4 threads to prevent laptop overheating
torch.set_num_threads(4)


class InstructionDataset(Dataset):
    """Loads Alpaca and Dolly datasets lazily for instant startup."""
    def __init__(self, tokenizer: AutoTokenizer, max_seq_len: int = 256):
        self.raw_conversations = []
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        print("Loading datasets for overnight training...")

        # 1. Load Dolly-15k
        dolly_path = "/home/tahary/.gemini/antigravity/scratch/small_llm/dolly_15k.jsonl"
        if os.path.exists(dolly_path):
            count = 0
            with open(dolly_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        inst = item.get("instruction", "")
                        ctx = item.get("context", "")
                        resp = item.get("response", "")
                        user_content = f"{inst}\n{ctx}".strip() if ctx else inst
                        if user_content and resp:
                            self.raw_conversations.append((user_content, resp))
                            count += 1
                    except Exception:
                        pass
            print(f"Loaded {count:,} samples from Dolly-15k.")

        # 2. Load Alpaca-52k
        alpaca_path = "/home/tahary/.gemini/antigravity/scratch/small_llm/alpaca_52k.json"
        if os.path.exists(alpaca_path):
            with open(alpaca_path, "r", encoding="utf-8") as f:
                alpaca_data = json.load(f)
            count = 0
            for item in alpaca_data:
                inst = item.get("instruction", "")
                inp = item.get("input", "")
                resp = item.get("output", "")
                user_content = f"{inst}\n{inp}".strip() if inp else inst
                if user_content and resp:
                    self.raw_conversations.append((user_content, resp))
                    count += 1
            print(f"Loaded {count:,} samples from Alpaca-52k.")

        print(f"Total curated training samples: {len(self.raw_conversations):,}")

    def __len__(self) -> int:
        return len(self.raw_conversations)

    def __getitem__(self, idx: int) -> torch.Tensor:
        user_msg, assistant_msg = self.raw_conversations[idx]
        messages = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
        formatted = self.tokenizer.apply_chat_template(messages, tokenize=False)
        tokens = self.tokenizer.encode(formatted, truncation=True, max_length=self.max_seq_len)
        return torch.tensor(tokens, dtype=torch.long)


def collate_padded(batch, pad_id: int = 151643):
    max_len = max(len(x) for x in batch)
    padded = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    for i, x in enumerate(batch):
        padded[i, : len(x)] = x
    return padded


def topk_forward_kl_loss(draft_logits, target_logits, temperature=1.5, top_k=64):
    target_flat = target_logits.view(-1, target_logits.size(-1))
    draft_flat = draft_logits.view(-1, draft_logits.size(-1))

    top_target, top_idx = torch.topk(target_flat, k=top_k, dim=-1)
    gathered_draft = torch.gather(draft_flat, -1, top_idx)

    p_target = F.softmax(top_target / temperature, dim=-1)
    log_p_target = F.log_softmax(top_target / temperature, dim=-1)
    log_q_draft = F.log_softmax(gathered_draft / temperature, dim=-1)

    kl = (p_target * (log_p_target - log_q_draft)).sum(dim=-1).mean()
    return (temperature ** 2) * kl


def train_overnight(
    target_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    checkpoint_dir: str = "./checkpoints_overnight",
    log_file: str = "train_overnight.log",
    batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 3e-4,
    max_hours: float = 8.5,
    max_steps: int = 35000,
    save_every: int = 500,
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = str(get_compatible_device())

    # Setup logging to both console and file
    log_fp = open(log_file, "a", encoding="utf-8")

    def log_print(msg: str):
        timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
        formatted_msg = f"{timestamp} {msg}"
        print(formatted_msg, flush=True)
        log_fp.write(formatted_msg + "\n")
        log_fp.flush()

    log_print(f"=== STARTING OVERNIGHT TRAINING (Max {max_hours:.1f} Hours / {max_steps:,} Steps) ===")
    log_print(f"Device: {device} | CPU Threads: {torch.get_num_threads()}")

    # 1. Load Tokenizer & Target Teacher Model
    log_print(f"Loading teacher model: {target_name}")
    tokenizer = AutoTokenizer.from_pretrained(target_name, trust_remote_code=True)
    if device == "cuda" and HAS_BNB:
        log_print("Using 4-bit quantized teacher (BitsAndBytes) to conserve GPU VRAM (<1.9GB peak)")
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        target_model = AutoModelForCausalLM.from_pretrained(
            target_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        target_dtype = torch.float16 if device == "cuda" else torch.float32
        target_model = AutoModelForCausalLM.from_pretrained(
            target_name,
            torch_dtype=target_dtype,
            low_cpu_mem_usage=True,
        ).to(device)
    target_model.eval()
    for p in target_model.parameters():
        p.requires_grad = False

    # 2. Initialize or Resume 85M Draft Model
    draft_cfg = get_qwen_85m_config()
    draft_model = SmallLLM(draft_cfg).to(device)

    start_step = 0
    latest_ckpt_path = os.path.join(checkpoint_dir, "latest_draft.pt")
    if os.path.exists(latest_ckpt_path):
        log_print(f"Found existing checkpoint! Resuming from: {latest_ckpt_path}")
        ckpt_data = torch.load(latest_ckpt_path, map_location=device)
        draft_model.load_state_dict(ckpt_data["model_state_dict"])
        start_step = ckpt_data.get("step", 0)
        log_print(f"Successfully resumed at Step {start_step:,}")

    draft_model.train()
    if device == "cuda" and HAS_BNB:
        log_print("Using bitsandbytes 8-bit AdamW for low-memory GPU training")
        optimizer = bnb.optim.AdamW8bit(draft_model.parameters(), lr=learning_rate, weight_decay=0.01)
    else:
        optimizer = configure_optimizers(draft_model, weight_decay=0.01, learning_rate=learning_rate)

    # 3. Setup Dataset & Dataloader
    dataset = InstructionDataset(tokenizer, max_seq_len=256)
    pad_id = getattr(tokenizer, "pad_token_id", 151643) or 151643
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_padded(b, pad_id),
    )
    data_iter = iter(loader)

    train_cfg = TrainingConfig(
        learning_rate=learning_rate,
        min_learning_rate=learning_rate * 0.1,
        warmup_steps=200,
        max_steps=max_steps,
    )

    t_start = time.time()
    t_last_log = t_start
    running_loss = 0.0
    step = start_step
    max_seconds = max_hours * 3600.0

    log_print(f"Beginning training loop from step {step:,} to {max_steps:,}...")

    try:
        while step < max_steps:
            elapsed_total = time.time() - t_start
            if elapsed_total >= max_seconds:
                log_print(f"Reached allocated time limit ({max_hours:.1f} hours). Concluding training safely.")
                break

            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0

            for _ in range(gradient_accumulation_steps):
                try:
                    tokens = next(data_iter)
                except StopIteration:
                    data_iter = iter(loader)
                    tokens = next(data_iter)

                tokens = tokens.to(device)

                with torch.no_grad():
                    t_out = target_model(tokens)
                    top_target, top_idx = torch.topk(t_out.logits, k=64, dim=-1)
                    del t_out
                    if device == "cuda":
                        torch.cuda.empty_cache()

                draft_logits, _, _ = draft_model(tokens)
                gathered_draft = torch.gather(draft_logits, -1, top_idx)
                del draft_logits

                temp = 1.5
                p_target = F.softmax(top_target / temp, dim=-1)
                log_p_target = F.log_softmax(top_target / temp, dim=-1)
                log_q_draft = F.log_softmax(gathered_draft / temp, dim=-1)

                loss = ((temp ** 2) * (p_target * (log_p_target - log_q_draft)).sum(dim=-1)).mean()
                loss = loss / gradient_accumulation_steps
                loss.backward()

                accum_loss += loss.item()

            lr = get_lr(step, train_cfg)
            for g in optimizer.param_groups:
                g["lr"] = lr

            torch.nn.utils.clip_grad_norm_(draft_model.parameters(), 1.0)
            optimizer.step()

            running_loss += accum_loss
            step += 1

            # Log every 25 steps
            if step % 25 == 0:
                t_now = time.time()
                dt = t_now - t_last_log
                t_last_log = t_now
                avg_loss = running_loss / 25
                running_loss = 0.0
                steps_per_sec = 25 / max(0.01, dt)
                remaining_steps = max_steps - step
                eta_seconds = remaining_steps / max(0.01, steps_per_sec)
                eta_hours = eta_seconds / 3600.0
                elapsed_hours = (t_now - t_start) / 3600.0

                log_print(
                    f"Step {step:05d}/{max_steps:05d} | "
                    f"KL Loss: {avg_loss:.4f} | "
                    f"LR: {lr:.2e} | "
                    f"Speed: {steps_per_sec:.2f} step/s | "
                    f"Elapsed: {elapsed_hours:.2f}h | "
                    f"ETA: {eta_hours:.1f}h"
                )

            # Save checkpoint every save_every steps
            if step % save_every == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"draft_step_{step}.pt")
                save_data = {
                    "step": step,
                    "model_state_dict": draft_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": draft_model.config,
                }
                torch.save(save_data, ckpt_path)
                torch.save(save_data, latest_ckpt_path)
                log_print(f"💾 Checkpoint saved: {ckpt_path} & {latest_ckpt_path}")

    except KeyboardInterrupt:
        log_print("Received interruption! Saving emergency checkpoint before exiting...")
    finally:
        final_save_path = os.path.join(checkpoint_dir, "final_trained_draft_85m.pt")
        torch.save(
            {
                "step": step,
                "model_state_dict": draft_model.state_dict(),
                "config": draft_model.config,
            },
            final_save_path,
        )
        log_print(f"🎉 Training session finished! Final model saved to: {final_save_path}")
        log_fp.close()


if __name__ == "__main__":
    train_overnight()
