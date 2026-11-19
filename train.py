"""
Pre-training script for SmallLLM.

Features:
- AdamW with decoupled weight decay (0 weight decay for RMSNorm/biases).
- Cosine Annealing learning rate schedule with linear warmup.
- Mixed precision training (bfloat16 / float16) via torch.amp.
- Gradient accumulation for scalable effective batch size.
- Gradient norm clipping to prevent gradient explosion.
- Step-by-step throughput (tokens/sec) and loss tracking.
- Model checkpointing.
"""

import math
import os
import time
from typing import Optional
import torch
from torch.utils.data import DataLoader

from config import ModelConfig, TrainingConfig, get_gtx1050_config, get_tiny_config, get_1b_config
from dataset import SyntheticDataset, TextDataset
from model import SmallLLM
from tokenizer import ByteTokenizer


def configure_optimizers(model: SmallLLM, weight_decay: float, learning_rate: float, betas=(0.9, 0.95)):
    """
    Separates parameters into decay and no-decay groups.
    2D parameters (matrix multiplications) get weight decay.
    1D parameters (RMSNorm, biases) get 0 weight decay.
    """
    decay_params = []
    nodecay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            nodecay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=1e-8)
    return optimizer


def get_lr(step: int, config: TrainingConfig) -> float:
    """Computes learning rate with linear warmup and cosine decay."""
    # 1) Linear warmup
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    # 2) If step > max_steps, return min learning rate
    if step > config.max_steps:
        return config.min_learning_rate
    # 3) Cosine decay down to min_learning_rate
    decay_ratio = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    assert 0.0 <= decay_ratio <= 1.0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_learning_rate + coeff * (config.learning_rate - config.min_learning_rate)


def get_compatible_device() -> torch.device:
    """Checks if CUDA is available AND supported by the installed PyTorch build."""
    if torch.cuda.is_available():
        try:
            # Test a small tensor operation to verify kernel support (e.g. avoiding sm_61 issues on cu130)
            t = torch.zeros(1, device="cuda")
            _ = t + 1
            return torch.device("cuda")
        except Exception as e:
            print(f"CUDA device detected but unsupported by current PyTorch binary ({e}). Falling back to CPU.")
            return torch.device("cpu")
    return torch.device("cpu")


def train(
    model: SmallLLM,
    dataset,
    train_cfg: TrainingConfig,
    checkpoint_dir: str = "./checkpoints",
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = get_compatible_device()
    print(f"Using device: {device}")

    model.to(device)
    model.train()

    optimizer = configure_optimizers(
        model,
        weight_decay=train_cfg.weight_decay,
        learning_rate=train_cfg.learning_rate,
    )

    # Use AMP for float16 / bfloat16 on CUDA
    use_amp = (device.type == "cuda") and (train_cfg.precision in ["bfloat16", "float16"])
    dtype = torch.bfloat16 if train_cfg.precision == "bfloat16" and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and dtype == torch.float16))

    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        drop_last=True,
    )

    data_iter = iter(dataloader)
    tokens_per_step = train_cfg.batch_size * model.config.max_seq_len * train_cfg.gradient_accumulation_steps
    print(f"Tokens per optimizer step: {tokens_per_step:,}")
    print(f"Total model parameters: {model.count_parameters()['total_millions']:.2f}M")

    step = 0
    running_loss = 0.0
    t0 = time.time()

    while step < train_cfg.max_steps:
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro_step in range(train_cfg.gradient_accumulation_steps):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                x, y = next(data_iter)

            x, y = x.to(device), y.to(device)

            if use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=dtype):
                    _, loss, _ = model(x, targets=y)
                    loss = loss / train_cfg.gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                _, loss, _ = model(x, targets=y)
                loss = loss / train_cfg.gradient_accumulation_steps
                loss.backward()

            accum_loss += loss.item()

        # Update learning rate schedule
        lr = get_lr(step, train_cfg)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Unscale and clip gradients
        if use_amp and scaler.is_enabled():
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.clip_grad_norm)
            optimizer.step()

        running_loss += accum_loss
        step += 1

        if step % 10 == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            avg_loss = running_loss / 10
            running_loss = 0.0
            tok_per_sec = (tokens_per_step * 10) / dt
            vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if device.type == "cuda" else 0
            print(
                f"Step {step:05d}/{train_cfg.max_steps} | "
                f"Loss: {avg_loss:.4f} | "
                f"LR: {lr:.2e} | "
                f"Throughput: {tok_per_sec:,.0f} tok/s | "
                f"VRAM: {vram_mb:.0f}MB"
            )

        if step % train_cfg.save_every == 0 or step == train_cfg.max_steps:
            ckpt_path = os.path.join(checkpoint_dir, f"model_step_{step}.pt")
            print(f"--> Saving checkpoint: {ckpt_path}")
            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": model.config,
                },
                ckpt_path,
            )

    print("Training finished successfully!")


if __name__ == "__main__":
    # Default to 87.5M 10-layer model optimized for GTX 1050
    print("Initializing 85M (10 layers) SmallLLM...")
    cfg = get_gtx1050_config()
    model = SmallLLM(cfg)

    train_cfg = TrainingConfig(
        batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=5e-4,
        warmup_steps=10,
        max_steps=30,
        device="cuda" if torch.cuda.is_available() else "cpu",
        precision="float16" if torch.cuda.is_available() else "float32",
    )

    dataset = SyntheticDataset(
        vocab_size=cfg.vocab_size,
        seq_len=cfg.max_seq_len,
        num_samples=200,
    )

    train(model, dataset, train_cfg)
