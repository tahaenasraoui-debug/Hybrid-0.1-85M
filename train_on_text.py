"""
Demo: Train SmallLLM on a text file and generate before & after text.
"""

import os
import torch
from config import get_gtx1050_config, get_tiny_config, TrainingConfig
from dataset import TextDataset
from generate import generate
from model import SmallLLM
from tokenizer import ByteTokenizer
from train import train, get_compatible_device


SAMPLE_TEXT = """
Once upon a time, in a small peaceful village surrounded by green hills, there lived a curious girl named Lily.
Lily loved books, ancient maps, and stars. Every evening, she walked up the hill to look through her small telescope.
One night, as the moon shone bright in the sky, Lily saw a faint golden light dancing in the quiet forest.
"What could that be?" she wondered. She took her lantern and walked into the whispering woods.
The path was carpeted with soft moss and glowing mushrooms. Soon, she arrived at a hidden clearing.
In the center of the clearing stood a grand, ancient tree with silver leaves that shimmered like diamonds.
Resting beneath the roots was a small creature with delicate sapphire wings and bright violet eyes.
"Hello, traveler," whispered the creature in a voice like gentle wind chimes. "I have guarded this secret library for centuries."
Around the tree roots were hollow shelves carved into the bark, filled with crystal scrolls containing the lost knowledge of the ancient world.
Lily sat beside the guardian, listening to tales of forgotten stars, deep oceans, and forgotten melodies.
From that magical night onward, Lily returned every week, learning the secrets of the cosmos and sharing stories with her village.
""" * 50  # Repeat to give sufficient tokens for demonstration


def main():
    data_path = "/home/tahary/.gemini/antigravity/scratch/small_llm/sample_data.txt"
    with open(data_path, "w", encoding="utf-8") as f:
        f.write(SAMPLE_TEXT)
    print(f"Created sample training data ({len(SAMPLE_TEXT):,} characters).")

    # 1. Initialize Model & Tokenizer
    cfg = get_gtx1050_config()
    cfg.max_seq_len = 256  # context length
    model = SmallLLM(cfg)
    tokenizer = ByteTokenizer(vocab_size=cfg.vocab_size)
    device = str(get_compatible_device())

    # 2. Test generation BEFORE training (untrained / random weights)
    print("\n" + "=" * 50)
    print("🤖 1. Text Generation BEFORE Training (Random Weights):")
    print("=" * 50)
    out_before = generate(model, tokenizer, "Once upon a time", max_new_tokens=40, temperature=0.7, device=device)
    print(out_before)

    # 3. Train on sample data
    print("\n" + "=" * 50)
    print("🏋️ 2. Training Model on Sample Text...")
    print("=" * 50)
    dataset = TextDataset(data_path, tokenizer, seq_len=cfg.max_seq_len, is_file=True)
    train_cfg = TrainingConfig(
        batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=1e-3,
        warmup_steps=10,
        max_steps=80,
        device=device,
        precision="float32",
    )
    train(model, dataset, train_cfg, checkpoint_dir="./checkpoints_demo")

    # 4. Test generation AFTER training
    print("\n" + "=" * 50)
    print("✨ 3. Text Generation AFTER Training (Learned Context):")
    print("=" * 50)
    out_after = generate(model, tokenizer, "Once upon a time", max_new_tokens=60, temperature=0.5, device=device)
    print(out_after)
    print("=" * 50)


if __name__ == "__main__":
    main()
