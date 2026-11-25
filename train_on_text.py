"""
Demo: Train SmallLLM on a text file and generate before & after text.
"""

import argparse

from config import TrainingConfig, get_gtx1050_config
from dataset import TextDataset
from generate import generate
from model import SmallLLM
from tokenizer import ByteTokenizer
from train import get_compatible_device, train


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
    parser = argparse.ArgumentParser(description="Demo: Train SmallLLM on a text file.")
    parser.add_argument(
        "--data",
        type=str,
        default="sample_data.txt",
        help="Path to training text file (created with sample text if absent)",
    )
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints_demo", help="Checkpoint output directory")
    parser.add_argument("--max_steps", type=int, default=80, help="Training steps")
    args = parser.parse_args()

    data_path = args.data
    with open(data_path, "w", encoding="utf-8") as f:
        f.write(SAMPLE_TEXT)
    print(f"Created sample training data at {data_path} ({len(SAMPLE_TEXT):,} characters).")

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
        max_steps=args.max_steps,
        device=device,
        precision="float32",
    )
    train(model, dataset, train_cfg, checkpoint_dir=args.checkpoint_dir)

    # 4. Test generation AFTER training
    print("\n" + "=" * 50)
    print("✨ 3. Text Generation AFTER Training (Learned Context):")
    print("=" * 50)
    out_after = generate(model, tokenizer, "Once upon a time", max_new_tokens=60, temperature=0.5, device=device)
    print(out_after)
    print("=" * 50)


if __name__ == "__main__":
    main()
