#!/usr/bin/env python3
"""
Interactive Speculative Decoding Chat with Qwen + 85M Draft Model.

Usage:
  python chat_speculative.py
  python chat_speculative.py --prompt "Why is the sky blue?"
  python chat_speculative.py --target Qwen/Qwen2.5-0.5B-Instruct --k 5
"""

import argparse
import os
import sys
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from config import get_qwen_85m_config
from model import SmallLLM
from src.speculative import SpeculativeEngine
from train import get_compatible_device


def main():
    parser = argparse.ArgumentParser(description="Speculative Decoding Chat with Qwen and 85M Draft")
    parser.add_argument("--target", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Target teacher model")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints_overnight/final_trained_draft_85m.pt", help="85M draft checkpoint")
    parser.add_argument("--k", type=int, default=5, help="Draft length K (default: 5)")
    parser.add_argument("--temp", type=float, default=0.0, help="Sampling temperature (0.0 for greedy)")
    parser.add_argument("--max_tokens", type=int, default=128, help="Max new tokens to generate")
    parser.add_argument("--prompt", type=str, default="", help="Single prompt mode (leave empty for interactive chat)")
    args = parser.parse_args()

    device = str(get_compatible_device())
    print(f"============================================================")
    print(f"  Speculative Decoding Engine: 85M Draft Model + {args.target}")
    print(f"  Device: {device} | Draft Window K = {args.k}")
    print(f"============================================================")

    # 1. Load Tokenizer
    print(f"\n[1/3] Loading Tokenizer ({args.target})...")
    tokenizer = AutoTokenizer.from_pretrained(args.target, trust_remote_code=True)

    # 2. Load Target Model (Qwen)
    print(f"[2/3] Loading Target Model ({args.target})...")
    if device == "cuda":
        try:
            bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            target_model = AutoModelForCausalLM.from_pretrained(
                args.target,
                quantization_config=bnb_config,
                device_map="auto",
            )
        except Exception:
            target_model = AutoModelForCausalLM.from_pretrained(
                args.target,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            ).to(device)
    else:
        target_model = AutoModelForCausalLM.from_pretrained(
            args.target,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        ).to(device)
    target_model.eval()

    # 3. Load Draft Model (85M)
    print(f"[3/3] Loading 85M Draft Model...")
    draft_cfg = get_qwen_85m_config()
    draft_model = SmallLLM(draft_cfg).to(device)

    if os.path.exists(args.checkpoint):
        print(f" -> Found trained weights at {args.checkpoint}! Loading checkpoint...")
        ckpt = torch.load(args.checkpoint, map_location=device)
        draft_model.load_state_dict(ckpt["model_state_dict"])
        step_num = ckpt.get("step", "unknown")
        print(f" -> Checkpoint loaded successfully (Trained for {step_num} steps)!")
    else:
        print(f" -> Note: No checkpoint found at {args.checkpoint} yet. Using initialized weights.")

    draft_model.eval()

    # 4. Initialize Speculative Decoding Engine
    engine = SpeculativeEngine(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        device=device,
    )
    print("\nEngine initialized and ready!")

    def run_generation(user_text: str):
        messages = [{"role": "user", "content": user_text}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        print("\nThinking and generating...", flush=True)
        t0 = time.perf_counter()
        result = engine.generate(
            prompt=formatted_prompt,
            max_new_tokens=args.max_tokens,
            k=args.k,
            temperature=args.temp,
            confidence_threshold=0.5,
        )
        total_time = result.get("wall_time_s", time.perf_counter() - t0)

        tokens_generated = result["num_generated"]
        tok_per_sec = result["tokens_per_sec"]

        print("-" * 60)
        print("Response:")
        print(result["text"])
        print("-" * 60)
        print(f"Speculative Decoding Stats:")
        print(f" • Tokens Generated:     {tokens_generated} tokens")
        print(f" • Total Time:           {total_time:.2f}s ({tok_per_sec:.1f} tok/s)")
        print(f" • Verification Passes:  {result['target_forward_passes']}")
        print(f" • Token Acceptance (τ): {result['mean_accepted_per_step']:.2f} tokens / verification step")
        print(f" • Acceptance Rate:      {result['acceptance_rate'] * 100:.1f}%")
        print("-" * 60 + "\n")

    # Single prompt or interactive loop
    if args.prompt:
        run_generation(args.prompt)
    else:
        print("Entering interactive mode. Type your message and hit Enter (or 'quit' to exit):")
        while True:
            try:
                user_input = input("\nYou: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("quit", "exit"):
                    print("Goodbye!")
                    break
                run_generation(user_input)
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                break


if __name__ == "__main__":
    main()
