"""
Dataset utilities for training SmallLLM.
"""

from typing import Tuple
import torch
from torch.utils.data import Dataset
from tokenizer import ByteTokenizer


class TextDataset(Dataset):
    """
    Tokenizes a text file and slices it into context windows of size seq_len.
    """
    def __init__(self, file_path_or_text: str, tokenizer: ByteTokenizer, seq_len: int = 512, is_file: bool = True):
        self.seq_len = seq_len
        self.tokenizer = tokenizer

        if is_file:
            with open(file_path_or_text, "r", encoding="utf-8") as f:
                raw_text = f.read()
        else:
            raw_text = file_path_or_text

        print(f"Tokenizing text of length {len(raw_text):,} characters...")
        self.tokens = torch.tensor(self.tokenizer.encode(raw_text), dtype=torch.long)
        print(f"Total tokens: {len(self.tokens):,}")

    def __len__(self) -> int:
        # We need seq_len + 1 tokens for input + target shift
        return max(0, (len(self.tokens) - 1) // self.seq_len)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        end = start + self.seq_len
        x = self.tokens[start:end]
        y = self.tokens[start + 1 : end + 1]
        return x, y


class SyntheticDataset(Dataset):
    """
    Synthetic dataset for profiling, testing training throughput, and sanity checks.
    """
    def __init__(self, vocab_size: int = 4096, seq_len: int = 512, num_samples: int = 1000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Generate random tokens
        data = torch.randint(0, self.vocab_size, (self.seq_len + 1,), dtype=torch.long)
        x = data[:-1]
        y = data[1:]
        return x, y
