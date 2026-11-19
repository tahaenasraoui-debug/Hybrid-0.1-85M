"""
Tokenizer module for SmallLLM.

Provides:
1. ByteTokenizer: Pure Python byte-level tokenizer with ZERO external dependencies.
   Encodes any UTF-8 string into byte tokens (256 raw bytes + special tokens).
2. TiktokenTokenizer / PretrainedTokenizer: Optional adapters if tiktoken or transformers is installed.
"""

from typing import List, Optional


class ByteTokenizer:
    """
    Self-contained byte-level tokenizer.
    Can tokenize ANY unicode text into UTF-8 bytes without out-of-vocabulary errors.
    """
    PAD_TOKEN = "<|pad|>"
    BOS_TOKEN = "<|bos|>"
    EOS_TOKEN = "<|eos|>"
    UNK_TOKEN = "<|unk|>"

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.special_tokens = {
            self.PAD_TOKEN: 0,
            self.BOS_TOKEN: 1,
            self.EOS_TOKEN: 2,
            self.UNK_TOKEN: 3,
        }
        self.special_tokens_inv = {v: k for k, v in self.special_tokens.items()}
        self.byte_offset = len(self.special_tokens)  # bytes start at 4

    @property
    def pad_id(self) -> int:
        return self.special_tokens[self.PAD_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.special_tokens[self.BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.special_tokens[self.EOS_TOKEN]

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        byte_bytes = text.encode("utf-8")
        tokens = [b + self.byte_offset for b in byte_bytes]
        if add_bos:
            tokens = [self.bos_id] + tokens
        if add_eos:
            tokens = tokens + [self.eos_id]
        return tokens

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        raw_bytes = bytearray()
        for t in token_ids:
            if t in self.special_tokens_inv:
                if not skip_special_tokens:
                    raw_bytes.extend(self.special_tokens_inv[t].encode("utf-8"))
            elif self.byte_offset <= t < self.byte_offset + 256:
                raw_bytes.append(t - self.byte_offset)
            else:
                if not skip_special_tokens:
                    raw_bytes.extend(b"<unk>")
        return raw_bytes.decode("utf-8", errors="replace")


def get_tokenizer(name_or_path: Optional[str] = None, vocab_size: int = 32000):
    """
    Factory function: returns Hugging Face tokenizer if installed/specified,
    otherwise falls back to native ByteTokenizer.
    """
    if name_or_path is not None:
        try:
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(name_or_path)
        except ImportError:
            print("transformers not installed. Falling back to built-in ByteTokenizer.")
    return ByteTokenizer(vocab_size=vocab_size)
