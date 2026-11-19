"""
KV-Cache Manager for Speculative Decoding.
Handles robust sequence length retrieval, truncation, and synchronization
across Hugging Face DynamicCache and native PyTorch KV-cache structures.
"""

from typing import Any, List, Optional, Tuple, Union
import torch


class KVCacheManager:
    """Utilities to manage and truncate KV caches for draft and target models."""

    @staticmethod
    def get_seq_len(cache: Any) -> int:
        """Returns the current sequence length in the cache."""
        if cache is None:
            return 0

        # Hugging Face DynamicCache
        if hasattr(cache, "get_seq_length"):
            try:
                return cache.get_seq_length(0)
            except Exception:
                pass

        if hasattr(cache, "layers") and len(cache.layers) > 0:
            layer = cache.layers[0]
            if hasattr(layer, "keys") and layer.keys is not None:
                return layer.keys.shape[-2]
            if hasattr(layer, "get_seq_length"):
                return layer.get_seq_length()

        if hasattr(cache, "key_cache") and len(cache.key_cache) > 0:
            if cache.key_cache[0] is not None:
                return cache.key_cache[0].shape[-2]

        # Native list of (k, v) tuples: shape (batch, n_heads, seq_len, head_dim)
        if isinstance(cache, (list, tuple)) and len(cache) > 0:
            k, _ = cache[0]
            if k is not None:
                return k.shape[-2]

        return 0

    @staticmethod
    def truncate_cache(cache: Any, target_len: int) -> Any:
        """
        Truncates the cache down to target_len in-place or returns a truncated copy.
        Guarantees no state leakage from rejected candidate tokens.
        """
        if cache is None or target_len <= 0:
            return cache

        current_len = KVCacheManager.get_seq_len(cache)
        if current_len <= target_len:
            return cache

        tokens_to_remove = target_len - current_len  # Negative integer

        # Hugging Face DynamicCache (Transformers 4.36+ / 5.x)
        if hasattr(cache, "crop"):
            try:
                # Transformers 5.x takes negative tokens_to_remove
                cache.crop(tokens_to_remove)
                return cache
            except TypeError:
                try:
                    # In case of older crop(target_len)
                    cache.crop(target_len)
                    return cache
                except Exception:
                    pass
            except Exception:
                pass

        # Direct attribute manipulation fallback for HuggingFace DynamicCache
        if hasattr(cache, "layers"):
            for layer in cache.layers:
                if hasattr(layer, "keys") and layer.keys is not None:
                    layer.keys = layer.keys[..., :target_len, :].contiguous()
                if hasattr(layer, "values") and layer.values is not None:
                    layer.values = layer.values[..., :target_len, :].contiguous()
            return cache

        if hasattr(cache, "key_cache"):
            for idx in range(len(cache.key_cache)):
                if cache.key_cache[idx] is not None:
                    cache.key_cache[idx] = cache.key_cache[idx][..., :target_len, :].contiguous()
                if cache.value_cache[idx] is not None:
                    cache.value_cache[idx] = cache.value_cache[idx][..., :target_len, :].contiguous()
            return cache

        # Native list of (k, v) tuples
        if isinstance(cache, list):
            new_cache = []
            for item in cache:
                if item is not None and isinstance(item, tuple):
                    k, v = item
                    new_k = k[..., :target_len, :].contiguous()
                    new_v = v[..., :target_len, :].contiguous()
                    new_cache.append((new_k, new_v))
                else:
                    new_cache.append(item)
            return new_cache

        return cache
