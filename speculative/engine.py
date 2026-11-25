"""
Deprecated compatibility shim for the legacy SpeculativeDecodingEngine API.

The canonical speculative decoding engine lives in ``src.speculative.SpeculativeEngine``
(and ``src.speculative.speculative_generate``). This module delegates all execution to
``SpeculativeEngine`` so there is a single source of truth for speculative decoding.
"""

import warnings
from typing import Any, Dict, List, Optional, Union

import torch


class SpeculativeDecodingEngine:
    """
    Deprecated compatibility wrapper around ``src.speculative.SpeculativeEngine``.

    Preserves the legacy constructor argument order ``(draft_model, target_model, ...)``
    and ``generate(..., K=...)`` return keys while delegating all speculative decoding
    logic to ``src.speculative.SpeculativeEngine``.
    """

    def __init__(
        self,
        draft_model: Any,
        target_model: Any,
        tokenizer: Any,
        device: Optional[Union[str, torch.device]] = "cpu",
        dtype: torch.dtype = torch.float32,
        max_seq_len: int = 2048,
    ):
        warnings.warn(
            "speculative.engine.SpeculativeDecodingEngine is deprecated; "
            "use src.speculative.SpeculativeEngine instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Lazy import avoids a circular import because src.speculative imports
        # speculative.cache_manager.KVCacheManager.
        from src.speculative import SpeculativeEngine

        self._engine = SpeculativeEngine(
            target_model=target_model,
            draft_model=draft_model,
            tokenizer=tokenizer,
            device=device,
            max_seq_len=max_seq_len,
        )
        self.draft_model = self._engine.draft_model
        self.target_model = self._engine.target_model
        self.tokenizer = self._engine.tokenizer
        self.device = self._engine.device
        self.dtype = dtype
        self.eos_token_id = self._engine.eos_token_id

    @torch.no_grad()
    def generate(
        self,
        prompt: Union[str, List[int]],
        max_new_tokens: int = 128,
        K: int = 5,
        k: Optional[int] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        confidence_threshold: Optional[float] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Delegate generation to ``src.speculative.SpeculativeEngine.generate``."""
        del verbose  # Preserved for legacy call-site compatibility.
        lookahead = k if k is not None else K
        result = self._engine.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            k=lookahead,
            temperature=temperature,
            top_p=top_p,
            confidence_threshold=confidence_threshold,
        )
        # Include legacy return keys alongside canonical SpeculativeEngine keys.
        return {
            **result,
            "generated_text": result["text"],
            "mean_accepted_per_round": result["mean_accepted_per_step"],
            "wall_time_ms": result["wall_time_s"] * 1000.0,
        }
