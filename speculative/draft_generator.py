"""
Autoregressive Draft Token Generator with Confidence-Gated Early Stopping.
"""

from typing import List, Optional, Tuple, Any
import torch
import torch.nn.functional as F

from .cache_manager import KVCacheManager
from .rejection_sampler import RejectionSampler


class DraftGenerator:
    """
    Autoregressively drafts K tokens using the small draft model.
    Supports confidence gating to avoid drafting uncertain tokens.
    """

    @staticmethod
    def draft(
        draft_model: Any,
        start_tokens: torch.Tensor,  # (1, 1) or current last token
        draft_cache: Any,
        K: int = 5,
        temperature: float = 0.0,
        top_p: float = 1.0,
        confidence_threshold: Optional[float] = None,
        eos_token_id: Optional[int] = None,
    ) -> Tuple[List[int], torch.Tensor, Any]:
        """
        Generates up to K draft tokens.

        Returns:
            draft_tokens: List[int] of drafted token IDs
            draft_probs: torch.Tensor of shape (actual_K, vocab_size)
            draft_cache: updated KV cache for draft model
        """
        draft_tokens: List[int] = []
        draft_probs_list: List[torch.Tensor] = []
        curr_input = start_tokens
        device = start_tokens.device

        for step in range(K):
            # Forward pass through draft model
            # Check if model is Hugging Face or native SmallLLM
            if hasattr(draft_model, "forward"):
                # Check signature or use kwargs
                if hasattr(draft_model, "config") and hasattr(draft_model, "layers") and hasattr(draft_model, "tok_embeddings"):
                    # Native SmallLLM
                    start_pos = KVCacheManager.get_seq_len(draft_cache)
                    logits, _, draft_cache = draft_model(
                        curr_input,
                        kv_caches=draft_cache,
                        use_cache=True,
                        start_pos=start_pos,
                    )
                else:
                    # Hugging Face causal LM
                    out = draft_model(
                        input_ids=curr_input,
                        past_key_values=draft_cache,
                        use_cache=True,
                    )
                    logits = out.logits
                    draft_cache = out.past_key_values
            else:
                raise ValueError("Unsupported draft model type")

            next_logits = logits[0, -1, :]  # Shape: (vocab_size,)

            # Compute probability distribution
            if temperature > 0.0:
                scaled_logits = next_logits / temperature
                probs = F.softmax(scaled_logits, dim=-1)
            else:
                probs = F.softmax(next_logits, dim=-1)

            max_prob, top_token_id = torch.max(probs, dim=-1)
            max_prob_val = max_prob.item()

            # Confidence-gated early stopping
            if confidence_threshold is not None and step > 0:
                if max_prob_val < confidence_threshold:
                    # Confidence is too low; stop drafting to avoid rejection penalties
                    break

            # Sample next token
            next_token = RejectionSampler.sample_token(probs, temperature=temperature, top_p=top_p)

            draft_tokens.append(next_token)
            draft_probs_list.append(probs.unsqueeze(0))

            if eos_token_id is not None and next_token == eos_token_id:
                # Stop if EOS is drafted
                break

            curr_input = torch.tensor([[next_token]], dtype=torch.long, device=device)

        if len(draft_tokens) == 0:
            draft_probs_tensor = torch.empty((0, next_logits.shape[-1]), device=device)
        else:
            draft_probs_tensor = torch.cat(draft_probs_list, dim=0)

        return draft_tokens, draft_probs_tensor, draft_cache
