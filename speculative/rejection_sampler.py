"""
Speculative Rejection Sampler implementing the exact mathematically lossless
algorithm from Leviathan et al. (2023).
"""

from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class SpeculativeSampleResult:
    """Outcome of a speculative verification round."""
    accepted_tokens: List[int]
    num_drafted: int
    num_accepted: int
    all_accepted: bool
    rejection_index: Optional[int]  # None if all accepted, else index 0..K-1 where rejection occurred


class RejectionSampler:
    """
    Executes lossless rejection sampling matching the target model's exact distribution.
    Supports both greedy (temperature=0) and stochastic (temperature > 0) sampling.
    """

    @staticmethod
    def apply_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
        """Apply nucleus (top-p) filtering along the last dimension of a probability tensor."""
        if top_p >= 1.0:
            return probs
        if top_p <= 0.0:
            top_idx = torch.argmax(probs, dim=-1, keepdim=True)
            filtered = torch.zeros_like(probs)
            filtered.scatter_(-1, top_idx, 1.0)
            return filtered

        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cum_probs > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(mask, 0.0)
        filtered = torch.zeros_like(probs).scatter_(-1, sorted_indices, sorted_probs)
        denom = filtered.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return filtered / denom

    @staticmethod
    def sample_token(
        probs: torch.Tensor,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> int:
        """Helper to sample a single token from a probability distribution or logits."""
        if temperature <= 0.0:
            return int(torch.argmax(probs).item())

        probs = RejectionSampler.apply_top_p(probs, top_p)
        return int(torch.multinomial(probs, num_samples=1).item())

    @staticmethod
    def sample(
        draft_tokens: List[int],
        draft_probs: torch.Tensor,   # Shape: (K, vocab_size)
        target_probs: torch.Tensor,  # Shape: (K + 1, vocab_size)
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> SpeculativeSampleResult:
        """
        Runs speculative verification across K drafted tokens.

        draft_tokens: list of length K
        draft_probs: draft model probabilities at positions 0..K-1
        target_probs: target model probabilities at positions 0..K (K+1 positions)
        """
        K = len(draft_tokens)
        accepted_tokens: List[int] = []

        # -------------------------------------------------------------
        # 1. Greedy Verification (Temperature = 0)
        # -------------------------------------------------------------
        if temperature <= 0.0:
            for i in range(K):
                target_token = int(torch.argmax(target_probs[i]).item())
                draft_tok = draft_tokens[i]

                if target_token == draft_tok:
                    # Accepted
                    accepted_tokens.append(draft_tok)
                else:
                    # Rejected at index i: substitute with target model's choice and stop
                    accepted_tokens.append(target_token)
                    return SpeculativeSampleResult(
                        accepted_tokens=accepted_tokens,
                        num_drafted=K,
                        num_accepted=len(accepted_tokens) - 1,  # corrected token does not count as accepted draft
                        all_accepted=False,
                        rejection_index=i,
                    )

            # All K tokens accepted! Sample bonus token from position K
            bonus_token = int(torch.argmax(target_probs[K]).item())
            accepted_tokens.append(bonus_token)
            return SpeculativeSampleResult(
                accepted_tokens=accepted_tokens,
                num_drafted=K,
                num_accepted=K,
                all_accepted=True,
                rejection_index=None,
            )

        # -------------------------------------------------------------
        # 2. Stochastic Lossless Rejection Sampling (Temperature > 0)
        # -------------------------------------------------------------
        filtered_draft_probs = RejectionSampler.apply_top_p(draft_probs, top_p)
        filtered_target_probs = RejectionSampler.apply_top_p(target_probs, top_p)

        for i in range(K):
            tok = draft_tokens[i]
            p_val = filtered_target_probs[i, tok].item()
            q_val = filtered_draft_probs[i, tok].item()

            # Acceptance probability: min(1, p(x) / q(x))
            alpha = min(1.0, p_val / max(q_val, 1e-12))
            r = torch.rand(1).item()

            if r < alpha:
                # Accept token
                accepted_tokens.append(tok)
            else:
                # Reject token: resample from positive residual distribution:
                # P_res(x) = max(0, p(x) - q(x)) / norm
                residual = torch.clamp(filtered_target_probs[i] - filtered_draft_probs[i], min=0.0)
                res_sum = residual.sum().item()

                if res_sum > 1e-12:
                    p_res = residual / res_sum
                    replacement_tok = int(torch.multinomial(p_res, num_samples=1).item())
                else:
                    # Degenerate case fallback: sample directly from filtered target
                    replacement_tok = int(torch.multinomial(filtered_target_probs[i], num_samples=1).item())

                accepted_tokens.append(replacement_tok)
                return SpeculativeSampleResult(
                    accepted_tokens=accepted_tokens,
                    num_drafted=K,
                    num_accepted=len(accepted_tokens) - 1,
                    all_accepted=False,
                    rejection_index=i,
                )

        # All K tokens accepted! Sample bonus token from position K
        bonus_token = int(torch.multinomial(filtered_target_probs[K], num_samples=1).item())
        accepted_tokens.append(bonus_token)
        return SpeculativeSampleResult(
            accepted_tokens=accepted_tokens,
            num_drafted=K,
            num_accepted=K,
            all_accepted=True,
            rejection_index=None,
        )
