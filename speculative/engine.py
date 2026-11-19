"""
Speculative Decoding Engine.
Orchestrates fast draft generation, single parallel target verification,
lossless rejection sampling, and strict KV-cache synchronization.
"""

import time
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F

from .cache_manager import KVCacheManager
from .rejection_sampler import RejectionSampler, SpeculativeSampleResult


class SpeculativeDecodingEngine:
    """
    Production-grade, lossless Speculative Decoding Engine.
    Compatible with native SmallLLM draft models and Hugging Face target models.
    """

    def __init__(
        self,
        draft_model: Any,
        target_model: Any,
        tokenizer: Any,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.draft_model = draft_model.to(device)
        self.target_model = target_model.to(device)
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.dtype = dtype

        self.draft_model.eval()
        self.target_model.eval()

        self.eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if self.eos_token_id is None and hasattr(draft_model, "config"):
            self.eos_token_id = getattr(draft_model.config, "eos_token_id", None)

    @torch.no_grad()
    def _forward_model(self, model: Any, input_ids: torch.Tensor, cache: Any) -> Tuple[torch.Tensor, Any]:
        """Unified forward pass handler for SmallLLM and HuggingFace models."""
        if hasattr(model, "config") and hasattr(model, "layers") and hasattr(model, "tok_embeddings"):
            # Native SmallLLM
            start_pos = KVCacheManager.get_seq_len(cache)
            logits, _, cache = model(
                input_ids,
                kv_caches=cache,
                use_cache=True,
                start_pos=start_pos,
            )
            return logits, cache
        else:
            # Hugging Face model
            out = model(
                input_ids=input_ids,
                past_key_values=cache,
                use_cache=True,
            )
            return out.logits, out.past_key_values

    @torch.no_grad()
    def generate(
        self,
        prompt: Union[str, List[int]],
        max_new_tokens: int = 128,
        K: int = 5,
        temperature: float = 0.0,
        top_p: float = 1.0,
        confidence_threshold: Optional[float] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes lossless speculative decoding matching target model autoregression.
        """
        t_start = time.perf_counter()

        # 1. Tokenize prompt
        if isinstance(prompt, str):
            if hasattr(self.tokenizer, "encode"):
                prompt_ids = self.tokenizer.encode(prompt)
            else:
                prompt_ids = self.tokenizer(prompt)["input_ids"]
        else:
            prompt_ids = list(prompt)

        input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        # 2. Prefill Phase
        target_logits, target_cache = self._forward_model(self.target_model, input_tensor, None)
        draft_logits, draft_cache = self._forward_model(self.draft_model, input_tensor, None)

        curr_target_logits = target_logits[0, -1, :]
        curr_draft_logits = draft_logits[0, -1, :]

        generated_ids: List[int] = []
        total_drafted = 0
        total_accepted = 0
        target_forward_passes = 1  # 1 prefill pass

        while len(generated_ids) < max_new_tokens:
            base_len = KVCacheManager.get_seq_len(target_cache)
            actual_k = min(K, max_new_tokens - len(generated_ids))
            if actual_k <= 0:
                break

            # -------------------------------------------------------------
            # Step A: Draft K tokens autoregressively
            # -------------------------------------------------------------
            draft_tokens: List[int] = []
            draft_probs_list: List[torch.Tensor] = []
            step_draft_logits = curr_draft_logits

            for step in range(actual_k):
                # Compute draft probability distribution
                if temperature > 0.0:
                    step_probs = F.softmax(step_draft_logits / temperature, dim=-1)
                else:
                    step_probs = F.softmax(step_draft_logits, dim=-1)

                max_prob, _ = torch.max(step_probs, dim=-1)

                # Confidence gating: stop early if draft model is uncertain
                if confidence_threshold is not None and step > 0:
                    if max_prob.item() < confidence_threshold:
                        break

                d_tok = RejectionSampler.sample_token(step_probs, temperature=temperature, top_p=top_p)
                draft_tokens.append(d_tok)
                draft_probs_list.append(step_probs.unsqueeze(0))

                if self.eos_token_id is not None and d_tok == self.eos_token_id:
                    break

                # Forward d_tok through draft model to get logits for next token
                d_tensor = torch.tensor([[d_tok]], dtype=torch.long, device=self.device)
                out_draft_logits, draft_cache = self._forward_model(self.draft_model, d_tensor, draft_cache)
                step_draft_logits = out_draft_logits[0, -1, :]

            actual_draft_len = len(draft_tokens)
            if actual_draft_len == 0:
                # Fallback: sample directly from target
                target_prob = F.softmax(curr_target_logits / temperature, dim=-1) if temperature > 0 else curr_target_logits
                next_tok = RejectionSampler.sample_token(target_prob, temperature=temperature, top_p=top_p)
                generated_ids.append(next_tok)
                next_tok_tensor = torch.tensor([[next_tok]], dtype=torch.long, device=self.device)
                target_logits, target_cache = self._forward_model(self.target_model, next_tok_tensor, target_cache)
                draft_logits, draft_cache = self._forward_model(self.draft_model, next_tok_tensor, draft_cache)
                curr_target_logits = target_logits[0, -1, :]
                curr_draft_logits = draft_logits[0, -1, :]
                target_forward_passes += 1
                if self.eos_token_id is not None and next_tok == self.eos_token_id:
                    break
                continue

            total_drafted += actual_draft_len
            draft_probs_tensor = torch.cat(draft_probs_list, dim=0)

            # -------------------------------------------------------------
            # Step B: Parallel Target Verification (Single Forward Pass)
            # -------------------------------------------------------------
            draft_tensor = torch.tensor([draft_tokens], dtype=torch.long, device=self.device)
            target_out_logits, target_cache = self._forward_model(self.target_model, draft_tensor, target_cache)
            target_forward_passes += 1

            all_target_logits = torch.cat(
                [curr_target_logits.unsqueeze(0), target_out_logits[0, :, :]], dim=0
            )

            if temperature > 0:
                all_target_probs = F.softmax(all_target_logits / temperature, dim=-1)
            else:
                all_target_probs = all_target_logits

            # -------------------------------------------------------------
            # Step C: Rejection Sampling
            # -------------------------------------------------------------
            sample_result = RejectionSampler.sample(
                draft_tokens=draft_tokens,
                draft_probs=draft_probs_tensor,
                target_probs=all_target_probs,
                temperature=temperature,
                top_p=top_p,
            )

            total_accepted += sample_result.num_accepted
            accepted_batch = sample_result.accepted_tokens

            # Append accepted tokens up to max_new_tokens
            for tok in accepted_batch:
                generated_ids.append(tok)
                if len(generated_ids) >= max_new_tokens:
                    break
                if self.eos_token_id is not None and tok == self.eos_token_id:
                    break

            if len(generated_ids) >= max_new_tokens or (self.eos_token_id is not None and generated_ids[-1] == self.eos_token_id):
                break

            # -------------------------------------------------------------
            # Step D: Strict KV-Cache Synchronization
            # -------------------------------------------------------------
            valid_accepted_prefix = base_len + sample_result.num_accepted
            target_cache = KVCacheManager.truncate_cache(target_cache, valid_accepted_prefix)
            draft_cache = KVCacheManager.truncate_cache(draft_cache, valid_accepted_prefix)

            # Feed final token (replacement token or bonus token) into both models
            final_token = accepted_batch[-1]
            final_token_tensor = torch.tensor([[final_token]], dtype=torch.long, device=self.device)

            target_final_logits, target_cache = self._forward_model(
                self.target_model, final_token_tensor, target_cache
            )
            draft_final_logits, draft_cache = self._forward_model(
                self.draft_model, final_token_tensor, draft_cache
            )
            target_forward_passes += 1

            curr_target_logits = target_final_logits[0, -1, :]
            curr_draft_logits = draft_final_logits[0, -1, :]

        # Guarantee max_new_tokens bound
        if len(generated_ids) > max_new_tokens:
            generated_ids = generated_ids[:max_new_tokens]

        t_end = time.perf_counter()
        wall_time_ms = (t_end - t_start) * 1000.0
        num_generated = len(generated_ids)
        tokens_per_sec = (num_generated / (wall_time_ms / 1000.0)) if wall_time_ms > 0 else 0.0
        tau = (total_accepted / max(1, (target_forward_passes - 1)))

        if hasattr(self.tokenizer, "decode"):
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        else:
            generated_text = str(generated_ids)

        return {
            "generated_ids": generated_ids,
            "generated_text": generated_text,
            "num_generated": num_generated,
            "num_drafted": total_drafted,
            "num_accepted": total_accepted,
            "target_forward_passes": target_forward_passes,
            "acceptance_rate": (total_accepted / max(1, total_drafted)) if total_drafted > 0 else 0.0,
            "mean_accepted_per_round": tau,
            "wall_time_ms": wall_time_ms,
            "tokens_per_sec": tokens_per_sec,
        }
