"""
Text generation engine for SmallLLM with KV-caching, top-k, and nucleus (top-p) sampling.
"""

from typing import List, Optional
import torch
import torch.nn.functional as F

from model import SmallLLM
from tokenizer import ByteTokenizer


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    generated_tokens: Optional[List[int]] = None,
) -> int:
    """
    Applies temperature, repetition penalty, top-k, and top-p filtering to logits,
    then samples a token.
    """
    logits = logits.clone()

    # Repetition penalty
    if repetition_penalty != 1.0 and generated_tokens:
        for prev_token in set(generated_tokens):
            if logits[prev_token] < 0:
                logits[prev_token] *= repetition_penalty
            else:
                logits[prev_token] /= repetition_penalty

    # Temperature scaling
    if temperature <= 0.0 or temperature < 1e-5:
        # Greedy decoding
        return int(torch.argmax(logits).item())

    logits = logits / temperature

    # Top-k filtering
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1, None]
        logits[indices_to_remove] = -float("Inf")

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep the first token above threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = -float("Inf")

    # Sample from probability distribution
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return int(next_token.item())


@torch.no_grad()
def generate(
    model: SmallLLM,
    tokenizer: ByteTokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    device: str = "cpu",
    stream: bool = False,
) -> str:
    """
    Generates text autoregressively using KV caching for maximum efficiency.
    """
    model.eval()
    model.to(device)

    # Encode initial prompt
    token_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)

    # Step 1: Prefill phase (process whole prompt, populate KV caches)
    logits, _, kv_caches = model(input_tensor, use_cache=True, start_pos=0)
    last_logit = logits[0, -1, :]

    next_tok = sample_next_token(
        last_logit,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        generated_tokens=token_ids,
    )
    token_ids.append(next_tok)

    if stream:
        print(prompt, end="", flush=True)
        print(tokenizer.decode([next_tok]), end="", flush=True)

    # Step 2: Incremental decoding phase with KV-cache (process only 1 token per step)
    for _step in range(max_new_tokens - 1):
        if next_tok == tokenizer.eos_id:
            break

        curr_token_tensor = torch.tensor([[next_tok]], dtype=torch.long, device=device)
        start_pos = len(token_ids) - 1

        logits, _, kv_caches = model(
            curr_token_tensor,
            kv_caches=kv_caches,
            use_cache=True,
            start_pos=start_pos,
        )
        last_logit = logits[0, -1, :]

        next_tok = sample_next_token(
            last_logit,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            generated_tokens=token_ids,
        )
        token_ids.append(next_tok)

        if stream:
            print(tokenizer.decode([next_tok]), end="", flush=True)

    if stream:
        print()

    return tokenizer.decode(token_ids)


if __name__ == "__main__":
    from config import get_gtx1050_config

    cfg = get_gtx1050_config()
    model = SmallLLM(cfg)
    tokenizer = ByteTokenizer(vocab_size=cfg.vocab_size)

    test_prompt = "Once upon a time in a distant realm,"
    print(f"Generating from prompt: '{test_prompt}'")
    out = generate(model, tokenizer, test_prompt, max_new_tokens=30, temperature=0.8, stream=True)
    print("\nGeneration complete!")
