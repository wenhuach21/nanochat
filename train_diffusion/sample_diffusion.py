"""
LLaDA-style iterative sampler for the diffusion Qwen3 model.

Generation starts from an all-[MASK] block and iteratively unmasks the most
confident tokens over `steps` denoising steps, optionally conditioned on a prompt.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def generate(
    model,
    prompt_ids: torch.LongTensor,
    gen_len: int = 64,
    steps: int = 64,
    temperature: float = 0.0,
):
    """Generate `gen_len` tokens after the prompt via masked diffusion.

    prompt_ids: [1, P] token ids (may be empty). Returns [1, P+gen_len].
    """
    device = next(model.parameters()).device
    mask_id = model.mask_token_id
    P = prompt_ids.shape[1]
    x = torch.full((1, P + gen_len), mask_id, dtype=torch.long, device=device)
    x[:, :P] = prompt_ids.to(device)

    gen_slice = slice(P, P + gen_len)
    steps = max(1, min(steps, gen_len))
    per_step = max(1, gen_len // steps)

    while (x[:, gen_slice] == mask_id).any():
        logits = model.logits(x).float()
        if temperature > 0:
            probs = F.softmax(logits / temperature, dim=-1)
            pred = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(1, -1)
            conf = probs.view(-1, probs.size(-1)).gather(1, pred.view(-1, 1)).view(1, -1)
        else:
            conf, pred = logits.max(dim=-1)

        masked = x == mask_id
        masked[:, :P] = False  # never touch the prompt
        conf = conf.masked_fill(~masked, -1e9)

        k = min(per_step, int(masked.sum().item()))
        if k == 0:
            break
        idx = conf.topk(k, dim=-1).indices
        x.scatter_(1, idx, pred.gather(1, idx))
    return x

