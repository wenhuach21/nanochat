"""
Bidirectional Qwen3 backbone for a LLaDA-style masked diffusion language model.

The only architectural change vs. a standard autoregressive Qwen3 is:
  * the causal mask is removed -> every token attends to every other token
  * a dedicated [MASK] token id is reserved at the end of the vocabulary

Everything else (RMSNorm, RoPE, GQA, SwiGLU MLP) is reused unchanged from
`nanochat.qwen3`, so weights are 1:1 compatible with the AR Qwen3 model.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from nanochat.qwen3 import Qwen3Config, Qwen3ForCausalLM


class DiffusionQwen3(nn.Module):
    """Masked-diffusion wrapper around Qwen3.

    A reserved mask token is appended at index `vocab_size - 1`. The model is a
    bidirectional encoder: it predicts the original tokens at masked positions.
    """

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.mask_token_id = config.vocab_size - 1  # last vocab id is the [MASK] token
        # Reuse the full Qwen3 stack; we only bypass the causal mask in forward().
        self.backbone = Qwen3ForCausalLM(config)

    # -- helpers --------------------------------------------------------------
    @property
    def model(self):
        return self.backbone.model

    @property
    def lm_head(self):
        return self.backbone.lm_head

    def post_init(self):
        self.backbone.post_init()

    def re_init_weights(self):
        self.backbone.re_init_weights()

    # -- forward (bidirectional, no causal mask) ------------------------------
    def logits(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Full bidirectional pass. Returns logits [B, T, V]."""
        m = self.model
        inputs_embeds = m.embed_tokens(input_ids)
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
        hidden_states = inputs_embeds
        position_embeddings = m.rotary_emb(hidden_states, position_ids)
        for layer in m.layers[: m.config.num_hidden_layers]:
            hidden_states = layer(
                hidden_states,
                attention_mask=None,  # <-- key change: bidirectional (no causal mask)
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
            )
        hidden_states = m.norm(hidden_states)
        return self.lm_head(hidden_states)

    def forward(self, input_ids, eps: float = 1e-3):
        """LLaDA training step. Returns the diffusion loss for a clean batch."""
        return diffusion_loss(self, input_ids, eps=eps)

    @torch.no_grad()
    def eval_loglikelihood(self, seq, prompt_index, mc_num: int = 128, batch_size: int = 16):
        """Monte-Carlo log-likelihood of the answer span (LLaDA get_log_likelihood).

        seq: [L] full prompt+answer ids. prompt_index: [L] bool, True over prompt.
        Masks a random number k of *answer* tokens, predicts them, weights CE by
        1/p_mask (=target_len/k), and averages over mc_num samples. Returns scalar
        log-likelihood (negative mean loss). Higher = better.
        """
        device = seq.device
        target_len = int((~prompt_index).sum().item())
        if target_len == 0:
            return 0.0
        seq = seq[None, :]
        bs = min(batch_size, max(1, mc_num))
        loss_acc = []
        for _ in range(max(1, mc_num // bs)):
            batch = seq.repeat(bs, 1)
            noisy, p_mask = _llada_forward_process(batch, prompt_index, self.mask_token_id)
            mask_idx = noisy == self.mask_token_id
            logits = self.logits(noisy).float()
            ce = F.cross_entropy(logits[mask_idx], batch[mask_idx], reduction="none") / p_mask[mask_idx]
            loss_acc.append((ce.sum() / bs).item())
        return -sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def suffix_greedy_match(self, prefix, target, steps: int = 0):
        """Greedy iterative decode of the target span; True if it matches exactly.
        (LLaDA suffix_greedy_prediction.) steps<=0 -> one token per round."""
        device = prefix.device
        x = torch.full((1, len(prefix) + len(target)), self.mask_token_id, dtype=torch.long, device=device)
        x[0, :len(prefix)] = prefix
        rounds = len(target) if steps <= 0 else max(1, min(steps, len(target)))
        per = max(1, len(target) // rounds)
        for _ in range(rounds):
            mask_idx = x == self.mask_token_id
            if not mask_idx.any():
                break
            logits = self.logits(x).float()
            conf, pred = logits.max(dim=-1)
            conf = conf.masked_fill(~mask_idx, -1e9)
            k = min(per, int(mask_idx.sum().item()))
            sel = conf.topk(k, dim=-1).indices
            x.scatter_(1, sel, pred.gather(1, sel))
        return bool((x[0, len(prefix):] == target).all().item())


def forward_process(input_ids: torch.Tensor, mask_token_id: int, eps: float = 1e-3):
    """Mask each token independently with prob p~U(eps,1). (LLaDA GUIDELINES.md)"""
    b, l = input_ids.shape
    t = torch.rand(b, device=input_ids.device)
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)
    masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask
    noisy = torch.where(masked_indices, mask_token_id, input_ids)
    return noisy, masked_indices, p_mask


def _llada_forward_process(batch, prompt_index, mask_id):
    """Mask exactly k answer tokens per row (k~U[1,target_len]) and report p_mask.
    Mirrors LLaDA get_log_likelihood.forward_process for MC likelihood estimation."""
    b, l = batch.shape
    target_len = (l - prompt_index.sum()).item()
    k = torch.randint(1, target_len + 1, (), device=batch.device)
    x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
    x = ((x - 1) % target_len) + 1
    indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
    is_mask = indices < x.unsqueeze(1)
    for i in range(b):
        is_mask[i] = is_mask[i][torch.randperm(target_len, device=batch.device)]
    is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)
    noisy = torch.where(is_mask, mask_id, batch)
    return noisy, (x / target_len).unsqueeze(1).repeat(1, l)


def diffusion_loss(model: DiffusionQwen3, input_ids: torch.Tensor, eps: float = 1e-3):
    # 1% of batches use a random truncated length, like LLaDA pretraining (GUIDELINES.md)
    if torch.rand(1).item() < 0.01:
        rl = int(torch.randint(1, input_ids.shape[1] + 1, (1,)).item())
        input_ids = input_ids[:, :rl]
    noisy, masked_indices, p_mask = forward_process(input_ids, model.mask_token_id, eps)
    logits = model.logits(noisy).float()
    if masked_indices.sum() == 0:
        return logits.sum() * 0.0  # degenerate batch, no masked tokens
    tok = F.cross_entropy(logits[masked_indices], input_ids[masked_indices], reduction="none") / p_mask[masked_indices]
    return tok.sum() / (input_ids.shape[0] * input_ids.shape[1])


def save_diffusion(model: DiffusionQwen3, path: str):
    """Save model weights + config so it can be rebuilt for eval."""
    torch.save({"state_dict": model.state_dict(), "config": model.config.to_dict()}, path)


def load_diffusion(path: str, device="cpu"):
    """Rebuild a DiffusionQwen3 from a checkpoint written by save_diffusion."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = Qwen3Config(**ckpt["config"])
    model = DiffusionQwen3(config).to(torch.float32)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()



