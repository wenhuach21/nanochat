"""
Standalone training of Qwen3 as a (LLaDA-style) masked diffusion LLM.

Run from the project root:

    python -m train_diffusion.train --depth 14

Small CPU/Mac smoke test:

    python -m train_diffusion.train --depth 4 --max-seq-len 512 \
        --device-batch-size 1 --total-batch-size 512 --num-iterations 20

Distributed:

    torchrun --nproc_per_node=8 -m train_diffusion.train --depth 14

This reuses the nanochat tokenizer + dataloader for data, but the model is a
bidirectional Qwen3 trained with the masked-diffusion objective from
LLaDA/GUIDELINES.md. A reserved [MASK] token is appended at the end of the vocab.
"""

import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import gc
import time
import argparse
from contextlib import nullcontext

import torch

from nanochat.common import compute_init, compute_cleanup, print0, autodetect_device_type, get_base_dir
from nanochat.tokenizer import get_tokenizer
from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit
from nanochat.qwen3 import Qwen3Config
from train_diffusion.diffusion_model import DiffusionQwen3, save_diffusion
from train_diffusion.sample_diffusion import generate

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train Qwen3 as a diffusion LLM")
parser.add_argument("--device-type", type=str, default="")
parser.add_argument("--depth", type=int, default=14)
parser.add_argument("--hidden-size", type=int, default=1024)
parser.add_argument("--head-dim", type=int, default=128)
parser.add_argument("--max-seq-len", type=int, default=2048)
parser.add_argument("--device-batch-size", type=int, default=16)
parser.add_argument("--total-batch-size", type=int, default=-1, help="total tokens per step (-1 = device_batch*seq*world)")
parser.add_argument("--num-iterations", type=int, default=5000)
parser.add_argument("--lr", type=float, default=3e-3)
parser.add_argument("--embedding-lr", type=float, default=0.3)
parser.add_argument("--weight-decay", type=float, default=0.1)
parser.add_argument("--warmup-ratio", type=float, default=0.0)
parser.add_argument("--final-lr-frac", type=float, default=0.0)
parser.add_argument("--grad-max-norm", type=float, default=-1.0)
parser.add_argument("--mask-prob-eps", type=float, default=1e-3)
parser.add_argument("--sample-every", type=int, default=1000)
parser.add_argument("--save-every", type=int, default=-1)
parser.add_argument("--model-tag", type=str, default=None)
args = parser.parse_args()

# -----------------------------------------------------------------------------
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

tokenizer = get_tokenizer()
base_vocab = tokenizer.get_vocab_size()
vocab_size = base_vocab + 1  # +1 for [MASK] token at the last index
print0(f"Vocab size: {base_vocab:,} (+1 mask token = {vocab_size:,})")

# -----------------------------------------------------------------------------
def build_config(depth):
    hidden_size = args.hidden_size
    head_dim = args.head_dim
    num_heads = hidden_size // head_dim
    return Qwen3Config(
        head_dim=head_dim, hidden_act="silu", hidden_size=hidden_size, initializer_range=0.02,
        intermediate_size=hidden_size * 3, max_position_embeddings=args.max_seq_len * 10,
        max_window_layers=depth, num_hidden_layers=depth, num_key_value_heads=num_heads,
        rms_norm_eps=1e-6, tie_word_embeddings=False, vocab_size=vocab_size, use_cache=False,
        num_attention_heads=num_heads,
    )

config = build_config(args.depth)
model = DiffusionQwen3(config).to(torch.float32).to(device)
model.post_init()
model.re_init_weights()
num_params = sum(p.numel() for p in model.parameters())
print0(f"Parameters: {num_params/1e9:.3f}B | mask token id: {model.mask_token_id}")

raw_model = model
if ddp:
    from torch.nn.parallel import DistributedDataParallel as DDP
    model = DDP(model, device_ids=[ddp_local_rank], broadcast_buffers=True)

# -----------------------------------------------------------------------------
# Optimizer: AdamW, separate higher LR for embeddings, no decay on norms/bias.
no_decay = ["bias", "norm.weight"]
groups = []
for n, p in raw_model.named_parameters():
    if any(nd in n for nd in no_decay):
        groups.append({"params": [p], "lr": args.lr, "weight_decay": 0.0})
    elif "embed" in n:
        groups.append({"params": [p], "lr": args.embedding_lr, "weight_decay": 0.0})
    elif "lm_head" in n:
        groups.append({"params": [p], "lr": args.lr, "weight_decay": 0.0})
    else:
        groups.append({"params": [p], "lr": args.lr, "weight_decay": args.weight_decay})
optimizer = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95), eps=1e-10)
for g in optimizer.param_groups:
    g["initial_lr"] = g["lr"]

# -----------------------------------------------------------------------------
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device)
x, _, _ = next(train_loader)

tokens_per_step = args.device_batch_size * args.max_seq_len * ddp_world_size
total_batch_size = args.total_batch_size if args.total_batch_size > 0 else tokens_per_step
assert total_batch_size % tokens_per_step == 0
grad_accum_steps = total_batch_size // tokens_per_step
num_iterations = args.num_iterations
print0(f"total_batch_size={total_batch_size:,} grad_accum={grad_accum_steps} iters={num_iterations}")

def lr_mult(it):
    warm = round(args.warmup_ratio * num_iterations)
    if it < warm:
        return (it + 1) / max(1, warm)
    prog = 1.0 - (it - warm) / max(1, num_iterations - warm)
    return args.final_lr_frac + (1.0 - args.final_lr_frac) * prog

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
base_dir = get_base_dir()
out_tag = args.model_tag or f"diff_d{args.depth}"
ckpt_dir = os.path.join(base_dir, "diffusion_checkpoints", out_tag)
if master_process:
    os.makedirs(ckpt_dir, exist_ok=True)
ckpt_path = os.path.join(ckpt_dir, "model.pt")

model.train()
smooth = 0.0
for step in range(num_iterations + 1):
    last = step == num_iterations
    if args.sample_every > 0 and master_process and (last or (step > 0 and step % args.sample_every == 0)):
        model.eval()
        prompt = torch.tensor([tokenizer("The capital of France is", prepend="<|bos|>")], device=device)
        out = generate(raw_model, prompt, gen_len=16, steps=16)
        print0("sample: " + tokenizer.decode(out[0].tolist()[: prompt.shape[1] + 16]))
        model.train()
    if master_process and (last or (args.save_every > 0 and step > 0 and step % args.save_every == 0)):
        save_diffusion(raw_model, ckpt_path)
        print0(f"saved checkpoint -> {ckpt_path}")
    if last:
        break

    t0 = time.time()
    for micro in range(grad_accum_steps):
        sync = model.no_sync() if (ddp and micro < grad_accum_steps - 1) else nullcontext()
        with sync, autocast_ctx:
            loss = raw_model(x, eps=args.mask_prob_eps)
        (loss / grad_accum_steps).backward()
        x, _, _ = next(train_loader)
    lrm = lr_mult(step)
    for g in optimizer.param_groups:
        g["lr"] = g["initial_lr"] * lrm
    if args.grad_max_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_max_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    lf = loss.item()
    smooth = 0.9 * smooth + 0.1 * lf
    print0(f"step {step:05d}/{num_iterations} | loss {smooth/(1-0.9**(step+1)):.4f} | lrm {lrm:.2f} | dt {(time.time()-t0)*1000:.0f}ms")
    if step % 2000 == 0:
        gc.collect()

compute_cleanup()

