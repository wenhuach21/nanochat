"""
Train the Qwen3.5 (text-only) LLM, nanochat-style. Run from the repo root:

python -m nanochat.qwen35.base_train_qwen3p5

or distributed:

torchrun --nproc_per_node=8 -m nanochat.qwen35.base_train_qwen3p5

Tiny CPU/laptop smoke run:
python -m nanochat.qwen35.base_train_qwen3p5 --depth=4 --hidden-size=128 --head-dim=32 \
    --max-seq-len=512 --device-batch-size=1 --core-metric-every=-1 \
    --total-batch-size=512 --num-iterations=20

This mirrors scripts/base_train_qwen3.py but builds a Qwen3_5ForCausalLM (hybrid
GatedDeltaNet + gated full-attention) and defaults to the transformers tokenizer.
"""

import os


os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import time
import math
import argparse
from contextlib import nullcontext, contextmanager

import wandb
import torch
import torch.nn.functional as F

from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit, tokenizing_distributed_data_loader_with_state_bos_bestfit
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type, get_peak_flops
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint, save_hf_checkpoint, load_checkpoint_any
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from nanochat.flash_attention import HAS_FA3
from scripts.base_eval import evaluate_core
print_banner()

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Pretrain Qwen3.5 text-only base model")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# Data
parser.add_argument("--data-dir", type=str, default="", help="data directory path. Use colon-separated (semicolon on Windows) paths to mix datasets. Empty = default fineweb-edu-100b-shuffle.")
parser.add_argument("--shuffle-files", action="store_true", default=True, help="shuffle parquet file order each epoch (recommended when mixing multiple data sources)")
parser.add_argument("--no-shuffle-files", action="store_false", dest="shuffle_files", help="disable file shuffling (read files in sorted order)")
parser.add_argument("--tokenizer-backend", type=str, default="transformers", choices=["transformers", "rustbpe"], help="tokenizer backend for Qwen3.5 (default: transformers-compatible PreTrainedTokenizerFast; rustbpe still selectable)")
# Model architecture
parser.add_argument("--depth", type=int, default=28, help="number of Transformer layers")
parser.add_argument("--hidden-size", type=int, default=1024, help="hidden size")
parser.add_argument("--head-dim", type=int, default=128, help="attention head dimension")
parser.add_argument("--num-attention-heads", type=int, default=-1, help="number of attention (query) heads (-1 = hidden_size // head_dim). Qwen3.5 decouples head_dim from hidden_size, so this may be set explicitly.")
parser.add_argument("--intermediate-size", type=int, default=-1, help="MLP intermediate size (-1 = hidden_size * 3)")
parser.add_argument("--num-kv-heads", type=int, default=-1, help="number of key/value heads for GQA (-1 = same as num_attention_heads)")
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
parser.add_argument("--rope-theta", type=float, default=1000000.0, help="RoPE theta base")
# Qwen3.5 hybrid layout / linear-attention (GatedDeltaNet) knobs
parser.add_argument("--full-attention-interval", type=int, default=4, help="every Nth layer is full (softmax) attention; the rest are GatedDeltaNet linear attention")
parser.add_argument("--linear-conv-kernel-dim", type=int, default=4, help="GatedDeltaNet short conv kernel size")
parser.add_argument("--linear-num-value-mult", type=int, default=2, help="linear_num_value_heads = num_kv_heads * this (must be an integer multiple). Ignored if --linear-num-value-heads is set.")
# GatedDeltaNet linear-attention head geometry (decoupled from softmax attention).
# Each defaults to -1, meaning "derive from head_dim / num_kv_heads" (previous behavior).
parser.add_argument("--linear-key-head-dim", type=int, default=-1, help="GatedDeltaNet key head dim (-1 = use --head-dim)")
parser.add_argument("--linear-value-head-dim", type=int, default=-1, help="GatedDeltaNet value head dim (-1 = use --head-dim)")
parser.add_argument("--linear-num-key-heads", type=int, default=-1, help="GatedDeltaNet number of key heads (-1 = use num_kv_heads)")
parser.add_argument("--linear-num-value-heads", type=int, default=-1, help="GatedDeltaNet number of value heads (-1 = num_kv_heads * --linear-num-value-mult)")
parser.add_argument("--partial-rotary-factor", type=float, default=1.0, help="fraction of head_dim that gets rotary embedding (Qwen3.5 2B uses 0.25)")
# Training horizon (only one used, in order of precedence)
parser.add_argument("--num-iterations", type=int, default=-1, help="explicit number of optimization steps (-1 = disable)")
parser.add_argument("--target-flops", type=float, default=-1.0, help="calculate num_iterations to reach target_flops (-1 = disable)")
parser.add_argument("--target-param-data-ratio", type=float, default=10.5, help="calculate num_iterations to maintain data:param ratio (Chinchilla=20, -1 = disable)")
# Optimization
parser.add_argument("--device-batch-size", type=int, default=8, help="per-device batch size. reduce to 16,8,4,... if you OOM on VRAM.")
parser.add_argument("--total-batch-size", type=int, default=-1, help="total batch size in tokens, e.g. 524288. (-1 = auto-compute optimal)")
parser.add_argument("--lr", type=float, default=3e-3, help="learning rate for AdamW params")
parser.add_argument("--embedding-lr", type=float, default=0.3, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="learning rate for unembedding (lm_head) parameters (Adam)")
parser.add_argument("--weight-decay", type=float, default=0.2, help="weight decay for the Muon optimizer (for weights)")
parser.add_argument("--muon-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--muon-per-head", action="store_true", help="orthogonalize softmax-attention q/k/v/o_proj weights per attention head (Kimi-K3 style) instead of as one fused matrix")
parser.add_argument("--optimizer-mode", type=str, default="hybrid", choices=["hybrid", "adamw"], help="optimizer setup: hybrid(AdamW+Muon) or adamw")
parser.add_argument("--ema-decay", type=float, default=0.0, help="EMA decay for model parameters (0 disables EMA, e.g. 0.999)")
parser.add_argument("--ema-eval", action="store_true", help="evaluate/sample/save with EMA parameters when EMA is enabled")
parser.add_argument("--mtp-num-heads", type=int, default=0, help="number of auxiliary MTP heads (0 disables MTP)")
parser.add_argument("--mtp-weight", type=float, default=0.0, help="weight for MTP auxiliary loss")
# Logit softcap: softcap*tanh(x/softcap) applied during training then annealed away.
parser.add_argument("--logit-softcap", type=float, default=15.0, help="logit softcap start value (0 disables softcap)")
parser.add_argument("--logit-softcap-end", type=float, default=100.0, help="logit softcap end value (grows toward this while annealing)")
parser.add_argument("--logit-softcap-anneal-steps", type=int, default=-1, help="steps over which the softcap anneals away. Set -1 to freeze the softcap at --logit-softcap (e.g. constant 15) forever.")
# DFLASH joint pretraining (block-diffusion draft trained online with the base model)
parser.add_argument("--dflash-enable", action="store_true", help="train a DFLASH draft jointly during pretraining")
parser.add_argument("--dflash-layers", type=int, default=1, help="number of DFLASH draft decoder layers")
parser.add_argument("--dflash-block-size", type=int, default=16, help="DFLASH block size")
parser.add_argument("--dflash-weight", type=float, default=0.3, help="weight for DFLASH draft loss")
parser.add_argument("--dflash-mask-token-id", type=int, default=-1, help="DFLASH mask token id (-1 = last vocab id)")
parser.add_argument("--dflash-grad-to-target", action="store_true", help="allow DFLASH loss to backprop into the base model (off = detached)")
parser.add_argument("--dflash-num-blocks", type=int, default=8, help="number of randomly-anchored blocks per sequence")
parser.add_argument("--dflash-gamma", type=float, default=4.0, help="exp-decay loss weighting gamma (w_k=exp(-(k)/gamma))")
parser.add_argument("--adam-beta1", type=float, default=0.9, help="Adam beta1 for embedding/unembedding")
parser.add_argument("--adam-beta2", type=float, default=0.95, help="Adam beta2 for embedding/unembedding")
parser.add_argument("--warmup-ratio", type=float, default=0.0, help="ratio of iterations for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.5, help="ratio of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.0, help="final LR as fraction of initial LR")
parser.add_argument("--grad-max-norm", type=float, default=-1.0, help="clip-grad-norm")
parser.add_argument("--lr-schedule", type=str, default="default", choices=["linear", "cosine", "default"], help="LR decay schedule during warmdown")
parser.add_argument("--resume-from-step", type=int, default=-1, help="resume training from this step (-1 = disable)")
parser.add_argument("--init-from", type=str, default="", help="checkpoint directory to initialize model weights from")
parser.add_argument("--init-step", type=int, default=-1, help="step of the checkpoint to load for --init-from (-1 = latest)")
parser.add_argument("--end-step", type=int, default=-1, help="stop THIS training session at this step (-1 = disable)")
# Evaluation
parser.add_argument("--eval-every", type=int, default=250, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=40*524288, help="number of tokens to evaluate val loss on")
parser.add_argument("--core-metric-every", type=int, default=5000, help="evaluate CORE metric every N steps (-1 = disable)")
parser.add_argument("--core-metric-max-per-task", type=int, default=-1, help="examples per task for CORE metric")
parser.add_argument("--sample-every", type=int, default=2000, help="sample from model every N steps (-1 = disable)")
parser.add_argument("--save-every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")
parser.add_argument("--save-format", type=str, default="both", choices=["pt", "hf", "both"], help="checkpoint export format. 'pt' = nanochat resume-able .pt shards only; 'hf' = transformers-loadable safetensors folder only (save_pretrained + tokenizer, loadable via AutoModelForCausalLM.from_pretrained(dir, trust_remote_code=True)); 'both' (default) writes both.")
# Output
parser.add_argument("--model-tag", type=str, default=None, help="override model tag for checkpoint directory name")
# Depth expansion (model growth)
parser.add_argument("--expand-from", type=str, default=None, help="path to a smaller checkpoint to expand from")
parser.add_argument("--expand-from-step", type=int, default=-1, help="step of the checkpoint to expand from (-1 = latest)")
args = parser.parse_args()
user_config = vars(args).copy()  # for logging
# -----------------------------------------------------------------------------
# Compute init and wandb logging

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')

use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=args.run, config=user_config)

if HAS_FA3:
    print0("✓ Using Flash Attention 3 (Hopper GPU detected), efficient, new and awesome.")
else:
    print0("!" * 80)
    print0("WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback")
    print0("WARNING: Training will be less efficient without FA3")
    print0("!" * 80)

# -----------------------------------------------------------------------------
# Tokenizer (transformers-compatible by default). Also gives us the vocab size.
tokenizer = get_tokenizer(backend=args.tokenizer_backend)
token_bytes = get_token_bytes(device=device, backend=args.tokenizer_backend)
print0(f"Tokenizer backend: {args.tokenizer_backend}")
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")
# -----------------------------------------------------------------------------
def build_model_meta(
    depth,
    hidden_size_override=None,
    num_kv_heads_override=None,
    head_dim_override=None,
    num_attention_heads_override=None,
    intermediate_size_override=None,
    linear_key_head_dim_override=None,
    linear_value_head_dim_override=None,
    linear_num_key_heads_override=None,
    linear_num_value_heads_override=None,
):
    from nanochat.qwen35 import Qwen3_5TextConfig, Qwen3_5ForCausalLM
    layers = depth
    hidden_size = hidden_size_override if hidden_size_override is not None else args.hidden_size
    head_dim = head_dim_override if head_dim_override is not None else args.head_dim
    if num_attention_heads_override is not None:
        num_attention_heads = num_attention_heads_override
    else:
        num_attention_heads = args.num_attention_heads if args.num_attention_heads > 0 else hidden_size // head_dim
    num_kv_heads = num_kv_heads_override if num_kv_heads_override is not None else (args.num_kv_heads if args.num_kv_heads > 0 else num_attention_heads)
    if intermediate_size_override is not None:
        intermediate_size = intermediate_size_override
    else:
        intermediate_size = args.intermediate_size if args.intermediate_size > 0 else hidden_size * 3
    # GatedDeltaNet (linear attention) head configuration. These are decoupled from the
    # softmax-attention head_dim / num_kv_heads; each override defaults to the previous
    # derive-from-attention behavior when left at -1.
    if linear_key_head_dim_override is not None:
        linear_key_head_dim = linear_key_head_dim_override
    else:
        linear_key_head_dim = args.linear_key_head_dim if args.linear_key_head_dim > 0 else head_dim
    if linear_value_head_dim_override is not None:
        linear_value_head_dim = linear_value_head_dim_override
    else:
        linear_value_head_dim = args.linear_value_head_dim if args.linear_value_head_dim > 0 else head_dim
    if linear_num_key_heads_override is not None:
        linear_num_key_heads = linear_num_key_heads_override
    else:
        linear_num_key_heads = args.linear_num_key_heads if args.linear_num_key_heads > 0 else num_kv_heads
    if linear_num_value_heads_override is not None:
        linear_num_value_heads = linear_num_value_heads_override
    else:
        linear_num_value_heads = (
            args.linear_num_value_heads if args.linear_num_value_heads > 0
            else max(1, args.linear_num_value_mult) * num_kv_heads
        )
    config = Qwen3_5TextConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_act="silu",
        max_position_embeddings=args.max_seq_len * 10,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        use_cache=False,
        rope_theta=args.rope_theta,
        partial_rotary_factor=args.partial_rotary_factor,
        full_attention_interval=args.full_attention_interval,
        linear_conv_kernel_dim=args.linear_conv_kernel_dim,
        linear_key_head_dim=linear_key_head_dim,
        linear_value_head_dim=linear_value_head_dim,
        linear_num_key_heads=linear_num_key_heads,
        linear_num_value_heads=linear_num_value_heads,
    )
    config.mtp_num_heads = max(0, int(args.mtp_num_heads))
    config.mtp_weight = max(0.0, float(args.mtp_weight))
    config.logit_softcap = float(args.logit_softcap)
    config.logit_softcap_end = float(args.logit_softcap_end)
    config.logit_softcap_anneal_steps = int(args.logit_softcap_anneal_steps)
    with torch.device("meta"):
        model_meta = Qwen3_5ForCausalLM(config)
    return model_meta

# Build the model, move to device, init the weights
model = build_model_meta(args.depth)  # 1) Build on meta device (only shapes/dtypes, no data)
model.to(torch.float32)
print0(model.__class__.__name__)
model_config = model.config
print0(model_config)
print0(f"Layer types: {model_config.layer_types}")
model.to_empty(device=device)  # 2) storage on device but uninitialized
model.post_init()              # 3) initialize tensors
model.re_init_weights()

# If softcap annealing is disabled (--logit-softcap-anneal-steps -1), freeze the softcap
# at --logit-softcap (e.g. constant 15) by setting the step counter to -1.
softcap_frozen = int(args.logit_softcap_anneal_steps) < 0
if softcap_frozen and hasattr(model, "set_softcap_step"):
    model.set_softcap_step(-1)

# ---- Depth Expansion (Model Growth) ----
if args.expand_from is not None:
    from scripts.grow_depth import expand_layers_depth

    expand_dir = args.expand_from
    if args.expand_from_step == -1:
        import glob as _glob
        model_files = _glob.glob(os.path.join(expand_dir, "model_step*.pt"))
        if not model_files:
            raise FileNotFoundError(f"No model checkpoints found in {expand_dir}")
        steps = [int(f.split("model_step")[1].split(".pt")[0]) for f in model_files]
        expand_step = max(steps)
    else:
        expand_step = args.expand_from_step

    expand_model_path = os.path.join(expand_dir, f"model_step{expand_step:05d}.pt")
    if not os.path.exists(expand_model_path):
        expand_model_path = os.path.join(expand_dir, f"model_step{expand_step}.pt")
    print0(f"[Depth Expansion] Loading source checkpoint: {expand_model_path}")
    expand_state = torch.load(expand_model_path, map_location=device, weights_only=False)

    src_layer_keys = [k for k in expand_state.keys() if "model.layers." in k]
    src_layer_indices = set()
    for k in src_layer_keys:
        parts = k.split("model.layers.")
        if len(parts) > 1:
            idx = int(parts[1].split(".")[0])
            src_layer_indices.add(idx)
    src_n_layers = max(src_layer_indices) + 1 if src_layer_indices else 0
    tgt_n_layers = args.depth
    if src_n_layers == 0:
        raise ValueError("[Depth Expansion] Could not detect layer count from source checkpoint")

    if src_n_layers == tgt_n_layers:
        print0(f"[Depth Expansion] Source has same depth ({src_n_layers}), loading weights directly.")
        model.load_state_dict(expand_state, strict=False, assign=True)
    elif src_n_layers < tgt_n_layers:
        print0(f"[Depth Expansion] Expanding depth: {src_n_layers} -> {tgt_n_layers} layers")
        src_model = build_model_meta(src_n_layers)
        src_model.to_empty(device=device)
        src_model.post_init()
        src_model.load_state_dict(expand_state, strict=False, assign=True)
        zero_proj_names = ["o_proj.weight", "down_proj.weight", "out_proj.weight"]
        expanded_layers = expand_layers_depth(src_model.model.layers, tgt_n_layers, zero_proj_names)
        model.model.embed_tokens.load_state_dict(src_model.model.embed_tokens.state_dict())
        model.model.norm.load_state_dict(src_model.model.norm.state_dict())
        model.lm_head.load_state_dict(src_model.lm_head.state_dict())
        if hasattr(src_model, 'mtp_heads') and len(src_model.mtp_heads) > 0:
            for i, head in enumerate(src_model.mtp_heads):
                if i < len(model.mtp_heads):
                    model.mtp_heads[i].load_state_dict(head.state_dict())
        model.model.layers = expanded_layers
        for idx, layer in enumerate(model.model.layers):
            if hasattr(layer, 'self_attn'):
                layer.self_attn.layer_idx = idx
            if hasattr(layer, 'linear_attn'):
                layer.linear_attn.layer_idx = idx
        del src_model, expand_state
        if device_type == "cuda":
            torch.cuda.empty_cache()
        print0("[Depth Expansion] Verifying expanded model...")
        model.eval()
        test_ids = torch.randint(0, vocab_size, (1, 32), device=device)
        with torch.no_grad(), autocast_ctx:
            test_logits = model(test_ids)
        assert not torch.isnan(test_logits).any(), "NaN detected in expanded model output!"
        print0("[Depth Expansion] ✅ Expansion successful! Model output is valid.")
        model.train()
    else:
        raise ValueError(f"[Depth Expansion] Source has MORE layers ({src_n_layers}) than target ({tgt_n_layers}).")

# If we are resuming, overwrite the model parameters with those of the checkpoint
base_dir = get_base_dir()
output_dirname = args.model_tag if args.model_tag else f"qwen3p5_d{args.depth}"
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
resuming = args.resume_from_step != -1
optimizer_data = None
meta_data = None
if resuming:
    print0(f"Resuming optimization from step {args.resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint_any(checkpoint_dir, args.resume_from_step, device, load_optimizer=True, rank=ddp_rank)
    # Drop the legacy `_softcap_step` buffer if present: it is now a non-persistent buffer
    # and no longer part of the model state_dict (strict=True would otherwise reject it).
    model_data.pop("_softcap_step", None)
    model.load_state_dict(model_data, strict=True, assign=True)
    # `_softcap_step` is a non-persistent buffer (not in the checkpoint), so re-sync it
    # from the resumed step to keep the logit-softcap annealing schedule correct.
    # When the softcap is frozen (--logit-softcap-anneal-steps -1) keep it pinned at -1.
    if hasattr(model, "set_softcap_step"):
        model.set_softcap_step(-1 if softcap_frozen else args.resume_from_step)
    del model_data
elif args.init_from:
    init_dir = os.path.expanduser(args.init_from)
    if args.init_step == -1:
        import re as _re, glob as _glob
        model_files = _glob.glob(os.path.join(init_dir, "model_*.pt"))
        if not model_files:
            raise FileNotFoundError(f"No model checkpoints found in {init_dir}")
        steps = [int(_re.search(r"model_(\d+)\.pt", f).group(1)) for f in model_files]
        init_step = max(steps)
    else:
        init_step = args.init_step
    print0(f"Initializing model weights from {init_dir} step {init_step} (no optimizer/schedule restore)")
    init_model_data, _, _ = load_checkpoint(init_dir, init_step, device, load_optimizer=False)
    missing, unexpected = model.load_state_dict(init_model_data, strict=False, assign=True)
    if missing:
        print0(f"  Warning: {len(missing)} missing keys (randomly initialized): {missing[:5]}...")
    if unexpected:
        print0(f"  Warning: {len(unexpected)} unexpected keys (ignored): {unexpected[:5]}...")
    del init_model_data

# -----------------------------------------------------------------------------
# EMA helpers

def init_ema_state(model):
    return {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}


@torch.no_grad()
def update_ema_state(model, ema_state, decay):
    one_minus_decay = 1.0 - decay
    for name, p in model.named_parameters():
        if name in ema_state:
            ema_state[name].mul_(decay).add_(p.detach(), alpha=one_minus_decay)


@contextmanager
def apply_ema_weights(model, ema_state):
    if not ema_state:
        yield
        return
    backup = {}
    try:
        for name, p in model.named_parameters():
            if name in ema_state:
                backup[name] = p.detach().clone()
                p.data.copy_(ema_state[name])
        yield
    finally:
        for name, p in model.named_parameters():
            if name in backup:
                p.data.copy_(backup[name])

# -----------------------------------------------------------------------------
orig_model = model  # uncompiled model, for saving/eval/inference
print0(orig_model.dtype)

if ddp:
    from torch.nn.parallel import DistributedDataParallel as DDP
    model = DDP(model, device_ids=[ddp_local_rank], broadcast_buffers=True)
    print0(f"Wrapped model in DistributedDataParallel (world_size={ddp_world_size})")

# -----------------------------------------------------------------------------
# DFLASH draft model (optional, jointly trained during pretraining)
draft_model = None
draft_layer_ids = None
dflash_mask_token_id = None
if args.dflash_enable:
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "dflash"))
    from dflash.model import DFlashDraftModel, build_target_layer_ids, extract_context_feature
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Config as HFQwen3Config
    num_heads = args.hidden_size // args.head_dim
    num_kv_heads_draft = args.num_kv_heads if args.num_kv_heads > 0 else num_heads
    draft_cfg = HFQwen3Config(
        head_dim=args.head_dim, hidden_act="silu", hidden_size=args.hidden_size, initializer_range=0.02,
        intermediate_size=args.hidden_size * 3, max_position_embeddings=args.max_seq_len * 10,
        max_window_layers=args.dflash_layers, num_hidden_layers=args.dflash_layers,
        num_key_value_heads=num_kv_heads_draft, rms_norm_eps=1e-6, tie_word_embeddings=False,
        vocab_size=vocab_size, use_cache=False, num_attention_heads=num_heads,
    )
    draft_cfg.num_target_layers = args.depth
    draft_cfg.block_size = args.dflash_block_size
    dflash_mask_token_id = (vocab_size - 1) if args.dflash_mask_token_id < 0 else args.dflash_mask_token_id
    draft_layer_ids = build_target_layer_ids(args.depth, args.dflash_layers)
    draft_cfg.dflash_config = {"target_layer_ids": draft_layer_ids, "mask_token_id": dflash_mask_token_id}
    draft_model = DFlashDraftModel(draft_cfg).to(device).to(torch.float32)
    draft_model.train()
    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        draft_model = DDP(draft_model, device_ids=[ddp_local_rank], broadcast_buffers=True)
    print0(f"DFLASH draft enabled: layers={args.dflash_layers} block={args.dflash_block_size} layer_ids={draft_layer_ids} mask_id={dflash_mask_token_id}")

# -----------------------------------------------------------------------------
# Scaling laws / batch size / LR corrections
num_params = sum(p.numel() for p in model.parameters())
print0(f"Parameters: {num_params/1e9:.3f}B")

target_tokens = int(args.target_param_data_ratio * num_params)
B_REF = 256 * 2048
# Scaling-law reference model = the ~0.8B Qwen3.5 config we tune hyperparameters on.
# HPs are searched on 0.8B and transferred to larger models (e.g. 2B), so D_REF must be
# the param ratio relative to this FIXED 0.8B reference, independent of the current run's
# args (otherwise, e.g. running the 2B sweep would build a wrong reference from 2B args
# and mis-scale batch size / weight decay). Keep in sync with run_sweep_3p5_0p8b.py.
d_ref_model = build_model_meta(
    24,
    hidden_size_override=1024,
    head_dim_override=256,
    num_attention_heads_override=8,
    num_kv_heads_override=2,
    intermediate_size_override=3584,
    linear_key_head_dim_override=128,
    linear_value_head_dim_override=128,
    linear_num_key_heads_override=16,
    linear_num_value_heads_override=16,
)
d_ref_params = sum(p.numel() for p in d_ref_model.parameters())
D_REF = num_params / d_ref_params
del d_ref_model
total_batch_size = args.total_batch_size
if total_batch_size == -1:
    predicted_batch_size = B_REF * D_REF ** 0.383
    total_batch_size = 2 ** round(math.log2(predicted_batch_size))
    print0(f"Auto-computed optimal batch size: {total_batch_size:,} tokens")

if resuming and meta_data is not None and meta_data.get("total_batch_size") is not None:
    ckpt_total_batch_size = int(meta_data["total_batch_size"])
    if total_batch_size != ckpt_total_batch_size:
        print0(f"[Resume] Overriding total_batch_size {total_batch_size:,} -> {ckpt_total_batch_size:,} (from checkpoint).")
    total_batch_size = ckpt_total_batch_size

batch_ratio = total_batch_size / B_REF
batch_lr_scale = batch_ratio ** 0.5 if batch_ratio != 1.0 else 1.0
if batch_lr_scale != 1.0:
    print0(f"Scaling LRs by {batch_lr_scale:.4f} for batch size {total_batch_size:,} (reference: {B_REF:,})")

weight_decay_scaled = args.weight_decay * math.sqrt(total_batch_size / B_REF) * (1.0 / max(D_REF, 1.0))
if weight_decay_scaled != args.weight_decay:
    print0(f"Scaling weight decay from {args.weight_decay:.6f} to {weight_decay_scaled:.6f} for depth {args.depth}")

lr = args.lr * batch_lr_scale
weight_decay = args.weight_decay
no_decay = ["bias", "norm.weight"]

# -----------------------------------------------------------------------------
# Optimizer parameter grouping.
# NOTE vs qwen3: the GatedDeltaNet introduces non-2D params (A_log, dt_bias 1D;
# conv1d.weight 3D). Muon only handles 2D matrices, so any non-2D param is routed
# to the AdamW (no weight decay) group instead of the Muon default group.
base_fan_in = 1024
embed_params = []
other_params = {}
no_weight_decay = {"params": [], "weight_decay": 0.0}
muon_lr = args.muon_lr
matrix_lr = muon_lr if muon_lr > 0 else lr
default_group = {"params": [], "lr": matrix_lr}
lm_head_params = []
adamw_all_params = []

# Per-head Muon (Kimi-K3 style): softmax-attention projections concatenate heads
# along one dim. Splitting them into per-head blocks lets Muon orthogonalize each
# head independently instead of mixing heads in a single Newton-Schulz iteration.
# q/k/v_proj concatenate heads along dim 0 (output rows); o_proj along dim 1 (input
# cols). Under GQA q and k/v have different head counts, so they need separate groups.
muon_per_head = args.muon_per_head and args.optimizer_mode == "hybrid"
_ph_num_attention_heads = args.num_attention_heads if args.num_attention_heads > 0 else args.hidden_size // args.head_dim
_ph_num_kv_heads = args.num_kv_heads if args.num_kv_heads > 0 else _ph_num_attention_heads
per_head_q = {"params": [], "lr": matrix_lr, "num_split": _ph_num_attention_heads, "split_dim": 0}
per_head_kv = {"params": [], "lr": matrix_lr, "num_split": _ph_num_kv_heads, "split_dim": 0}
per_head_o = {"params": [], "lr": matrix_lr, "num_split": _ph_num_attention_heads, "split_dim": 1}

def _match_per_head_group(name, m):
    """Return the per-head Muon group for a softmax-attention proj weight, else None."""
    if not muon_per_head or ".self_attn." not in name or m.dim() != 2:
        return None
    if name.endswith(".q_proj.weight") and m.shape[0] % _ph_num_attention_heads == 0:
        return per_head_q
    if (name.endswith(".k_proj.weight") or name.endswith(".v_proj.weight")) and m.shape[0] % _ph_num_kv_heads == 0:
        return per_head_kv
    if name.endswith(".o_proj.weight") and m.shape[1] % _ph_num_attention_heads == 0:
        return per_head_o
    return None

for n, m in model.named_parameters():
    is_no_decay = any(nd in n for nd in no_decay)
    is_embed = "embed" in n
    is_lm_head = "lm_head" in n

    # AdamW-all mode groups (all params optimized by AdamW)
    if is_no_decay or (m.dim() != 2):
        adamw_all_params.append({"params": [m], "lr": args.lr, "weight_decay": 0.0})
    elif is_embed:
        adamw_all_params.append({"params": [m], "lr": args.embedding_lr, "weight_decay": 0.0})
    elif is_lm_head:
        adamw_all_params.append({"params": [m], "lr": args.lr, "weight_decay": 0.0})
    else:
        adamw_all_params.append({"params": [m], "lr": args.lr, "weight_decay": weight_decay_scaled})

    # Hybrid mode groups
    if is_no_decay or (m.dim() != 2):
        # 1D/3D params (norms, biases, A_log, dt_bias, conv1d.weight) -> AdamW, no wd
        no_weight_decay["params"].append(m)
    elif is_lm_head:
        lm_head_params.append({"params": [m], "lr": args.lr, "weight_decay": 0.0})
    elif is_embed:
        embed_params.append({"params": [m], "lr": args.embedding_lr, "weight_decay": 0.0})
    elif m.shape[-1] != base_fan_in:
        fan_in = m.shape[-1]
        if fan_in not in other_params:
            other_params[fan_in] = {"lr": matrix_lr, "params": [m]}
        else:
            other_params[fan_in]["params"].append(m)
    else:
        default_group["params"].append(m)

optimizer_grouped_parameters = [no_weight_decay]
optimizer_grouped_parameters.extend(embed_params)
optimizer_grouped_parameters.extend(lm_head_params)

if args.optimizer_mode == "adamw":
    optimizer = torch.optim.AdamW(
        adamw_all_params, lr=lr, weight_decay=weight_decay_scaled,
        betas=(args.adam_beta1, args.adam_beta2), eps=1e-10,
    )
else:
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, lr=lr, weight_decay=weight_decay_scaled,
        betas=(args.adam_beta1, args.adam_beta2), eps=1e-10,
    )
print0(f"Optimizer mode: {args.optimizer_mode}")

muon_optimizer = None
if args.optimizer_mode == "hybrid" and args.muon_lr >= 0:
    moun_parameters = []
    moun_parameters.extend(other_params.values())
    if default_group["params"]:
        moun_parameters.append(default_group)
    # Per-head attention groups (only populated when --muon-per-head is set)
    for ph_group in (per_head_q, per_head_kv, per_head_o):
        if ph_group["params"]:
            moun_parameters.append(ph_group)
    if muon_per_head:
        n_ph = sum(len(g["params"]) for g in (per_head_q, per_head_kv, per_head_o))
        print0(f"Muon per-head enabled: {n_ph} attention proj matrices split "
               f"(q heads={_ph_num_attention_heads}, kv heads={_ph_num_kv_heads})")
    if moun_parameters:
        muon_optimizer = torch.optim.Muon(moun_parameters, weight_decay=weight_decay_scaled, adjust_lr_fn="match_rms_adamw", lr=muon_lr)
        for group in muon_optimizer.param_groups:
            group["initial_lr"] = group["lr"]

for group in optimizer.param_groups:
    group["initial_lr"] = group["lr"]

ema_state = None
if args.ema_decay > 0:
    ema_state = init_ema_state(orig_model)
    print0(f"EMA enabled (decay={args.ema_decay})")

draft_optimizer = None
if draft_model is not None:
    draft_optimizer = torch.optim.AdamW(draft_model.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
    for group in draft_optimizer.param_groups:
        group["initial_lr"] = group["lr"]

if resuming:
    if isinstance(optimizer_data, dict) and "adamw" in optimizer_data:
        optimizer.load_state_dict(optimizer_data["adamw"])
        if muon_optimizer is not None and optimizer_data.get("muon") is not None:
            muon_optimizer.load_state_dict(optimizer_data["muon"])
        if ema_state is not None and optimizer_data.get("ema") is not None:
            for name, ema_tensor in optimizer_data["ema"].items():
                if name in ema_state:
                    ema_state[name].copy_(ema_tensor.to(device=ema_state[name].device, dtype=ema_state[name].dtype))
    else:
        optimizer.load_state_dict(optimizer_data)
    del optimizer_data

# -----------------------------------------------------------------------------
# DataLoaders
dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
data_dir = args.data_dir if args.data_dir else None

from nanochat.dataset import print_dataset_summary
print0("=" * 60)
if master_process:
    print_dataset_summary(data_dir, split="train")
print0("=" * 60)

train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device, resume_state_dict=dataloader_resume_state_dict, data_dir=data_dir, shuffle_files=args.shuffle_files)
build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device, data_dir=data_dir)
x, y, dataloader_state_dict = next(train_loader)

# -----------------------------------------------------------------------------
# Number of iterations + schedulers
if args.num_iterations > 0:
    num_iterations = args.num_iterations
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif args.target_param_data_ratio > 0:
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")

total_tokens = total_batch_size * num_iterations
print0(f"Total number of training tokens: {total_tokens:,}")

end_step = args.end_step
if end_step is not None and end_step > 0:
    if end_step >= num_iterations:
        print0(f"[end-step] Requested end_step={end_step} >= num_iterations={num_iterations}; ignoring.")
        end_step = -1
    elif resuming and meta_data is not None and end_step <= int(meta_data["step"]):
        raise ValueError(f"[end-step] end_step={end_step} must be greater than the resume step ({int(meta_data['step'])}).")
    else:
        print0(f"[end-step] This session will stop at step {end_step} (full horizon is {num_iterations}).")


def get_lr_multiplier(it):
    warmup_iters = round(args.warmup_ratio * num_iterations)
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    else:
        progress = 1.0 - (it - warmup_iters) / (num_iterations - warmup_iters)
        if args.lr_schedule == "cosine":
            decay = (1 + math.cos(math.pi * (1.0 - progress))) / 2
            return args.final_lr_frac + (1.0 - args.final_lr_frac) * decay
        elif args.lr_schedule == "linear":
            decay = progress
            return args.final_lr_frac + (1.0 - args.final_lr_frac) * decay
        else:
            if it <= num_iterations - warmdown_iters:
                return 1.0
            progress = (num_iterations - it) / warmdown_iters
            return progress * 1.0 + (1 - progress) * args.final_lr_frac


def get_muon_momentum(it):
    frac = min(it / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95


def get_weight_decay(it):
    return weight_decay_scaled * (1 - it / num_iterations)

# -----------------------------------------------------------------------------
# Training loop
if not resuming:
    step = 0
    val_bpb = None
    min_val_bpb = float("inf")
    smooth_train_loss = 0
    total_training_time = 0
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_bpb = meta_data["val_bpb"]
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]

tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

while True:
    last_step = step == num_iterations
    session_end = end_step > 0 and step == end_step
    if session_end:
        last_step = True
    flops_so_far = 0

    results = {}
    if args.core_metric_every > 0 and (last_step or (step > 0 and step != args.resume_from_step and step % args.core_metric_every == 0)):
        model.eval()
        ema_ctx = apply_ema_weights(orig_model, ema_state) if (args.ema_eval and ema_state is not None) else nullcontext()
        with ema_ctx:
            with autocast_ctx:
                results = evaluate_core(orig_model, tokenizer, device, max_per_task=args.core_metric_max_per_task)
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "core_metric": results["core_metric"],
            "centered_results": results["centered_results"],
        })
        model.train()

    if args.sample_every > 0 and master_process and (last_step or (step > 0 and step != args.resume_from_step and step % args.sample_every == 0)):
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        engine = Engine(orig_model, tokenizer)
        ema_ctx = apply_ema_weights(orig_model, ema_state) if (args.ema_eval and ema_state is not None) else nullcontext()
        with ema_ctx:
            for prompt in prompts:
                tokens = tokenizer(prompt, prepend="<|bos|>")
                with autocast_ctx:
                    sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
                print0(tokenizer.decode(sample[0]))
        model.train()

    if last_step or (step > 0 and step != args.resume_from_step and args.save_every > 0 and step % args.save_every == 0):
        meta_dict = {
            "step": step,
            "val_bpb": val_bpb,
            "user_config": user_config,
            "device_batch_size": args.device_batch_size,
            "max_seq_len": args.max_seq_len,
            "total_batch_size": total_batch_size,
            "dataloader_state_dict": dataloader_state_dict,
            "loop_state": {
                "min_val_bpb": min_val_bpb,
                "smooth_train_loss": smooth_train_loss,
                "total_training_time": total_training_time,
            },
        }
        save_pt = args.save_format in ("pt", "both")
        save_hf = args.save_format in ("hf", "both")

        model_state_for_save = None
        optimizer_state = None
        if save_pt or save_hf:
            if args.ema_eval and ema_state is not None:
                with apply_ema_weights(orig_model, ema_state):
                    model_state_for_save = {k: v.detach().clone() for k, v in orig_model.state_dict().items()}
            else:
                model_state_for_save = orig_model.state_dict()
            optimizer_state = {
                "adamw": optimizer.state_dict(),
                "muon": muon_optimizer.state_dict() if muon_optimizer is not None else None,
                "ema": ema_state,
                "dflash_draft": (draft_model.module if ddp else draft_model).state_dict() if draft_model is not None else None,
                "dflash_opt": draft_optimizer.state_dict() if draft_optimizer is not None else None,
            }

        if save_pt:
            if ddp:
                for save_rank in range(ddp_world_size):
                    if ddp_rank == save_rank:
                        save_checkpoint(checkpoint_dir, step, model_state_for_save, optimizer_state, meta_dict, rank=ddp_rank)
                    torch.distributed.barrier()
            else:
                save_checkpoint(checkpoint_dir, step, model_state_for_save, optimizer_state, meta_dict, rank=ddp_rank)

        if save_hf:
            if master_process:
                hf_dir = os.path.join(checkpoint_dir, f"hf_{step:06d}")
                ema_ctx = apply_ema_weights(orig_model, ema_state) if (args.ema_eval and ema_state is not None) else nullcontext()
                with ema_ctx:
                    save_hf_checkpoint(hf_dir, orig_model, tokenizer=tokenizer, optimizer_data=optimizer_state, meta_data=meta_dict)
            if ddp:
                torch.distributed.barrier()

    if last_step:
        break

    # -------------------------------------------------------------------------
    # single training step
    synchronize()
    t0 = time.time()

    for micro_step in range(grad_accum_steps):
        is_last_micro = micro_step == grad_accum_steps - 1
        sync_ctx = model.no_sync() if (ddp and not is_last_micro) else nullcontext()
        with sync_ctx:
            with autocast_ctx:
                loss = model(x, y)
            train_loss = loss.detach()
            loss = loss / grad_accum_steps
            loss.backward()
        if draft_model is not None:
            draft_sync = draft_model.no_sync() if (ddp and not is_last_micro) else nullcontext()
            with draft_sync:
                with autocast_ctx:
                    B = args.dflash_block_size
                    T = x.shape[1]
                    bsz = x.shape[0]
                    nb = args.dflash_num_blocks
                    if T - B > 0 and nb >= 1:
                        if args.dflash_grad_to_target:
                            all_hidden = orig_model.all_hidden_states_with_grad(x)
                        else:
                            all_hidden = orig_model.get_all_hidden_states(x)
                        feat = extract_context_feature(all_hidden, draft_layer_ids)
                        anchors = torch.randint(0, T - B, (bsz, nb), device=device)
                        rng = torch.arange(B, device=device)
                        idx = anchors[:, :, None] + rng[None, None, :]
                        bf = idx.reshape(bsz, nb * B)
                        ctx = torch.gather(feat, 1, bf[:, :, None].expand(-1, -1, feat.shape[-1])).reshape(bsz * nb, B, feat.shape[-1])
                        blk = torch.gather(x, 1, bf).reshape(bsz, nb, B).clone()
                        blk[:, :, 1:] = dflash_mask_token_id
                        blk = blk.reshape(bsz * nb, B)
                        emb = orig_model.model.embed_tokens(blk)
                        noise_emb = emb if args.dflash_grad_to_target else emb.detach()
                        pos = torch.arange(B, device=device).repeat(2).unsqueeze(0).expand(bsz * nb, 2 * B)
                        h = draft_model(target_hidden=ctx, noise_embedding=noise_emb, position_ids=pos, use_cache=False, is_causal=False)
                        lm_w = orig_model.lm_head.weight if args.dflash_grad_to_target else orig_model.lm_head.weight.detach()
                        draft_logits = F.linear(h, lm_w).float()
                        labels = torch.gather(y, 1, bf).reshape(bsz * nb, B)
                        per_tok = F.cross_entropy(draft_logits.reshape(-1, draft_logits.size(-1)), labels.reshape(-1), reduction="none").reshape(bsz * nb, B)
                        wk = torch.exp(-rng.float() / max(1e-6, args.dflash_gamma))
                        draft_loss = (per_tok * wk[None, :]).sum() / (per_tok.shape[0] * wk.sum())
                        (args.dflash_weight * draft_loss / grad_accum_steps).backward()
        x, y, dataloader_state_dict = next(train_loader)

    lrm = get_lr_multiplier(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm

    if muon_optimizer is not None:
        muon_momentum = get_muon_momentum(step)
        muon_weight_decay = get_weight_decay(step)
        for group in muon_optimizer.param_groups:
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
            group["lr"] = group["initial_lr"] * lrm

    if args.grad_max_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_max_norm)
    optimizer.step()
    if muon_optimizer is not None:
        muon_optimizer.step()
    if draft_optimizer is not None:
        for group in draft_optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
        if args.grad_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(draft_model.parameters(), max_norm=args.grad_max_norm)
        draft_optimizer.step()
        draft_optimizer.zero_grad(set_to_none=True)
    model.zero_grad(set_to_none=True)
    if ema_state is not None:
        update_ema_state(orig_model, ema_state, args.ema_decay)
    train_loss_f = train_loss.item()
    synchronize()
    t1 = time.time()
    dt = t1 - t0

    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    mfu = 0.0

    if step > 10:
        total_training_time += dt
    steps_done = step - 10
    if steps_done > 0:
        avg_time_per_step = total_training_time / steps_done
        eta_seconds = (num_iterations - step) * avg_time_per_step
        eta_str = f" | eta: {eta_seconds/60:.1f}m"
    else:
        eta_str = ""
    epoch = dataloader_state_dict["epoch"]
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | bf16_mfu: {mfu:.2f} | epoch: {epoch} | total time: {total_training_time/60:.2f}m{eta_str}")
    if step % 100 == 0:
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lr * lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/epoch": epoch,
        })

    first_step_of_run = (step == 0) or (resuming and step == args.resume_from_step)
    step += 1

    if first_step_of_run:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif step % 5000 == 0:
        gc.collect()

print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
if val_bpb is not None:
    print0(f"Minimum validation bpb: {min_val_bpb:.6f}")

from nanochat.report import get_report
get_report().log(section="Qwen3.5 base model training", data=[
    user_config,
    {
        "Number of parameters": num_params,
        "Calculated number of iterations": num_iterations,
        "Number of training tokens": total_tokens,
        "DDP world size": ddp_world_size,
        "warmup_ratio": args.warmup_ratio,
        "warmdown_ratio": args.warmdown_ratio,
        "final_lr_frac": args.final_lr_frac,
        "layer_types": model_config.layer_types,
    },
    {
        "Minimum validation bpb": min_val_bpb if val_bpb is not None else None,
        "Final validation bpb": val_bpb,
        "CORE metric estimate": results.get("core_metric", None),
        "Total training time": f"{total_training_time/60:.2f}m",
        "Peak memory usage": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
    }
])

wandb_run.finish()
compute_cleanup()

