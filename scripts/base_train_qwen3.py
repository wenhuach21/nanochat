"""
Train model. From root directory of the project, run as:

python -m scripts.base_train

or distributed as:

torchrun --nproc_per_node=8 -m scripts.base_train

If you are only on CPU/Macbook, you'll want to train a much much smaller LLM. Example:
python -m scripts.base_train --depth=4 --max-seq-len=512 --device-batch-size=1 --eval-tokens=512 --core-metric-every=-1 --total-batch-size=512 --num-iterations=20
"""

import os


os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import json
import time
import math
import argparse
from dataclasses import asdict
from contextlib import nullcontext, contextmanager

import wandb
import torch
import torch.nn.functional as F

from nanochat.gpt import GPT, GPTConfig
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit, tokenizing_distributed_data_loader_with_state_bos_bestfit
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type, get_peak_flops
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from nanochat.flash_attention import HAS_FA3
from scripts.base_eval import evaluate_core
print_banner()

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Pretrain base model")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# FP8 training
parser.add_argument("--fp8", action="store_true", help="enable FP8 training (requires H100+ GPU and torchao)")
parser.add_argument("--fp8-recipe", type=str, default="tensorwise", choices=["rowwise", "tensorwise"], help="FP8 scaling recipe: tensorwise (faster, recommended) or rowwise (more accurate but slower)")
# Model architecture
parser.add_argument("--depth", type=int, default=14, help="depth of the Transformer model")
# parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
parser.add_argument("--hidden-size", type=int, default=1024, help="hidden size")
parser.add_argument("--head-dim", type=int, default=128, help="target head dimension for attention")
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
parser.add_argument("--window-pattern", type=str, default="SSSL", help="sliding window pattern tiled across layers: L=full, S=half context (e.g. 'SSL')")
# Training horizon (only one used, in order of precedence)
parser.add_argument("--num-iterations", type=int, default=-1, help="explicit number of optimization steps (-1 = disable)")
parser.add_argument("--target-flops", type=float, default=-1.0, help="calculate num_iterations to reach target_flops (-1 = disable)")
parser.add_argument("--target-param-data-ratio", type=float, default=10.5, help="calculate num_iterations to maintain data:param ratio (Chinchilla=20, -1 = disable)")
# Optimization
parser.add_argument("--device-batch-size", type=int, default=16, help="per-device batch size. good number to reduce to 16,8,4,... if you OOM on VRAM.")
# parser.add_argument("--global-batch-size", type=int, default=256, help="all device batch size. good number to reduce to 16,8,4,... if you OOM on VRAM.")
parser.add_argument("--total-batch-size", type=int, default=-1, help="total batch size in tokens. decent numbers are e.g. 524288. (-1 = auto-compute optimal)")
parser.add_argument("--lr", type=float, default=3e-3, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--embedding-lr", type=float, default=0.3, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="learning rate for unembedding (lm_head) parameters (Adam)")
parser.add_argument("--weight-decay", type=float, default=0.2, help="cautious weight decay for the Muon optimizer (for weights)")
parser.add_argument("--muon-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--optimizer-mode", type=str, default="hybrid", choices=["hybrid", "adamw"], help="optimizer setup: hybrid(AdamW+Muon) or adamw")
parser.add_argument("--ema-decay", type=float, default=0.0, help="EMA decay for model parameters (0 disables EMA, e.g. 0.999)")
parser.add_argument("--ema-eval", action="store_true", help="evaluate/sample/save with EMA parameters when EMA is enabled")
parser.add_argument("--mtp-num-heads", type=int, default=0, help="number of auxiliary MTP heads (0 disables MTP)")
parser.add_argument("--mtp-weight", type=float, default=0.0, help="weight for MTP auxiliary loss")
# DFLASH joint pretraining (block-diffusion draft trained online with the base model)
parser.add_argument("--dflash-enable", action="store_true", help="train a DFLASH draft jointly during pretraining")
parser.add_argument("--dflash-layers", type=int, default=1, help="number of DFLASH draft decoder layers")
parser.add_argument("--dflash-block-size", type=int, default=16, help="DFLASH block size")
parser.add_argument("--dflash-weight", type=float, default=0.3, help="weight for DFLASH draft loss")
parser.add_argument("--dflash-mask-token-id", type=int, default=-1, help="DFLASH mask token id (-1 = last vocab id)")
parser.add_argument("--dflash-grad-to-target", action="store_true", help="allow DFLASH loss to backprop into the base model (off = detached)")
parser.add_argument("--dflash-num-blocks", type=int, default=8, help="number of randomly-anchored blocks per sequence")
parser.add_argument("--dflash-gamma", type=float, default=4.0, help="exp-decay loss weighting gamma (w_k=exp(-(k)/gamma))")
# parser.add_argument("--scalar-lr", type=float, default=0.5, help="learning rate for scalars (resid_lambdas, x0_lambdas)")
parser.add_argument("--adam-beta1", type=float, default=0.9, help="Adam beta1 for embedding/unembedding")
parser.add_argument("--adam-beta2", type=float, default=0.95, help="Adam beta2 for embedding/unembedding")
parser.add_argument("--warmup-ratio", type=float, default=0.0, help="ratio of iterations for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.5, help="ratio of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.0, help="final LR as fraction of initial LR")
parser.add_argument("--grad-max-norm", type=float, default=-1.0, help="clip-grad-norm")
parser.add_argument("--lr-schedule", type=str, default="default", choices=["linear", "cosine", "default"], help="LR decay schedule during warmdown: linear or cosine")
parser.add_argument("--resume-from-step", type=int, default=-1, help="resume training from this step (-1 = disable)")
# Evaluation
parser.add_argument("--eval-every", type=int, default=250, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=40*524288, help="number of tokens to evaluate val loss on")
parser.add_argument("--core-metric-every", type=int, default=5000, help="evaluate CORE metric every N steps (-1 = disable)")
parser.add_argument("--core-metric-max-per-task", type=int, default=-1, help="examples per task for CORE metric")
parser.add_argument("--sample-every", type=int, default=2000, help="sample from model every N steps (-1 = disable)")
parser.add_argument("--save-every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")
# Output
parser.add_argument("--model-tag", type=str, default=None, help="override model tag for checkpoint directory name")
args = parser.parse_args()
user_config = vars(args).copy()  # for logging
# -----------------------------------------------------------------------------
# Compute init and wandb logging

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')  # MFU not meaningful for CPU/MPS

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=args.run, config=user_config)

# Flash Attention status
if HAS_FA3:
    print0("✓ Using Flash Attention 3 (Hopper GPU detected), efficient, new and awesome.")
else:
    print0("!" * 80)
    print0("WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback")
    print0("WARNING: Training will be less efficient without FA3")
    if args.window_pattern != "L":
        print0(f"WARNING: SDPA has no support for sliding window attention (window_pattern='{args.window_pattern}'). Your GPU utilization will be terrible.")
        print0("WARNING: Recommend using --window-pattern L for full context attention without alternating sliding window patterns.")
    print0("!" * 80)

# -----------------------------------------------------------------------------
# Tokenizer will be useful for evaluation and also we need the vocab size to init the model
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")
# -----------------------------------------------------------------------------
def build_model_meta(depth):
    # from transformers import AutoConfig,AutoModelForCausalLM
    # config = AutoConfig.from_pretrained
    from nanochat.qwen3 import Qwen3Config,Qwen3ForCausalLM
    #TODO set tie_word_embeddings for easy train first
    # tie_word_embeddings=Ture
    ## layers= 28
    layers = depth
    hidden_size = args.hidden_size
    head_dim = args.head_dim
    num_attention_heads = hidden_size//head_dim
    intermediate_size = hidden_size * 3
    config = Qwen3Config(head_dim=head_dim, hidden_act="silu", hidden_size=hidden_size,initializer_range=0.02,intermediate_size=intermediate_size,max_position_embeddings=args.max_seq_len*10,max_window_layers=layers,model_type="qwen3",
                num_hidden_layers=layers,num_key_value_heads=num_attention_heads,rms_norm_eps=1e-6,tie_word_embeddings=False,vocab_size=vocab_size,use_cache=False, num_attention_heads=num_attention_heads)
    config.mtp_num_heads = max(0, int(args.mtp_num_heads))
    config.mtp_weight = max(0.0, float(args.mtp_weight))
    with torch.device("meta"):
        model_meta = Qwen3ForCausalLM(config)
    return model_meta

# Build the model, move to device, init the weights
model = build_model_meta(args.depth) # 1) Build on meta device (only shapes/dtypes, no data)
model.to(torch.float32)
print0(model.__class__.__name__)
model_config = model.config

print0(model_config)
model.to_empty(device=device) # 2) All tensors get storage on target device but with uninitialized (garbage) data
model.post_init() # 3) All tensors get initialized
model.re_init_weights()
# If we are resuming, overwrite the model parameters with those of the checkpoint
base_dir = get_base_dir()
output_dirname = args.model_tag if args.model_tag else f"d{args.depth}" # e.g. d12
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
resuming = args.resume_from_step != -1
optimizer_data = None
meta_data = None
if resuming:#  TODO this have bugs
    print0(f"Resuming optimization from step {args.resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, args.resume_from_step, device, load_optimizer=True, rank=ddp_rank)
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data # free up this memory after the copy

# -----------------------------------------------------------------------------
# FP8 training initialization and management (this has to be done before torch.compile)

# Convert Linear layers to Float8Linear if --fp8 is set
if args.fp8:
    if device_type != "cuda":
        print0("Warning: FP8 training requires CUDA, ignoring --fp8 flag")
    else:
        # our custom fp8 is simpler than torchao, written for exact API compatibility
        from nanochat.fp8 import Float8LinearConfig, convert_to_float8_training
        # from torchao.float8 import Float8LinearConfig, convert_to_float8_training
        import torch.nn as nn

        # Filter: dims must be divisible by 16 (FP8 hardware requirement) large enough
        def fp8_module_filter(mod: nn.Module, fqn: str) -> bool:
            if not isinstance(mod, nn.Linear):
                return False
            if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                return False
            if min(mod.in_features, mod.out_features) < 128:
                return False
            return True

        fp8_config = Float8LinearConfig.from_recipe_name(args.fp8_recipe)
        num_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
        convert_to_float8_training(model, config=fp8_config, module_filter_fn=fp8_module_filter)
        num_fp8 = sum(1 for m in model.modules() if 'Float8' in type(m).__name__)
        num_skipped = num_linear - num_fp8
        print0(f"✓ FP8 training enabled ({args.fp8_recipe} scaling) - converted {num_fp8}/{num_linear} linear layers, skipped {num_skipped} (too small)")

# Context manager to temporarily disable FP8 so that model evaluation remains in BF16
@contextmanager
def disable_fp8(model):
    """Temporarily swap Float8Linear modules with nn.Linear for BF16 evaluation.

    CastConfig is a frozen dataclass, so we can't mutate scaling_type. Instead,
    we swap out Float8Linear modules entirely and restore them after.
    """
    import torch.nn as nn

    # Find all Float8Linear modules and their locations
    fp8_locations = []  # list of (parent_module, attr_name, fp8_module)
    for name, module in model.named_modules():
        if 'Float8' in type(module).__name__:
            if '.' in name:
                parent_name, attr_name = name.rsplit('.', 1)
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name
            fp8_locations.append((parent, attr_name, module))

    if not fp8_locations:
        yield  # No FP8 modules, nothing to do
        return

    # Swap Float8Linear -> nn.Linear (shares the same weight tensor, no copy)
    for parent, attr_name, fp8_module in fp8_locations:
        linear = nn.Linear(
            fp8_module.in_features,
            fp8_module.out_features,
            bias=fp8_module.bias is not None,
            device=fp8_module.weight.device,
            dtype=fp8_module.weight.dtype,
        )
        linear.weight = fp8_module.weight  # share, don't copy
        if fp8_module.bias is not None:
            linear.bias = fp8_module.bias
        setattr(parent, attr_name, linear)

    try:
        yield
    finally:
        # Restore Float8Linear modules
        for parent, attr_name, fp8_module in fp8_locations:
            setattr(parent, attr_name, fp8_module)


def init_ema_state(model):
    return {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if p.requires_grad
    }


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
# Compile the model

orig_model = model # original, uncompiled model, for saving raw model state_dict and for inference/evaluation (because the shapes may change shape)
# model = torch.compile(model, dynamic=False) # may cause nan issue the inputs to model will never change shape so dynamic=False is safe
print0(orig_model.dtype)

# Wrap the model in DistributedDataParallel for multi-GPU training.
# CRITICAL: the plain torch.optim.AdamW / torch.optim.Muon used below do NOT synchronize
# gradients across ranks. Without DDP, every rank would train an independent model on its own
# data shard and only rank0's (severely under-trained) model would be saved/evaluated.
# DDP all-reduces (averages) gradients across ranks during backward, fixing this.
if ddp:
    from torch.nn.parallel import DistributedDataParallel as DDP
    model = DDP(model, device_ids=[ddp_local_rank], broadcast_buffers=True)
    print0(f"Wrapped model in DistributedDataParallel (world_size={ddp_world_size})")

# -----------------------------------------------------------------------------
# DFLASH draft model (jointly trained during pretraining)
draft_model = None
draft_layer_ids = None
dflash_mask_token_id = None
if args.dflash_enable:
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dflash"))
    from dflash.model import DFlashDraftModel, build_target_layer_ids, extract_context_feature
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Config as HFQwen3Config
    num_heads = args.hidden_size // args.head_dim
    draft_cfg = HFQwen3Config(
        head_dim=args.head_dim, hidden_act="silu", hidden_size=args.hidden_size, initializer_range=0.02,
        intermediate_size=args.hidden_size * 3, max_position_embeddings=args.max_seq_len * 10,
        max_window_layers=args.dflash_layers, num_hidden_layers=args.dflash_layers,
        num_key_value_heads=num_heads, rms_norm_eps=1e-6, tie_word_embeddings=False,
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
# Scaling laws and muP extrapolations to determine the optimal training horizon, batch size, learning rates, weight decay.

# Get the parameter counts of our model
# num_params = sum(p.numel() for p in model.parameters())
#
#
# print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# 1) Use scaling laws to determine the optimal training horizon in tokens
# The compute-optimal models satisfy the Tokens:Params ratio of --target-param-data-ratio (derived experimentally via scaling laws analysis).
# We've already initialized the model so we have Params. Optimal Tokens is now simply target-param-data-ratio * Params
# def get_scaling_params(m):
#     # As for which params to use exactly, transformer matrices + lm_head gives cleanest scaling laws (see dev/LOG.md Jan 27, 2026)
#     params_counts = m.num_scaling_params()
#     scaling_params = params_counts['transformer_matrices'] + params_counts['lm_head']
#     return scaling_params
# num_scaling_params = get_scaling_params(model)
num_params = sum(p.numel() for p in model.parameters())
print0(f"Parameters: {num_params/1e9:.3f}B")

target_tokens = int(args.target_param_data_ratio * num_params) # optimal tokens for the model we are about to train
B_REF = 256*2048
d14_model = build_model_meta(14)
d14_params = sum(p.numel() for p in d14_model.parameters())
D_REF = num_params/ d14_params # TODO this is different from origin
total_batch_size = args.total_batch_size # user-provided override is possible
if total_batch_size == -1:
    batch_size_ratio =  D_REF
    predicted_batch_size = B_REF * batch_size_ratio ** 0.383
    total_batch_size = 2 ** round(math.log2(predicted_batch_size)) # clamp to nearest power of 2 for efficiency
    print0(f"Auto-computed optimal batch size: {total_batch_size:,} tokens")
# Our reference model is d12, this is where a lot of hyperparameters are tuned and then transfered to higher depths (muP style)
# d12_ref = build_model_meta(12) # creates the model on meta device
# D_REF = args.target_param_data_ratio * get_scaling_params(d12_ref) # compute-optimal d12 training horizon in tokens (measured empirically)
# B_REF = 2**19 # optimal batch size at d12 ~= 524,288 tokens (measured empirically)
#
# # 2) Now that we have the token horizon, we can calculate the optimal batch size
# # We follow the Power Lines paper (Bopt ∝ D^0.383), ref: https://arxiv.org/abs/2505.13738
# # The optimal batch size grows as approximately D^0.383, so e.g. if D doubles from d12 to d24, B should grow by 2^0.383 ≈ 1.3x.
# total_batch_size = args.total_batch_size # user-provided override is possible
# if total_batch_size == -1:
#     batch_size_ratio = target_tokens / D_REF
#     predicted_batch_size = B_REF * batch_size_ratio ** 0.383
#     total_batch_size = 2 ** round(math.log2(predicted_batch_size)) # clamp to nearest power of 2 for efficiency
#     print0(f"Auto-computed optimal batch size: {total_batch_size:,} tokens")

# 3) Knowing the batch size, we can now calculate a learning rate correction (bigger batch size allows higher learning rates)
batch_lr_scale = 1.0
# batch_ratio = total_batch_size / B_REF # B/B_ref
# if batch_ratio != 1.0:
#     # SGD: linear scaling with batch size is standard (not used in nanochat)
#     # AdamW: sqrt scaling is standard: η ∝ √(B/B_ref)
#     # Muon: we will use the same scaling for Muon as for AdamW: η ∝ √(B/B_ref) (not studied carefully, assumption!)
#     batch_lr_scale = batch_ratio ** 0.5 # η ∝ √(B/B_ref)
#     print0(f"Scaling LRs by {batch_lr_scale:.4f} for batch size {total_batch_size:,} (reference: {B_REF:,})")

# 4) Knowing the batch size and the token horizon, we can now calculate the appropriate weight decay scaling
# We adopt the T_epoch framework from https://arxiv.org/abs/2405.13698
# Central idea of the paper is that T_epoch = B/(η·λ·D) should remain constant.
# Above, we used learning rate scaling η ∝ √(B/B_ref). So it's a matter of ~10 lines of math to derive that to keep T_epoch constant, we need:
# λ = λ_ref · √(B/B_ref) · (D_ref/D)
# Note that these papers study AdamW, *not* Muon. We are blindly following AdamW theory for scaling hoping it ~works for Muon too.

# num_scaling_params = get_scaling_params(model)
# target_tokens = int(args.target_param_data_ratio * num_scaling_params) # optimal tokens for the model we are about to train
#
# weight_decay_scaled = args.weight_decay * math.sqrt(total_batch_size / B_REF) * (1.0/D_REF)
# if weight_decay_scaled != args.weight_decay:
#     print0(f"Scaling weight decay from {args.weight_decay:.6f} to {weight_decay_scaled:.6f} for depth {args.depth}")
#
# # -----------------------------------------------------------------------------
# # Initialize the Optimizer (combined MuonAdamW: Muon for matrix params, AdamW for rest)
# optimizer = model.setup_optimizer(
#     # AdamW hyperparameters
#     unembedding_lr=args.unembedding_lr * batch_lr_scale,
#     embedding_lr=args.embedding_lr * batch_lr_scale,
#     scalar_lr=args.scalar_lr * batch_lr_scale,
#     adam_betas=(args.adam_beta1, args.adam_beta2),
#     # Muon hyperparameters
#     matrix_lr=args.matrix_lr * batch_lr_scale,
#     weight_decay=weight_decay_scaled,
# )

batch_size = 256
# total_batch_size = args.max_seq_len*batch_size
# num_iterations = 10000 # TODO
lr = args.lr * batch_lr_scale
weight_decay = args.weight_decay
weight_decay_scaled = args.weight_decay* 1.0
# Optimizer
# Split weights in two groups, one with weight decay and the other not.
no_decay = ["bias", "norm.weight"]  # norm.weight change by wenhua
# all_p = [n for n, p in model.named_parameters()]
# print(all_p,flush=True)
no_decay_p = [n for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]
# optimizer_grouped_parameters = [
#     # {
#     #     "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
#     #     "weight_decay": weight_decay,
#     # },
#     {
#         "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
#         "weight_decay": 0.0,
#     },
# ]
optimizer_grouped_parameters = []
base_fan_in = 1024
embed_params=[]
other_params = {}
no_weight_decay = {"params":[],  "weight_decay": 0.0,}
muon_lr = args.muon_lr
matrix_lr = muon_lr if muon_lr>0 else lr
default_group = {"params":[], "lr":matrix_lr}
lm_head_params=[]
adamw_all_params = []


for n,m in model.named_parameters():
    is_no_decay = any(nd in n for nd in no_decay)
    is_embed = "embed" in n
    is_lm_head = "lm_head" in n

    # AdamW-all mode groups (all params are optimized by AdamW)
    if is_no_decay:
        adamw_all_params.append({"params": [m], "lr": args.lr, "weight_decay": 0.0})
    elif is_embed:
        adamw_all_params.append({"params": [m], "lr": args.embedding_lr, "weight_decay": 0.0})
    elif is_lm_head:
        adamw_all_params.append({"params": [m], "lr": args.lr, "weight_decay": 0.0})
    else:
        adamw_all_params.append({"params": [m], "lr": args.lr, "weight_decay": weight_decay_scaled})

    if any(nd in n for nd in no_decay):
        no_weight_decay["params"].append(m)
    elif "lm_head" in n:
        lm_head_params.append({"params": [m], "lr": args.lr, "weight_decay": 0.0, })
    elif "embed" in n:
        embed_params.append({"params":[m],"lr":args.embedding_lr, "weight_decay": 0.0,}) # this could be change
    elif "norm" not in n and len(m.shape)==2 and m.shape[-1]!=base_fan_in:
        # matrix params whose fan_in != base_fan_in (e.g. down_proj: in=intermediate_size) -> Muon
        fan_in = m.shape[-1]
        if fan_in not in other_params:
            other_params[fan_in] = {"lr": matrix_lr, "params": [m]}
        else:
            other_params[fan_in]["params"].append(m)
    else:
        default_group["params"].append(m)
optimizer_grouped_parameters.append(no_weight_decay)
optimizer_grouped_parameters.extend(embed_params)
optimizer_grouped_parameters.extend(lm_head_params)
# optimizer_grouped_parameters.extend(other_params.values())
# optimizer_grouped_parameters.append(default_group)

if args.optimizer_mode == "adamw":
    optimizer = torch.optim.AdamW(
        adamw_all_params,
        lr=lr,
        weight_decay=weight_decay_scaled,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=1e-10,
    )
else:
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=lr,
        weight_decay=weight_decay_scaled,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=1e-10,
    )  # betas are changed by wenhua
print0(f"Optimizer mode: {args.optimizer_mode}")

# Only create muon_optimizer if muon_lr >= 0
muon_optimizer = None
if args.optimizer_mode == "hybrid" and args.muon_lr >= 0:
    moun_parameters=[]
    moun_parameters.extend(other_params.values())
    moun_parameters.append(default_group)
    muon_optimizer = torch.optim.Muon(moun_parameters, weight_decay=weight_decay_scaled, adjust_lr_fn="match_rms_adamw",lr=muon_lr)
    for group in muon_optimizer.param_groups:
        group["initial_lr"] = group["lr"]

for group in optimizer.param_groups:
    group["initial_lr"] = group["lr"]

ema_state = None
if args.ema_decay > 0:
    ema_state = init_ema_state(orig_model)
    print0(f"EMA enabled (decay={args.ema_decay})")

# Optimizer for the DFLASH draft (trained jointly, kept separate from base optimizers)
draft_optimizer = None
if draft_model is not None:
    draft_optimizer = torch.optim.AdamW(draft_model.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
    for group in draft_optimizer.param_groups:
        group["initial_lr"] = group["lr"]

if resuming:
    # Backward compatible optimizer restore:
    # - old checkpoints: optimizer_data is AdamW state_dict
    # - new checkpoints: optimizer_data is {"adamw": ..., "muon": ...}
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
# Initialize the DataLoaders for train/val
dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device, resume_state_dict=dataloader_resume_state_dict)
build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device)
x, y, dataloader_state_dict = next(train_loader) # kick off load of the very first batch of data

# -----------------------------------------------------------------------------
# Calculate the number of iterations we will train for and set up the various schedulers

# num_iterations: either it is given, or from target flops, or from target data:param ratio (in that order)
# assert args.num_iterations > 0 or args.target_param_data_ratio > 0 or args.target_flops > 0
if args.num_iterations > 0:
    # Override num_iterations to a specific value if given
    num_iterations = args.num_iterations
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
# elif args.target_flops > 0:
#     # Calculate the number of iterations from the target flops (used in scaling laws analysis, e.g. runs/scaling_laws.sh)
#     num_iterations = round(args.target_flops / (num_flops_per_token * total_batch_size))
#     print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif args.target_param_data_ratio > 0:
    # Calculate the number of iterations from the target param data ratio (the most common use case)
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")


total_tokens = total_batch_size * num_iterations # the actual number of tokens we will train for
print0(f"Total number of training tokens: {total_tokens:,}")
# print0(f"Tokens : Scaling params ratio: {total_batch_size * num_iterations / num_scaling_params:.2f}") # e.g. Chinchilla was ~20
# print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# Learning rate schedule (linear warmup, constant, linear/cosine warmdown)
def get_lr_multiplier(it):
    warmup_iters = round(args.warmup_ratio * num_iterations)
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    # 1) warmup: linear ramp from 0 to 1
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    # 3) warmdown: linear or cosine decay to final_lr_frac
    else:
        progress = 1.0-(it-warmup_iters) / (num_iterations-warmup_iters)  #  1.0 -> 0.0
        if args.lr_schedule == "cosine":
            # cosine annealing: smooth decay, never fully zero until the end
            decay = (1 + math.cos(math.pi * (1.0-progress))) / 2  # 1.0 -> 0.0
            return args.final_lr_frac + (1.0 - args.final_lr_frac) * decay
        elif args.lr_schedule == "linear":
            # linear decay (default)
            decay = progress
            return args.final_lr_frac + (1.0 - args.final_lr_frac) * decay
        else:
            if it <= num_iterations - warmdown_iters:
                return 1.0
            progress = (num_iterations - it) / warmdown_iters
            return progress * 1.0 + (1 - progress) * args.final_lr_frac



# Momentum scheduler for Muon optimizer (warms up to 0.95 over the first 300 steps)
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum

# Weight decay scheduler for Muon optimizer (linearly decays to zero over the course of training)
def get_weight_decay(it):
    return weight_decay_scaled * (1 - it / num_iterations)

# -----------------------------------------------------------------------------
# Training loop

# Loop state (variables updated by the training loop)
if not resuming:
    step = 0
    val_bpb = None # will be set if eval_every > 0
    min_val_bpb = float("inf")
    smooth_train_loss = 0 # EMA of training loss
    total_training_time = 0 # total wall-clock time of training
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_bpb = meta_data["val_bpb"]
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]

# Figure out the needed gradient accumulation micro-steps to reach the desired total batch size per step
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# Go!
while True:
    last_step = step == num_iterations # loop runs num_iterations+1 times so that we can eval/save at the end
    # flops_so_far = num_flops_per_token * total_batch_size * step
    flops_so_far = 0
    #
    # # once in a while: evaluate the val bpb (all ranks participate)
    # if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
    #     model.eval()
    #     val_loader = build_val_loader()
    #     eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
    #     with disable_fp8(model), autocast_ctx:
    #         val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
    #     print0(f"Step {step:05d} | Validation bpb: {val_bpb:.6f}")
    #     if val_bpb < min_val_bpb:
    #         min_val_bpb = val_bpb
    #     wandb_run.log({
    #         "step": step,
    #         "total_training_flops": flops_so_far,
    #         "total_training_time": total_training_time,
    #         "val/bpb": val_bpb,
    #     })
    #     model.train()

    # once in a while: estimate the CORE metric (all ranks participate)
    # use the original uncompiled model because the inputs keep changing shape
    # disable FP8 for evaluation to use BF16 for more consistent/accurate results
    results = {}
    if args.core_metric_every > 0 and (last_step or (step > 0 and step % args.core_metric_every == 0)):
        model.eval()
        ema_ctx = apply_ema_weights(orig_model, ema_state) if (args.ema_eval and ema_state is not None) else nullcontext()
        with ema_ctx:
            with disable_fp8(orig_model), autocast_ctx:
                results = evaluate_core(orig_model, tokenizer, device, max_per_task=args.core_metric_max_per_task)
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "core_metric": results["core_metric"],
            "centered_results": results["centered_results"],
        })
        model.train()

    # once in a while: sample from the model (only on master process)
    # use the original uncompiled model because the inputs keep changing shape
    if args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):
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
        engine = Engine(orig_model, tokenizer) # use orig_model to avoid recompilation
        ema_ctx = apply_ema_weights(orig_model, ema_state) if (args.ema_eval and ema_state is not None) else nullcontext()
        with ema_ctx:
            for prompt in prompts:
                tokens = tokenizer(prompt, prepend="<|bos|>")
                with disable_fp8(orig_model), autocast_ctx:
                    sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
                print0(tokenizer.decode(sample[0]))
        model.train()

    # save checkpoint: at the end of the run, or every save_every steps, except at the first step or the resume step
    if last_step or (step > 0 and step != args.resume_from_step and args.save_every > 0 and step % args.save_every == 0):
        meta_dict = { # metadata saved as json
            "step": step,
            "val_bpb": val_bpb, # loss at last step
            # "model_config": model_config_kwargs,
            "user_config": user_config, # inputs to the training script
            "device_batch_size": args.device_batch_size,
            "max_seq_len": args.max_seq_len,
            "total_batch_size": total_batch_size,
            "dataloader_state_dict": dataloader_state_dict,
            "loop_state": { # all loop state (other than step) so that we can resume training
                "min_val_bpb": min_val_bpb,
                "smooth_train_loss": smooth_train_loss,
                "total_training_time": total_training_time,
            },
        }
        if args.ema_eval and ema_state is not None:
            with apply_ema_weights(orig_model, ema_state):
                model_state_for_save = {
                    k: v.detach().clone()
                    for k, v in orig_model.state_dict().items()
                }
        else:
            model_state_for_save = orig_model.state_dict()

        optimizer_state = {
            "adamw": optimizer.state_dict(),
            "muon": muon_optimizer.state_dict() if muon_optimizer is not None else None,
            "ema": ema_state,
            "dflash_draft": (draft_model.module if ddp else draft_model).state_dict() if draft_model is not None else None,
            "dflash_opt": draft_optimizer.state_dict() if draft_optimizer is not None else None,
        }

        if ddp:
            # Save sequentially across ranks to avoid concurrent heavy writes to the same filesystem.
            for save_rank in range(ddp_world_size):
                if ddp_rank == save_rank:
                    save_checkpoint(
                        checkpoint_dir,
                        step,
                        model_state_for_save, # model parameters
                        optimizer_state, # optimizer state(s), saved via torch.save
                        meta_dict,
                        rank=ddp_rank,
                    )
                torch.distributed.barrier()
        else:
            save_checkpoint(
                checkpoint_dir,
                step,
                model_state_for_save, # model parameters
                optimizer_state, # optimizer state(s), saved via torch.save
                meta_dict,
                rank=ddp_rank,
            )

    # termination conditions (TODO: possibly also add loss explosions etc.)
    if last_step:
        break

    # -------------------------------------------------------------------------
    # single training step
    # evaluate the gradient
    synchronize()
    t0 = time.time()

    for micro_step in range(grad_accum_steps):
        # Under DDP with gradient accumulation, only synchronize (all-reduce) gradients on the
        # final micro-step. no_sync() skips the all-reduce on the intermediate micro-steps; the
        # final backward then reduces the fully-accumulated gradient -> 1 comm per optimizer step.
        is_last_micro = micro_step == grad_accum_steps - 1
        sync_ctx = model.no_sync() if (ddp and not is_last_micro) else nullcontext()
        with sync_ctx:
            with autocast_ctx:
                loss = model(x, y)
            train_loss = loss.detach() # for logging
            loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
            loss.backward()
        # DFLASH draft loss: predict masked blocks from detached target hidden states (online)
        if draft_model is not None:
            draft_sync = draft_model.no_sync() if (ddp and not is_last_micro) else nullcontext()
            with draft_sync:
                with autocast_ctx:
                    B = args.dflash_block_size
                    T = x.shape[1]
                    bsz = x.shape[0]
                    nb = args.dflash_num_blocks
                    # need a clean anchor + (B-1) future tokens, so anchor in [0, T-B)
                    if T - B > 0 and nb >= 1:
                        if args.dflash_grad_to_target:
                            all_hidden = orig_model.all_hidden_states_with_grad(x)
                        else:
                            all_hidden = orig_model.get_all_hidden_states(x)
                        feat = extract_context_feature(all_hidden, draft_layer_ids)  # [bsz, T, H*L]
                        # randomly sample nb anchor positions per sequence (block = anchor..anchor+B-1)
                        anchors = torch.randint(0, T - B, (bsz, nb), device=device)  # [bsz, nb]
                        rng = torch.arange(B, device=device)
                        idx = anchors[:, :, None] + rng[None, None, :]  # [bsz, nb, B]
                        bf = idx.reshape(bsz, nb * B)
                        ctx = torch.gather(feat, 1, bf[:, :, None].expand(-1, -1, feat.shape[-1])).reshape(bsz * nb, B, feat.shape[-1])
                        blk = torch.gather(x, 1, bf).reshape(bsz, nb, B).clone()
                        blk[:, :, 1:] = dflash_mask_token_id  # keep clean anchor, mask the rest
                        blk = blk.reshape(bsz * nb, B)
                        emb = orig_model.model.embed_tokens(blk)
                        noise_emb = emb if args.dflash_grad_to_target else emb.detach()
                        pos = torch.arange(B, device=device).repeat(2).unsqueeze(0).expand(bsz * nb, 2 * B)
                        h = draft_model(target_hidden=ctx, noise_embedding=noise_emb, position_ids=pos, use_cache=False, is_causal=False)
                        lm_w = orig_model.lm_head.weight if args.dflash_grad_to_target else orig_model.lm_head.weight.detach()
                        draft_logits = F.linear(h, lm_w).float()
                        labels = torch.gather(y, 1, bf).reshape(bsz * nb, B)  # next-token targets per position
                        per_tok = F.cross_entropy(draft_logits.reshape(-1, draft_logits.size(-1)), labels.reshape(-1), reduction="none").reshape(bsz * nb, B)
                        wk = torch.exp(-rng.float() / max(1e-6, args.dflash_gamma))  # [B], emphasize early positions
                        draft_loss = (per_tok * wk[None, :]).sum() / (per_tok.shape[0] * wk.sum())
                        (args.dflash_weight * draft_loss / grad_accum_steps).backward()
        x, y, dataloader_state_dict = next(train_loader) # prefetch the next batch while the GPU is busy with forward/backward
    # step the optimizer
    lrm = get_lr_multiplier(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm

    # Update muon_optimizer parameters if it exists
    if muon_optimizer is not None:
        muon_momentum = get_muon_momentum(step)
        muon_weight_decay = get_weight_decay(step)
        for group in muon_optimizer.param_groups:
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
            group["lr"] = group["initial_lr"] * lrm

    if args.grad_max_norm>0:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_max_norm)
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
    train_loss_f = train_loss.item() # .item() is a CPU-GPU sync point
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # logging (CPU action only)
    ema_beta = 0.9 # EMA decay factor for some smoothing just for nicer logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # debias the EMA
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    # flops_per_sec = num_flops_per_token * total_batch_size / dt
    # mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    flops_per_sec = 0.0 #TODO hard coded
    mfu  = 0.0 # TODO hard coded
    num_params = 1
    num_flops_per_token = 1

    if step > 10:
        total_training_time += dt # only count the time after the first 10 steps
    # Calculate ETA based on average time per step (excluding first 10 steps)
    steps_done = step - 10
    if steps_done > 0:
        avg_time_per_step = total_training_time / steps_done
        remaining_steps = num_iterations - step
        eta_seconds = remaining_steps * avg_time_per_step
        eta_str = f" | eta: {eta_seconds/60:.1f}m"
    else:
        eta_str = ""
    epoch = dataloader_state_dict["epoch"]
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | bf16_mfu: {mfu:.2f} | epoch: {epoch} | total time: {total_training_time/60:.2f}m{eta_str}")
    if step % 100 == 0:
        log_data = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lr*lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/epoch": epoch,
        }
        wandb_run.log(log_data)

    # state update
    first_step_of_run = (step == 0) or (resuming and step == args.resume_from_step)
    step += 1

    # The garbage collector is sadly a little bit overactive and for some poorly understood reason,
    # it spends ~500ms scanning for cycles quite frequently, just to end up cleaning up very few tiny objects each time.
    # So we manually manage and help it out here
    if first_step_of_run:
        gc.collect() # manually collect a lot of garbage from setup
        gc.freeze() # immediately freeze all currently surviving objects and exclude them from GC
        gc.disable() # nuclear intervention here: disable GC entirely except:
    elif step % 5000 == 0: # every 5000 steps...
        gc.collect() # manually collect, just to be safe for very, very long runs

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
if val_bpb is not None:
    print0(f"Minimum validation bpb: {min_val_bpb:.6f}")

# Log to report
from nanochat.report import get_report
get_report().log(section="Base model training", data=[
    user_config, # CLI args
    { # stats about the training setup
        "Number of parameters": num_params,
        "Number of FLOPs per token": f"{num_flops_per_token:e}",
        "Calculated number of iterations": num_iterations,
        "Number of training tokens": total_tokens,
        # "Tokens : Scaling params ratio": total_batch_size * num_iterations / num_scaling_params,
        "DDP world size": ddp_world_size,
        "warmup_ratio": args.warmup_ratio,
        "warmdown_ratio": args.warmdown_ratio,
        "final_lr_frac": args.final_lr_frac,
    },
    { # stats about training outcomes
        "Minimum validation bpb": min_val_bpb if val_bpb is not None else None,
        "Final validation bpb": val_bpb,
        "CORE metric estimate": results.get("core_metric", None),
        "MFU %": f"{mfu:.2f}%",
        "Total training flops": f"{flops_so_far:e}",
        "Total training time": f"{total_training_time/60:.2f}m",
        "Peak memory usage": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
    }
])

# cleanup
wandb_run.finish() # wandb run finish
compute_cleanup()
