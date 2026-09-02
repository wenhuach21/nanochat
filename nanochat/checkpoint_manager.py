"""
Utilities for saving and loading model/optim/state checkpoints.
"""
import os
import re
import glob
import json
import logging
import tempfile
import torch

from nanochat.common import get_base_dir
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer
from nanochat.common import setup_default_logging

# Set up logging
setup_default_logging()
logger = logging.getLogger(__name__)
def log0(message):
    if int(os.environ.get('RANK', 0)) == 0:
        logger.info(message)

def _patch_missing_config_keys(model_config_kwargs):
    """Add default values for new config keys missing in old checkpoints."""
    # Old models were trained with full context (no sliding window)
    if "window_pattern" not in model_config_kwargs:
        model_config_kwargs["window_pattern"] = "L"
        log0(f"Patching missing window_pattern in model config to 'L'")

def _patch_missing_keys(model_data, model_config):
    """Add default values for new parameters that may be missing in old checkpoints."""
    n_layer = model_config.n_layer
    # resid_lambdas defaults to 1.0 (identity scaling)
    if "resid_lambdas" not in model_data:
        model_data["resid_lambdas"] = torch.ones(n_layer)
        log0(f"Patching missing resid_lambdas in model data to 1.0")
    # x0_lambdas defaults to 0.0 (disabled)
    if "x0_lambdas" not in model_data:
        model_data["x0_lambdas"] = torch.zeros(n_layer)
        log0(f"Patching missing x0_lambdas in model data to 0.0")


def _to_cpu(obj):
    """Recursively move tensors in a nested structure to CPU for safer checkpoint writes."""
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu(v) for v in obj)
    return obj


def _atomic_torch_save(obj, path):
    """Write via a temp file and rename atomically when complete."""
    directory = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".__tmp_", suffix=".pt", dir=directory)
    os.close(fd)
    try:
        # Use legacy serialization format to avoid occasional inline_container zip writer errors.
        torch.save(obj, tmp_path, _use_new_zipfile_serialization=False)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data, rank=0):
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Save the model state parameters
        model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
        _atomic_torch_save(_to_cpu(model_data), model_path)
        logger.info(f"Saved model parameters to: {model_path}")
        # Save the metadata dict as json
        meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)
        logger.info(f"Saved metadata to: {meta_path}")
    # NOTE: multi-GPU training uses DistMuonAdamW, a ZeRO-2 style *sharded* optimizer
    # (see nanochat/optim.py). Each rank holds only its own 1/world_size slice of the
    # optimizer state (AdamW exp_avg/exp_avg_sq for large params, and Muon momentum /
    # second-momentum buffers), so the per-rank shards are NOT identical and NOT
    # interchangeable -- every rank must save its own shard for a correct resume.
    # Only tiny (<1024 elem) AdamW params are replicated across ranks. Because the shard
    # boundaries depend on world_size, a checkpoint can in general only be resumed with the
    # same world_size it was saved with; resuming with a different world_size will not
    # reconstruct the correct state (see load_checkpoint / load_optimizer_state).
    if optimizer_data is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        _atomic_torch_save(_to_cpu(optimizer_data), optimizer_path)
        logger.info(f"Saved optimizer state to: {optimizer_path}")

def _register_custom_model_for_auto_class(m):
    """If `m` is a *custom* (remote-code) model whose class lives outside the installed
    `transformers` package (e.g. nanochat's Qwen3.5 `Qwen3_5ForCausalLM`), register its
    config + model classes for the transformers "auto" classes so that a subsequent
    `save_pretrained` will:
      * copy the standalone modeling/configuration `.py` files into the checkpoint folder, and
      * write an `auto_map` into `config.json`.

    This is what makes the exported folder loadable on any machine with just transformers via
    `AutoModelForCausalLM.from_pretrained(save_dir, trust_remote_code=True)` (and likewise
    `AutoConfig`). Built-in transformers models (e.g. `Qwen3ForCausalLM`) are left untouched.
    """
    model_module = type(m).__module__ or ""
    is_custom = not model_module.startswith("transformers.")
    if not is_custom:
        return
    cfg = getattr(m, "config", None)
    cfg_cls = type(cfg) if cfg is not None else None
    try:
        if cfg_cls is not None and hasattr(cfg_cls, "register_for_auto_class"):
            cfg_cls.register_for_auto_class()  # -> AutoConfig
        if hasattr(type(m), "register_for_auto_class"):
            type(m).register_for_auto_class("AutoModelForCausalLM")
        logger.info(
            f"Registered custom model {type(m).__name__} for auto classes "
            f"(will copy modeling code + write auto_map for trust_remote_code loading)."
        )
    except Exception as e:  # pragma: no cover - best effort
        logger.warning(f"Could not register custom model for auto class ({e}); "
                       f"the exported folder may require the source package to load.")


def _resolve_special_token_ids(tokenizer):
    """Best-effort extract (bos_id, eos_ids, pad_id) from a nanochat tokenizer wrapper
    (RustBPETokenizer / TransformersTokenizer, which expose get_bos_token_id / encode_special)
    or a raw transformers tokenizer.

    For nanochat models the document delimiter is `<|bos|>`, so for a *base* LM that is also the
    natural stop token; we additionally include `<|assistant_end|>` (if present) so the same
    export also stops correctly once chat-finetuned. eos is returned as a list of ids.
    """
    bos_id = None
    eos_ids = None
    pad_id = None
    if hasattr(tokenizer, "get_bos_token_id") and hasattr(tokenizer, "encode_special"):
        try:
            bos_id = tokenizer.get_bos_token_id()
        except Exception:
            bos_id = None

        def _sp(name):
            try:
                i = tokenizer.encode_special(name)
                return int(i) if i is not None else None
            except Exception:
                return None

        assistant_end = _sp("<|assistant_end|>")
        eos_list = []
        for i in (bos_id, assistant_end):
            if i is not None and i not in eos_list:
                eos_list.append(int(i))
        eos_ids = eos_list or None
        pad_id = int(bos_id) if bos_id is not None else None
    else:
        inner = getattr(tokenizer, "tokenizer", tokenizer)
        bos_id = getattr(inner, "bos_token_id", None)
        eos_ids = getattr(inner, "eos_token_id", None)
        pad_id = getattr(inner, "pad_token_id", None)
        if pad_id is None:
            pad_id = bos_id
    return (int(bos_id) if bos_id is not None else None), eos_ids, (int(pad_id) if pad_id is not None else None)


def _sync_special_token_ids_into_model(m, tokenizer):
    """Populate bos/eos/pad token ids on the model config + generation_config from the tokenizer
    so the exported `config.json` and `generation_config.json` carry them (like the original HF
    release), and `model.generate()` knows where to stop out of the box."""
    if tokenizer is None:
        return
    bos_id, eos_ids, pad_id = _resolve_special_token_ids(tokenizer)
    cfg = getattr(m, "config", None)
    gen = getattr(m, "generation_config", None)
    for obj in (cfg, gen):
        if obj is None:
            continue
        if bos_id is not None:
            obj.bos_token_id = bos_id
        if eos_ids is not None:
            obj.eos_token_id = eos_ids
        if pad_id is not None:
            obj.pad_token_id = pad_id
    if gen is not None:
        # It's no longer a bare copy of the model config now that we've set ids explicitly.
        try:
            gen._from_model_config = False
        except Exception:
            pass
    logger.info(f"Synced special token ids into config/generation_config: "
                f"bos={bos_id}, eos={eos_ids}, pad={pad_id}")


def _save_tokenizer_into(save_dir, tokenizer):
    """Save a tokenizer into `save_dir` in transformers format, handling nanochat wrappers.

    Accepts either a raw transformers tokenizer (has `save_pretrained`), a nanochat wrapper
    exposing an inner `.tokenizer` (e.g. `TransformersTokenizer`), or a wrapper exposing a
    transformers-export `save_pretrained` (e.g. `RustBPETokenizer`). Ensures
    `AutoTokenizer.from_pretrained(save_dir)` works.
    """
    if tokenizer is None:
        return
    # 1) nanochat wrapper around a real transformers tokenizer -> use the inner one directly
    inner = getattr(tokenizer, "tokenizer", None)
    if inner is not None and hasattr(inner, "save_pretrained"):
        inner.save_pretrained(save_dir)
        logger.info(f"Saved transformers-compatible tokenizer to: {save_dir}")
        return
    # 2) object that already knows how to export a transformers tokenizer folder
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(save_dir)
        logger.info(f"Saved transformers-compatible tokenizer to: {save_dir}")
        return
    logger.warning(
        f"Tokenizer of type {type(tokenizer).__name__} could not be exported "
        f"(no save_pretrained / inner .tokenizer); skipping."
    )


def save_hf_checkpoint(save_dir, model, tokenizer=None, optimizer_data=None, meta_data=None, dtype=None, safe_serialization=True):
    """Save a model (and optionally the tokenizer) in a transformers-compatible format.

    Produces `config.json` + `model.safetensors` (via `PreTrainedModel.save_pretrained`) so the
    result is directly loadable with `AutoModelForCausalLM.from_pretrained(save_dir)`. For custom
    (remote-code) models such as nanochat's Qwen3.5 the standalone modeling/configuration `.py`
    files are also copied into the folder and an `auto_map` is written into `config.json`, so the
    folder loads with `AutoModelForCausalLM.from_pretrained(save_dir, trust_remote_code=True)` on
    any machine with just transformers. When a tokenizer is given it is exported into the same
    directory so `AutoTokenizer.from_pretrained(save_dir)` works too.

    Optionally also saves the optimizer state (`optim.pt`) and training metadata (`meta.json`)
    alongside the model so that training can still be resumed from an HF-format checkpoint
    (see `load_hf_checkpoint` / `load_checkpoint_any`).
    """
    os.makedirs(save_dir, exist_ok=True)
    # Unwrap common wrappers: DDP (.module) and torch.compile (._orig_mod)
    m = getattr(model, "module", model)
    m = getattr(m, "_orig_mod", m)
    if not hasattr(m, "save_pretrained"):
        raise TypeError(
            f"Model of type {type(m).__name__} does not support save_pretrained; "
            f"expected a transformers PreTrainedModel (e.g. Qwen3ForCausalLM)."
        )
    if dtype is not None:
        m = m.to(dtype)
    # For custom models (e.g. Qwen3.5), register for auto class so save_pretrained copies the
    # modeling code + writes auto_map (needed for trust_remote_code loading of the folder).
    _register_custom_model_for_auto_class(m)
    # Populate bos/eos/pad token ids from the tokenizer so the exported config.json /
    # generation_config.json carry them (like the original HF release) and generate() can stop.
    _sync_special_token_ids_into_model(m, tokenizer)
    m.save_pretrained(save_dir, safe_serialization=safe_serialization)
    logger.info(f"Saved transformers-compatible model to: {save_dir}")
    _save_tokenizer_into(save_dir, tokenizer)
    # Save optimizer/loop state so an HF checkpoint can also be resumed.
    if optimizer_data is not None:
        optim_path = os.path.join(save_dir, "optim.pt")
        _atomic_torch_save(_to_cpu(optimizer_data), optim_path)
        logger.info(f"Saved optimizer state to: {optim_path}")
    if meta_data is not None:
        meta_path = os.path.join(save_dir, "meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)
        logger.info(f"Saved metadata to: {meta_path}")


def export_gpt_to_transformers(model, meta_or_config, save_dir, tokenizer=None, dtype=None, safe_serialization=True):
    """Export a nanochat `GPT` model to a transformers-loadable directory.

    Produces a self-contained checkpoint folder that can be loaded on any machine with only
    `transformers` installed (no nanochat / flash-attn dependency)::

        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(save_dir, trust_remote_code=True)
        tok   = AutoTokenizer.from_pretrained(save_dir)

    Writes `config.json` + `model.safetensors` (via `save_pretrained`), copies the standalone
    `modeling_nanochat.py` into the folder (with an `auto_map` in the config so `trust_remote_code`
    works), and exports the tokenizer via its `save_pretrained` so `AutoTokenizer` works too.

    Args:
        model: the nanochat `GPT` model (may be wrapped in DDP / torch.compile; will be unwrapped).
        meta_or_config: either the `GPTConfig` used to build the model, or a `meta_data` dict that
            contains a "model_config" entry (as saved by the training scripts).
        save_dir: output directory.
        tokenizer: optional tokenizer exposing `save_pretrained(save_dir)` (e.g. nanochat RustBPETokenizer).
        dtype: optional torch dtype to cast weights to before saving (e.g. torch.bfloat16).
    """
    from nanochat.modeling_nanochat import NanochatConfig, NanochatForCausalLM

    os.makedirs(save_dir, exist_ok=True)

    # Unwrap common wrappers: DDP (.module) and torch.compile (._orig_mod)
    m = getattr(model, "module", model)
    m = getattr(m, "_orig_mod", m)

    # Resolve the GPTConfig (dataclass) whether we were handed a config or a meta dict
    if isinstance(meta_or_config, dict) and "model_config" in meta_or_config:
        gpt_cfg_kwargs = dict(meta_or_config["model_config"])
    elif hasattr(meta_or_config, "__dict__") and hasattr(meta_or_config, "n_layer"):
        gpt_cfg_kwargs = dict(meta_or_config.__dict__)
    else:
        gpt_cfg_kwargs = dict(meta_or_config)
    _patch_missing_config_keys(gpt_cfg_kwargs)

    # Grab the (possibly compiled) state dict and strip the torch.compile prefix
    state_dict = m.state_dict()
    state_dict = {k.removeprefix("_orig_mod."): v.detach().cpu() for k, v in state_dict.items()}
    if dtype is not None:
        state_dict = {k: (v.to(dtype) if v.is_floating_point() else v) for k, v in state_dict.items()}

    # The embedding rows define the (padded) vocab size actually stored in the weights
    padded_vocab_size = state_dict["transformer.wte.weight"].shape[0]

    hf_config = NanochatConfig(
        vocab_size=padded_vocab_size,
        text_vocab_size=gpt_cfg_kwargs["vocab_size"],
        n_layer=gpt_cfg_kwargs["n_layer"],
        n_head=gpt_cfg_kwargs["n_head"],
        n_kv_head=gpt_cfg_kwargs["n_kv_head"],
        n_embd=gpt_cfg_kwargs["n_embd"],
        sequence_len=gpt_cfg_kwargs["sequence_len"],
        window_pattern=gpt_cfg_kwargs.get("window_pattern", "SSSL"),
        architectures=["NanochatForCausalLM"],
    )

    # Build the HF model on meta then load weights via assign to avoid double memory
    with torch.device("meta"):
        hf_model = NanochatForCausalLM(hf_config)
    hf_model.load_state_dict(state_dict, strict=True, assign=True)
    if dtype is not None:
        hf_model.config.torch_dtype = str(dtype).replace("torch.", "")

    # Register for auto classes so save_pretrained copies modeling_nanochat.py + sets auto_map,
    # making the folder loadable with AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)
    NanochatConfig.register_for_auto_class()
    NanochatForCausalLM.register_for_auto_class("AutoModelForCausalLM")

    hf_model.save_pretrained(save_dir, safe_serialization=safe_serialization)
    logger.info(f"Saved transformers-compatible nanochat model to: {save_dir}")

    if tokenizer is not None:
        if hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(save_dir)
            logger.info(f"Saved transformers-compatible tokenizer to: {save_dir}")
        else:
            logger.warning(f"Tokenizer of type {type(tokenizer).__name__} has no save_pretrained; skipping.")

    return save_dir


def _load_hf_state_dict(hf_dir, device):
    """Load a model state_dict from an HF-format directory (safetensors, possibly sharded)."""
    from safetensors.torch import load_file
    index_path = os.path.join(hf_dir, "model.safetensors.index.json")
    state = {}
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
        for shard in shard_files:
            state.update(load_file(os.path.join(hf_dir, shard), device=str(device)))
    else:
        single = os.path.join(hf_dir, "model.safetensors")
        if os.path.exists(single):
            state = load_file(single, device=str(device))
        else:
            bin_path = os.path.join(hf_dir, "pytorch_model.bin")
            if not os.path.exists(bin_path):
                raise FileNotFoundError(f"No model weights (safetensors/bin) found in {hf_dir}")
            state = torch.load(bin_path, map_location=device)
    return state


def load_hf_checkpoint(hf_dir, device, load_optimizer=False):
    """Load model_data / optimizer_data / meta_data from an HF-format checkpoint directory."""
    model_data = _load_hf_state_dict(hf_dir, device)
    optimizer_data = None
    if load_optimizer:
        optim_path = os.path.join(hf_dir, "optim.pt")
        if os.path.exists(optim_path):
            optimizer_data = torch.load(optim_path, map_location=device)
        else:
            log0(f"No optimizer state found in HF checkpoint {hf_dir} (optim.pt missing)")
    meta_path = os.path.join(hf_dir, "meta.json")
    meta_data = None
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_data = json.load(f)
    return model_data, optimizer_data, meta_data


def load_checkpoint_any(checkpoint_dir, step, device, load_optimizer=False, rank=0):
    """Load a checkpoint at `step`, auto-detecting nanochat (.pt) vs HF (hf_<step>/) format."""
    pt_model = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    if os.path.exists(pt_model):
        return load_checkpoint(checkpoint_dir, step, device, load_optimizer=load_optimizer, rank=rank)
    hf_dir = os.path.join(checkpoint_dir, f"hf_{step:06d}")
    if os.path.isdir(hf_dir):
        return load_hf_checkpoint(hf_dir, device, load_optimizer=load_optimizer)
    raise FileNotFoundError(
        f"No checkpoint found for step {step} in {checkpoint_dir} (looked for model_{step:06d}.pt and hf_{step:06d}/)"
    )

def load_checkpoint(checkpoint_dir, step, device, load_optimizer=False, rank=0):
    # Load the model state
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    model_data = torch.load(model_path, map_location=device)
    # Load the optimizer state if requested
    optimizer_data = None
    if load_optimizer:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        if not os.path.exists(optimizer_path):
            # The optimizer state is ZeRO-2 *sharded* (see save_checkpoint / optim.py), so a
            # shard is specific to its rank and NOT interchangeable. A missing shard usually
            # means we are resuming with a different world_size than was used when saving, in
            # which case the shard boundaries no longer line up and the state cannot be
            # reconstructed correctly. We fall back to an available shard only as a
            # best-effort last resort to allow training to continue, but WARN loudly because
            # the loaded optimizer state will be wrong for most parameters.
            available = sorted(
                int(re.search(r"_rank(\d+)\.pt$", f).group(1))
                for f in glob.glob(os.path.join(checkpoint_dir, f"optim_{step:06d}_rank*.pt"))
            )
            if not available:
                raise FileNotFoundError(
                    f"No optimizer checkpoints found for step {step} in {checkpoint_dir}"
                )
            fallback_rank = available[rank % len(available)]
            fallback_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{fallback_rank:d}.pt")
            log0(
                f"WARNING: optimizer shard for rank{rank} not found ({optimizer_path}); "
                f"the optimizer state is sharded, so this likely means the world_size changed "
                f"since saving ({len(available)} shards available: {available}). "
                f"Falling back to rank{fallback_rank} ({fallback_path}) as a best-effort resume; "
                f"the optimizer state will be INCORRECT for most parameters."
            )
            optimizer_path = fallback_path
        optimizer_data = torch.load(optimizer_path, map_location=device)
    # Load the metadata
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    return model_data, optimizer_data, meta_data


def build_model(checkpoint_dir, step, device, phase):
    """
    A bunch of repetitive code to build a model from a given checkpoint.
    Returns:
    - base model - uncompiled, not wrapped in DDP
    - tokenizer
    - meta data saved during base model training
    """
    assert phase in ["train", "eval"], f"Invalid phase: {phase}"
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)
    if device.type in {"cpu", "mps"}:
        # Convert bfloat16 tensors to float for CPU inference
        model_data = {
            k: v.float() if v.dtype == torch.bfloat16 else v
            for k, v in model_data.items()
        }
    # Hack: fix torch compile issue, which prepends all keys with _orig_mod.
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    model_config_kwargs = meta_data["model_config"]
    _patch_missing_config_keys(model_config_kwargs)
    log0(f"Building model with config: {model_config_kwargs}")
    model_config = GPTConfig(**model_config_kwargs)
    _patch_missing_keys(model_data, model_config)
    with torch.device("meta"):
        model = GPT(model_config)
    # Load the model state
    model.to_empty(device=device)
    model.init_weights() # note: this is dumb, but we need to init the rotary embeddings. TODO: fix model re-init
    model.load_state_dict(model_data, strict=True, assign=True)
    # Put the model in the right training phase / mode
    if phase == "eval":
        model.eval()
    else:
        model.train()
    # Load the Tokenizer
    tokenizer = get_tokenizer()
    # Sanity check: compatibility between model and tokenizer
    assert tokenizer.get_vocab_size() == model_config_kwargs["vocab_size"], f"Tokenizer vocab size {tokenizer.get_vocab_size()} does not match model config vocab size {model_config_kwargs['vocab_size']}"
    return model, tokenizer, meta_data


def find_largest_model(checkpoints_dir):
    # attempt to guess the model tag: take the biggest model available
    model_tags = [f for f in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, f))]
    if not model_tags:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")
    # 1) normally all model tags are of the form d<number>, try that first:
    candidates = []
    for model_tag in model_tags:
        match = re.match(r"d(\d+)", model_tag)
        if match:
            model_depth = int(match.group(1))
            candidates.append((model_depth, model_tag))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    # 2) if that failed, take the most recently updated model:
    model_tags.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoints_dir, x)), reverse=True)
    return model_tags[0]


def find_last_step(checkpoint_dir):
    # Look into checkpoint_dir and find model_<step>.pt with the highest step
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    last_step = int(max(os.path.basename(f).split("_")[-1].split(".")[0] for f in checkpoint_files))
    return last_step

# -----------------------------------------------------------------------------
# convenience functions that take into account nanochat's directory structure

def load_model_from_dir(checkpoints_dir, device, phase, model_tag=None, step=None):
    if model_tag is None:
        # guess the model tag by defaulting to the largest model
        model_tag = find_largest_model(checkpoints_dir)
        log0(f"No model tag provided, guessing model tag: {model_tag}")
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    if step is None:
        # guess the step by defaulting to the last step
        step = find_last_step(checkpoint_dir)
    assert step is not None, f"No checkpoints found in {checkpoint_dir}"
    # build the model
    log0(f"Loading model from {checkpoint_dir} with step {step}")
    model, tokenizer, meta_data = build_model(checkpoint_dir, step, device, phase)
    return model, tokenizer, meta_data

def load_model(source, *args, **kwargs):
    model_dir = {
        "base": "base_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }[source]
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, model_dir)
    return load_model_from_dir(checkpoints_dir, *args, **kwargs)

def load_optimizer_state(source, device, rank, model_tag=None, step=None):
    """Load just the optimizer shard for a given rank, without re-loading the model."""
    model_dir = {
        "base": "base_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }[source]
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, model_dir)
    if model_tag is None:
        model_tag = find_largest_model(checkpoints_dir)
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    if step is None:
        step = find_last_step(checkpoint_dir)
    optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
    if not os.path.exists(optimizer_path):
        # The optimizer state is ZeRO-2 *sharded* (see save_checkpoint / optim.py): each
        # shard belongs to a specific rank and is NOT interchangeable. A missing shard
        # generally means the world_size changed since saving, so the shard boundaries no
        # longer match and the state cannot be reconstructed correctly. Fall back to an
        # available shard only as a best-effort last resort, and WARN loudly.
        available = sorted(
            int(re.search(r"_rank(\d+)\.pt$", f).group(1))
            for f in glob.glob(os.path.join(checkpoint_dir, f"optim_{step:06d}_rank*.pt"))
        )
        if not available:
            log0(f"Optimizer checkpoint not found: {optimizer_path}")
            return None
        fallback_rank = available[rank % len(available)]
        fallback_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{fallback_rank:d}.pt")
        log0(
            f"WARNING: optimizer shard for rank{rank} not found; the optimizer state is "
            f"sharded, so this likely means the world_size changed since saving "
            f"({len(available)} shards available: {available}). Falling back to rank{fallback_rank} "
            f"as a best-effort resume; the optimizer state will be INCORRECT for most parameters."
        )
        optimizer_path = fallback_path
    log0(f"Loading optimizer state from {optimizer_path}")
    optimizer_data = torch.load(optimizer_path, map_location=device)
    return optimizer_data
