# Qwen3.5 *text-only* LLM adapted for nanochat-style training.
#
# This file started life as the auto-generated `modeling_qwen3_5.py` from transformers
# (kept as qwen3p5_generated_full.py.bak), but has been:
#   * reduced to the TEXT model only (vision / multimodal removed) -- we only train the LLM,
#   * switched to absolute `transformers.*` imports so it can live inside nanochat,
#   * given a `qwen3.py`-style training interface on `Qwen3_5ForCausalLM`:
#       - forward(idx, targets) -> loss (training) / logits (inference)
#       - forward1(...) -> raw logits (HF-style)
#       - re_init_weights(), logit-softcap annealing, optional MTP heads,
#         get_all_hidden_states() for DFLASH.
#
# Compatibility shims at the top let the file import on both the modern dev transformers
# (where the Qwen3.5 kernels/utilities exist) and older releases (fallbacks are used).

from collections.abc import Callable
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from .configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig

# -----------------------------------------------------------------------------
# Compatibility shims: the modern Qwen3.5 modeling file relies on a number of very
# new transformers utilities (kernel-from-hub decorators, capture_outputs, an
# `initialization` helper, recurrent attention masks, ...). They may not exist on
# older transformers, so we import them if available and otherwise fall back to
# no-op / pure-torch implementations. Behaviour is identical for dense training.

def _identity_decorator(*dargs, **dkwargs):
    """A decorator (or decorator factory) that returns the target unchanged."""
    if len(dargs) == 1 and not dkwargs and callable(dargs[0]):
        return dargs[0]

    def wrap(obj):
        return obj

    return wrap


def _call_mask_builder(fn, mask_kwargs):
    """Call a transformers mask builder, tolerating the `inputs_embeds` vs
    `input_embeds` kwarg rename across transformers versions."""
    try:
        return fn(**mask_kwargs)
    except TypeError as e:
        if "inputs_embeds" in mask_kwargs and "input_embeds" in str(e):
            alt = dict(mask_kwargs)
            alt["input_embeds"] = alt.pop("inputs_embeds")
            return fn(**alt)
        raise


try:  # kernel-from-hub decorators (optional acceleration only)
    from transformers.integrations import use_kernel_forward_from_hub
except Exception:  # pragma: no cover - older transformers
    use_kernel_forward_from_hub = _identity_decorator

try:
    from transformers.integrations import use_kernelized_func
except Exception:  # pragma: no cover
    use_kernelized_func = _identity_decorator

try:
    from transformers.integrations import use_kernel_func_from_hub_with_fallback
except Exception:  # pragma: no cover
    use_kernel_func_from_hub_with_fallback = _identity_decorator

try:
    from transformers.integrations.accelerate import force_accelerate_hooks
except Exception:  # pragma: no cover
    force_accelerate_hooks = _identity_decorator

try:
    from transformers.utils.output_capturing import capture_outputs
except Exception:  # pragma: no cover
    capture_outputs = _identity_decorator

try:
    from transformers.utils.generic import merge_with_config_defaults
except Exception:  # pragma: no cover
    merge_with_config_defaults = _identity_decorator

try:
    from transformers.utils.generic import maybe_autocast
except Exception:  # pragma: no cover
    @contextmanager
    def maybe_autocast(device_type="cpu", enabled=False, **kwargs):
        if enabled:
            with torch.autocast(device_type=device_type, **kwargs):
                yield
        else:
            yield

try:
    from transformers.masking_utils import create_recurrent_attention_mask
except Exception:  # pragma: no cover
    def create_recurrent_attention_mask(**kwargs):
        # For dense (non-padded) training the GatedDeltaNet only uses a 2D padding
        # mask; returning None means "no padding" which is correct here.
        return None

try:
    from transformers import initialization as init
except Exception:  # pragma: no cover
    class _InitShim:
        @staticmethod
        @torch.no_grad()
        def ones_(t):
            return t.fill_(1.0)

        @staticmethod
        @torch.no_grad()
        def zeros_(t):
            return t.zero_()

        @staticmethod
        @torch.no_grad()
        def copy_(t, src):
            return t.copy_(src)

    init = _InitShim()


# -----------------------------------------------------------------------------
# Rotary embedding (text). Supports the interleaved M-RoPE layout used by Qwen3.5.
# For pure-text training the 3 grid position ids are identical so it reduces to
# ordinary RoPE.

class Qwen3_5TextRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config: Qwen3_5TextConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config

        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)
        self.mrope_section = config.rope_parameters.get("mrope_section", [11, 11, 10])

    @staticmethod
    def compute_default_rope_parameters(config: Qwen3_5TextConfig, device=None, **kwargs):
        base = config.rope_parameters["rope_theta"]
        partial_rotary_factor = config.rope_parameters.get("partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial_rotary_factor)
        attention_factor = 1.0
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        return inv_freq.to(device), attention_factor

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        # Qwen3.5 has different position ids per grid, so expand inv_freq to (3, ...)
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        inv_freq_expanded = (
            self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1).to(x.device)
        )
        position_ids_expanded = position_ids[:, :, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def apply_interleaved_mrope(self, freqs, mrope_section):
        """Reorganize chunked [TTT..HHH..WWW] into interleaved [THWTHW..]."""
        freqs_t = freqs[0]
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t


@use_kernel_forward_from_hub("RMSNormGated")
class Qwen3_5RMSNormGated(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.activation = "silu"

    def forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * ACT2FN[self.activation](gate.to(torch.float32))
        return hidden_states.to(input_dtype)


def apply_mask_to_padding_states(hidden_states, attention_mask):
    if attention_mask is not None:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
    return hidden_states


@use_kernel_func_from_hub_with_fallback("causal_conv1d_update", "causal_conv1d")
def causal_conv1d_update(hidden_states, conv_state, weight, bias=None, activation=None):
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]
    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    out = out[:, :, -seq_len:]
    if activation is not None:
        out = ACT2FN[activation](out)
    return out.to(hidden_states.dtype)


@use_kernel_func_from_hub_with_fallback("causal_conv1d_fn", "causal_conv1d")
def causal_conv1d_fn(hidden_states, weight, bias=None, activation=None, **kwargs):
    _, hidden_size, seq_len = hidden_states.shape
    padding = weight.shape[-1] - 1
    out = F.conv1d(
        hidden_states.to(weight.dtype),
        weight=weight.unsqueeze(1),
        bias=bias,
        padding=padding,
        groups=hidden_size,
    )[:, :, :seq_len]
    if activation is not None:
        out = ACT2FN[activation](out)
    return out.to(hidden_states.dtype)


def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


@use_kernel_func_from_hub_with_fallback("chunk_gated_delta_rule", "fla")
def torch_chunk_gated_delta_rule(
    query, key, value, g, beta, chunk_size=64, initial_state=None,
    output_final_state=False, use_qk_l2norm_in_kernel=False, **kwargs,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


@use_kernel_func_from_hub_with_fallback("recurrent_gated_delta_rule", "fla")
def torch_recurrent_gated_delta_rule(
    query, key, value, g, beta, initial_state, output_final_state,
    use_qk_l2norm_in_kernel=False, **kwargs,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(
        batch_size, num_heads, sequence_length, v_head_dim, dtype=value.dtype, device=value.device
    )
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


@use_kernel_forward_from_hub("Qwen3_5GatedDeltaNet")
@use_kernelized_func(
    [torch_recurrent_gated_delta_rule, torch_chunk_gated_delta_rule, causal_conv1d_fn, causal_conv1d_update]
)
class Qwen3_5GatedDeltaNet(nn.Module):
    def __init__(self, config: Qwen3_5Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.activation = config.hidden_act
        self.layer_norm_epsilon = config.rms_norm_eps

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        A = torch.empty(self.num_v_heads).uniform_(0.01, 16)
        self.A_log = nn.Parameter(torch.log(A))

        self.norm = Qwen3_5RMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        self.layer_type = config.layer_types[layer_idx]

        self.in_proj_qkv = nn.Linear(self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

    @force_accelerate_hooks("conv1d")
    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len, _ = hidden_states.shape
        use_precomputed_states = cache_params is not None and cache_params.has_previous_state(self.layer_idx)

        mixed_qkv = self.in_proj_qkv(hidden_states)
        mixed_qkv = mixed_qkv.transpose(1, 2)

        z = self.in_proj_z(hidden_states)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        if use_precomputed_states and seq_len == 1 and not cache_params.layers[self.layer_idx].record_past:
            conv_state = cache_params.layers[self.layer_idx].conv_states[0]
            mixed_qkv = causal_conv1d_update(
                mixed_qkv, conv_state, self.conv1d.weight.squeeze(1), self.conv1d.bias, self.activation,
            )
        else:
            if cache_params is not None:
                mixed_qkv = cache_params.update_conv_state(
                    mixed_qkv, self.layer_idx, conv_kernel_size=self.conv_kernel_size
                )
            mixed_qkv = causal_conv1d_fn(
                mixed_qkv, self.conv1d.weight.squeeze(1), self.conv1d.bias, activation=self.activation, **kwargs,
            )
            if cache_params is not None:
                mixed_qkv = mixed_qkv[:, :, -seq_len:]

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1,
        )

        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        recurrent_state = cache_params.layers[self.layer_idx].recurrent_states[0] if use_precomputed_states else None
        if use_precomputed_states and seq_len == 1:
            core_attn_out, last_recurrent_state = torch_recurrent_gated_delta_rule(
                query, key, value, g=g, beta=beta, initial_state=recurrent_state,
                output_final_state=cache_params is not None, use_qk_l2norm_in_kernel=True,
                cu_seqlens=kwargs.pop("cu_seq_lens_q", None), **kwargs,
            )
        else:
            core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
                query, key, value, g=g, beta=beta, initial_state=recurrent_state,
                output_final_state=cache_params is not None, use_qk_l2norm_in_kernel=True,
                cu_seqlens=kwargs.pop("cu_seq_lens_q", None), **kwargs,
            )

        if cache_params is not None:
            cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        output = self.out_proj(core_attn_out)
        return output


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class Qwen3_5Attention(nn.Module):
    """Gated multi-headed softmax attention (full_attention layers)."""

    def __init__(self, config: Qwen3_5Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim * 2, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self, query_states, key_states, value_states, attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling, **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3_5MLP(nn.Module):
    def __init__(self, config: Qwen3_5Config, intermediate_size: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen3_5RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class Qwen3_5DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.block_type = config.layer_types[layer_idx]
        if self.block_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx)
        elif self.block_type == "full_attention":
            self.self_attn = Qwen3_5Attention(config, layer_idx)
        self.mlp = Qwen3_5MLP(config, config.intermediate_size)
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.FloatTensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.block_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states=hidden_states, cache_params=past_key_values,
                attention_mask=attention_mask, **kwargs,
            )
        elif self.block_type == "full_attention":
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states, attention_mask=attention_mask,
                position_ids=position_ids, past_key_values=past_key_values,
                position_embeddings=position_embeddings, **kwargs,
            )

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen3_5PreTrainedModel(PreTrainedModel):
    config: Qwen3_5TextConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3_5DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _keys_to_ignore_on_load_unexpected = [r"^mtp.*"]
    _can_record_outputs = {
        "hidden_states": Qwen3_5DecoderLayer,
        "attentions": Qwen3_5Attention,
    }
    _is_stateful = True
    _can_compile_fullgraph = True

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Qwen3_5GatedDeltaNet):
            init.ones_(module.dt_bias)
            init.copy_(
                module.A_log,
                torch.empty(module.num_v_heads, device=module.A_log.device).uniform_(0.01, 16).log_(),
            )
        elif isinstance(module, Qwen3_5RMSNorm):
            # 1-centered RMSNorm here uses (1 + weight), so init weight to 0.
            init.zeros_(module.weight)


class Qwen3_5TextModel(Qwen3_5PreTrainedModel):
    config: Qwen3_5TextConfig

    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [Qwen3_5DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    def _build_position_ids(self, inputs_embeds, position_ids, past_key_values):
        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = None
        return text_position_ids, position_ids

    def _build_masks(self, attention_mask, inputs_embeds, past_key_values, text_position_ids):
        if isinstance(attention_mask, dict):
            return attention_mask
        mask_kwargs = {
            "config": self.config,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "position_ids": text_position_ids,
        }
        return {
            "full_attention": _call_mask_builder(create_causal_mask, mask_kwargs),
            "linear_attention": _call_mask_builder(create_recurrent_attention_mask, mask_kwargs),
        }

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        text_position_ids, position_ids = self._build_position_ids(inputs_embeds, position_ids, past_key_values)
        causal_mask_mapping = self._build_masks(attention_mask, inputs_embeds, past_key_values, text_position_ids)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class Qwen3_5ForCausalLM(Qwen3_5PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    config: Qwen3_5TextConfig
    _keys_to_ignore_on_load_unexpected = [r"^mtp.*", r"^model.visual.*"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3_5TextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Logit softcap that AUTOMATICALLY anneals away during training and is NEVER
        # used at inference (see nanochat/qwen3.py for the full rationale).
        self.softcap_start = float(getattr(config, "logit_softcap", 15.0) or 0.0)
        self.softcap_end = float(getattr(config, "logit_softcap_end", 100.0) or 0.0)
        self.softcap_anneal_steps = int(getattr(config, "logit_softcap_anneal_steps", 2000) or 0)
        # NOTE: non-persistent so it is NOT saved into the model checkpoint (keeps exported
        # HF folders clean / loadable by the native transformers Qwen3_5ForCausalLM without
        # an "unexpected key" warning). On resume it is re-synced from the training step
        # (see set_softcap_step / the training loop), so annealing stays correct.
        self.register_buffer("_softcap_step", torch.zeros((), dtype=torch.long), persistent=False)
        # Optional MTP heads for predicting farther-future tokens from the same hidden state.
        self.mtp_num_heads = int(getattr(config, "mtp_num_heads", 0) or 0)
        self.mtp_weight = float(getattr(config, "mtp_weight", 0.0) or 0.0)
        self.mtp_heads = nn.ModuleList(
            [nn.Linear(config.hidden_size, config.vocab_size, bias=False) for _ in range(self.mtp_num_heads)]
        )
        self.post_init()

    def get_device(self):
        return self.device

    # ------------------------------------------------------------------
    # DFLASH helpers (frozen / with-grad hidden states of every layer)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_all_hidden_states(self, idx):
        return self._all_hidden_states(idx)

    def all_hidden_states_with_grad(self, idx):
        return self._all_hidden_states(idx)

    def _all_hidden_states(self, idx):
        m = self.model
        inputs_embeds = m.embed_tokens(idx)
        text_position_ids, position_ids = m._build_position_ids(inputs_embeds, None, None)
        causal_mask_mapping = m._build_masks(None, inputs_embeds, None, text_position_ids)
        hidden_states = inputs_embeds
        position_embeddings = m.rotary_emb(hidden_states, position_ids)
        all_hidden = [hidden_states]
        for i, decoder_layer in enumerate(m.layers[: m.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask_mapping[m.config.layer_types[i]],
                position_ids=text_position_ids,
                past_key_values=None,
                use_cache=False,
            )
            all_hidden.append(hidden_states)
        return all_hidden

    # ------------------------------------------------------------------
    # training-mode forward (mirrors nanochat/qwen3.py)
    # ------------------------------------------------------------------
    def _compute_mtp_loss(self, hidden_states, targets, loss_reduction="mean"):
        if self.mtp_num_heads <= 0 or self.mtp_weight <= 0 or targets is None:
            return None
        mtp_losses = []
        for i, head in enumerate(self.mtp_heads):
            horizon = i + 2  # head i predicts t+2, t+3, ...
            valid_t = hidden_states.size(1) - (horizon - 1)
            if valid_t <= 0:
                continue
            logits_h = head(hidden_states[:, :valid_t, :])
            targets_h = targets[:, horizon - 1 : horizon - 1 + valid_t]
            loss_h = F.cross_entropy(
                logits_h.reshape(-1, logits_h.size(-1)),
                targets_h.reshape(-1),
                ignore_index=-1,
                reduction=loss_reduction,
            )
            mtp_losses.append(loss_h)
        if not mtp_losses:
            return None
        return sum(mtp_losses) / len(mtp_losses)

    @torch.no_grad()
    def set_softcap_step(self, step):
        """Re-sync the (non-persistent) softcap annealing counter, e.g. after resuming
        training from a checkpoint that no longer stores `_softcap_step`."""
        self._softcap_step.fill_(int(step))

    def _current_softcap(self):
        if self.softcap_start <= 0:
            return None
        step = int(self._softcap_step.item())
        # step == -1 => freeze the softcap at softcap_start (e.g. 15) forever,
        # never annealing away and always applied during training.
        if step < 0:
            return self.softcap_start
        if self.softcap_anneal_steps <= 0:
            return None
        if step >= self.softcap_anneal_steps:
            return None
        p = step / self.softcap_anneal_steps
        return self.softcap_start * (self.softcap_end / self.softcap_start) ** p

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean"):
        logits, hidden_states = self.forward1(idx, labels=targets, return_hidden_states=True)
        logits = logits.float()  # fp32 for softcap + loss
        if targets is not None:
            softcap = self._current_softcap()
            if softcap is not None:
                logits = softcap * torch.tanh(logits / softcap)
            # When frozen at -1 (constant softcap) don't advance the annealing counter.
            if int(self._softcap_step.item()) >= 0:
                self._softcap_step += 1

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1),
                ignore_index=-1, reduction=loss_reduction,
            )
            mtp_loss = self._compute_mtp_loss(hidden_states, targets, loss_reduction=loss_reduction)
            if mtp_loss is not None:
                loss = loss + self.mtp_weight * mtp_loss
            return loss
        return logits

    @torch.no_grad()
    def re_init_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, torch.nn.Embedding):
                torch.nn.init.normal_(m.weight, mean=0, std=0.02)
            elif "lm_head" in n:
                torch.nn.init.normal_(m.weight, mean=0, std=0.001)
            elif "down_proj" in n or "o_proj" in n or "out_proj" in n:
                torch.nn.init.zeros_(m.weight)
            elif isinstance(m, torch.nn.Linear):
                fan_in = m.in_features
                s = 3**0.5 * fan_in**-0.5
                torch.nn.init.uniform_(m.weight, -s, s)

    def forward1(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        return_hidden_states: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ):
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        if return_hidden_states:
            return logits, hidden_states
        return logits


__all__ = [
    "Qwen3_5ForCausalLM",
    "Qwen3_5TextModel",
    "Qwen3_5PreTrainedModel",
    "Qwen3_5TextConfig",
    "Qwen3_5Config",
]



