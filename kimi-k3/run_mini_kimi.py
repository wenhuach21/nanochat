"""
run_mini_kimi.py  ──  放在 kimi-k3/ 目录下直接运行
python run_mini_kimi.py
"""
import sys, os
# 确保当前目录在 path 里（解决相对 import 问题）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from configuration_kimi_k3 import KimiLinearConfig
from modeling_kimi_linear import KimiLinearForCausalLM

# ─────────────────────────────────────────────────────────────────────────────
# Tiny Config（字段名完全对齐 config.json，只把维度改小）
# ─────────────────────────────────────────────────────────────────────────────
tiny_text_cfg = dict(
    vocab_size              = 1024,
    hidden_size             = 64,
    num_hidden_layers       = 4,
    rms_norm_eps            = 1e-5,

    num_attention_heads     = 4,
    num_key_value_heads     = 4,
    q_lora_rank             = 32,
    kv_lora_rank            = 16,
    qk_nope_head_dim        = 16,
    qk_rope_head_dim        = 8,
    v_head_dim              = 16,
    mla_use_nope            = True,
    mla_use_output_gate     = True,

    hidden_act                   = "situ",
    intermediate_size            = 128,
    activation_situ_beta         = 4.0,
    activation_situ_linear_beta  = 25.0,

    num_experts             = 8,
    num_experts_per_token   = 2,
    moe_intermediate_size   = 32,
    num_shared_experts      = 1,
    routed_scaling_factor   = 1.0,
    moe_renormalize         = True,
    moe_router_activation_func = "sigmoid",
    num_expert_group        = 1,
    topk_group              = 1,
    first_k_dense_replace   = 1,
    moe_layer_freq          = 1,
    topk_method             = "noaux_tc",

    routed_expert_hidden_size = 32,
    latent_moe_use_norm       = True,

    attn_res_block_size     = 2,

    linear_attn_config = dict(
        kda_layers             = [1, 2, 3],   # 1-indexed: layer_idx 0/1/2 是 KDA
        full_attn_layers       = [4],          # layer_idx 3 是 MLA
        short_conv_kernel_size = 4,
        head_dim               = 16,
        num_heads              = 4,
        use_full_rank_gate     = True,
        gate_lower_bound       = -5.0,
    ),

    max_position_embeddings = 4096,
    pad_token_id  = 0,
    bos_token_id  = 1,
    eos_token_id  = 2,
    use_cache     = False,
)

cfg = KimiLinearConfig(**tiny_text_cfg)
cfg._attn_implementation = "eager"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

# ─────────────────────────────────────────────────────────────────────────────
# 实例化
# ─────────────────────────────────────────────────────────────────────────────
model = KimiLinearForCausalLM(cfg).to(device).eval()

n = sum(p.numel() for p in model.parameters())
print(f"参数量: {n:,}")
print("\n层结构:")
for i, layer in enumerate(model.model.layers):
    attn = "KDA" if layer.is_linear_attn else "MLA"
    ffn  = "MoE" if hasattr(layer, "block_sparse_moe") else "dense-MLP"
    print(f"  Layer {i}: {attn} + {ffn}")

# ─────────────────────────────────────────────────────────────────────────────
# 前向
# ─────────────────────────────────────────────────────────────────────────────
B, T = 2, 8
input_ids      = torch.randint(1, cfg.vocab_size, (B, T)).to(device)
attention_mask = torch.ones(B, T, dtype=torch.long).to(device)
attention_mask[0, -2:] = 0   # 模拟 padding

print(f"\ninput_ids:      {input_ids.shape}")
print(f"attention_mask: {attention_mask.shape}")

with torch.no_grad():
    out = model(input_ids=input_ids, attention_mask=attention_mask)

print(f"\nlogits: {out.logits.shape}")
print(f"logits[0,0,:5]: {out.logits[0,0,:5].tolist()}")
print("\n✅ done")
