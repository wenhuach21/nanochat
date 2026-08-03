"""
debug_kimi_k3.py
================
Kimi-K3 架构学习与调试脚本
本脚本不依赖 fla-core / flash-attn，用纯 PyTorch 实现所有关键模块，
模型尺寸极小，适合在 CPU 上运行，附有大量注释和形状追踪。

架构总览 (text-only 部分 KimiLinearModel):
  Embedding
    └─> [KimiDecoderLayer] × num_layers
            ├─ 全量注意力层: KimiMLAAttention (Multi-Latent Attention, like DeepSeek-V3)
            └─ 线性注意力层: KimiDeltaAttention  (KDA, 线性复杂度)
            └─ FFN: KimiMLP (dense) 或 KimiSparseMoeBlock (MoE)
            └─ AttnResidual (可选): 跨层残差聚合
    └─> RMSNorm
    └─> lm_head

视觉部分 (MoonViT3d) 独立演示在最后一节。
"""

import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────────────
# 辅助工具
# ──────────────────────────────────────────────────────────────────────────────

def shape_str(t):
    return str(tuple(t.shape)) if isinstance(t, torch.Tensor) else str(t)

class ShapeTracer:
    """在 print 里顺便记录 tensor 形状，方便 debug."""
    def __init__(self, name, tensor):
        self.name = name
        self.tensor = tensor
        print(f"  📐 {name:45s} {shape_str(tensor)}")

    def __enter__(self):
        return self.tensor

    def __exit__(self, *_):
        pass

def trace(name, tensor):
    print(f"  📐 {name:45s} {shape_str(tensor)}")
    return tensor


# ══════════════════════════════════════════════════════════════════════════════
# 第一部分：基础组件
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*70)
print("第一部分：基础组件")
print("═"*70)


# ─────────────────────────────────────────────
# 1.1  KimiRMSNorm  (Root Mean Square LayerNorm)
# ─────────────────────────────────────────────
class KimiRMSNorm(nn.Module):
    """
    RMSNorm: 只用方差，不减均值。比 LayerNorm 更快更稳。
    x_norm = x / sqrt(mean(x^2) + eps)  ×  weight
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        var = x.float().pow(2).mean(-1, keepdim=True)           # (..., 1)
        x_norm = x.float() * torch.rsqrt(var + self.eps)        # (..., dim)
        return (self.weight * x_norm).to(x.dtype)


# 快速测试
rms = KimiRMSNorm(8)
x_test = torch.randn(2, 4, 8)
print(f"\n[RMSNorm] 输入 {shape_str(x_test)} → 输出 {shape_str(rms(x_test))}")


# ─────────────────────────────────────────────
# 1.2  SituAndMul  (Kimi 专属激活函数)
# ─────────────────────────────────────────────
class SituAndMul(nn.Module):
    """
    Kimi 自研激活函数：
      gate_activated = beta * tanh(gate / beta) * sigmoid(gate)
      output = gate_activated * up

    当 beta→∞ 时退化为 SiLU；当 beta=1 时是 tanh(x)*sigmoid(x)。
    这比 SiLU 在大激活值处有更强的梯度衰减，训练更稳定。
    """
    def __init__(self, beta: float = 1.0, linear_beta: float | None = None):
        super().__init__()
        self.beta = beta
        self.linear_beta = linear_beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        gate = x[..., :d].float()    # 门控分支
        up   = x[..., d:].float()    # 线性分支
        situ = self.beta * torch.tanh(gate / self.beta) * torch.sigmoid(gate)
        if self.linear_beta is not None:
            up = self.linear_beta * torch.tanh(up / self.linear_beta)
        return (situ * up).to(x.dtype)


# ─────────────────────────────────────────────
# 1.3  KimiMLP  (标准 FFN)
# ─────────────────────────────────────────────
class KimiMLP(nn.Module):
    """
    SwiGLU 风格 FFN：
      down_proj( act(gate_proj(x)) * up_proj(x) )

    Kimi 用 SituAndMul 替换了 SiLU，原理类似但更平滑。
    """
    def __init__(self, hidden_size: int, intermediate_size: int, beta: float = 1.0):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn    = SituAndMul(beta=beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # gate & up 拼接后送入 SituAndMul
        gate_up = torch.cat([self.gate_proj(x), self.up_proj(x)], dim=-1)
        return self.down_proj(self.act_fn(gate_up))


print(f"\n[KimiMLP] 参数演示:")
mlp = KimiMLP(hidden_size=16, intermediate_size=32)
x_test = torch.randn(1, 5, 16)
trace("MLP 输入", x_test)
trace("gate_proj 输出", mlp.gate_proj(x_test))
trace("SituAndMul 后 (猫接 gate+up)", torch.cat([mlp.gate_proj(x_test), mlp.up_proj(x_test)], -1))
trace("MLP 最终输出", mlp(x_test))


# ══════════════════════════════════════════════════════════════════════════════
# 第二部分：KimiMLAAttention（多潜空间注意力）
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*70)
print("第二部分：KimiMLAAttention（Multi-Latent Attention）")
print("═"*70)

"""
MLA 核心思想（来自 DeepSeek-V3）：
  传统 MHA：KV 缓存 = num_heads × head_dim × seq_len（显存瓶颈）
  MLA：     先把 hidden 压缩到小的 kv_lora_rank 维潜向量，
            推理时再展开，大幅减少 KV cache。

关键参数：
  qk_nope_head_dim: Q/K 中不加 RoPE 的部分（参与 KV 压缩）
  qk_rope_head_dim: Q/K 中加 RoPE 的部分（位置信息）
  kv_lora_rank:     KV 压缩维度（远小于 num_heads × head_dim）
  v_head_dim:       Value 维度

数据流（Kimi 版本，use_nope=True 即 q_rot 不实际加入 key 匹配）：
  hidden → q_a_proj → RMSNorm → q_b_proj → [q_pass | q_rot]
  hidden → kv_a_proj_with_mqa → [kv_latent | k_rot]
         kv_latent → RMSNorm → kv_b_proj → [k_pass | v]
  attn(q=[q_pass,q_rot], k=[k_pass,k_rot], v) → o_proj
  （当 use_nope=True 时，q_rot 和 k_rot 只是扩展维度，实际不做 RoPE）
"""

class TinyMLAAttention(nn.Module):
    """
    精简版 MLA，不含 RoPE，方便 CPU debug。
    去掉了 flash-attn，用标准 scaled dot-product。
    """
    def __init__(
        self,
        hidden_size: int = 32,
        num_heads: int = 4,
        qk_nope_head_dim: int = 8,   # nope 部分维度（主要参与注意力）
        qk_rope_head_dim: int = 4,   # rope 部分维度（本 demo 不加 RoPE）
        kv_lora_rank: int = 8,       # KV 压缩瓶颈维度（小！）
        v_head_dim: int = 8,         # Value 维度
        q_lora_rank: int = 16,       # Q 压缩维度（可选）
    ):
        super().__init__()
        self.num_heads = num_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.q_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.v_head_dim = v_head_dim
        self.hidden_size = hidden_size
        self.scale = self.q_head_dim ** -0.5

        # ── Q 路径（LoRA 风格低秩投影）────────────────────────────────────
        # hidden → [q_lora_rank] → [num_heads × q_head_dim]
        # 参数量：hidden*q_lora_rank + q_lora_rank*num_heads*q_head_dim
        # 对比直接投影：hidden * num_heads * q_head_dim（相同）
        self.q_a_proj = nn.Linear(hidden_size, q_lora_rank, bias=False)   # 压缩
        self.q_a_norm = KimiRMSNorm(q_lora_rank)
        self.q_b_proj = nn.Linear(q_lora_rank, num_heads * self.q_head_dim, bias=False)  # 展开

        # ── KV 路径（核心压缩）────────────────────────────────────────────
        # hidden → [kv_lora_rank + qk_rope_head_dim]
        # kv_lora_rank 部分：存 KV 的潜向量，推理时只缓存这个！
        # qk_rope_head_dim 部分：k_rot，每头共享一份
        self.kv_a_proj = nn.Linear(
            hidden_size,
            kv_lora_rank + qk_rope_head_dim,  # [KV潜向量 | K的RoPE部分]
            bias=False
        )
        self.kv_a_norm = KimiRMSNorm(kv_lora_rank)
        # 展开：kv_lora_rank → num_heads × (qk_nope_head_dim + v_head_dim)
        self.kv_b_proj = nn.Linear(
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),
            bias=False
        )

        # ── 输出投影 ────────────────────────────────────────────────────────
        self.o_proj = nn.Linear(num_heads * v_head_dim, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        B, T, _ = x.shape
        if verbose: trace("MLA 输入 x", x)

        # ─ Q ─────────────────────────────────────────────────────────────────
        q = self.q_b_proj(self.q_a_norm(self.q_a_proj(x)))          # B,T, H*q_head
        q = q.view(B, T, self.num_heads, self.q_head_dim)            # B,T,H,q_head
        q_pass, q_rot = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        if verbose:
            trace("  q_pass (nope part, attends to k_pass)", q_pass)
            trace("  q_rot  (rope part, attends to k_rot)",  q_rot)

        # ─ KV ────────────────────────────────────────────────────────────────
        compressed_kv = self.kv_a_proj(x)                            # B,T, lora_rank+rope_dim
        kv_latent, k_rot = compressed_kv.split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        # 💡 KV 缓存只存 kv_latent (B, T, kv_lora_rank)，远小于原始 KV
        if verbose:
            trace("  kv_latent (KV cache 存的就是这个！)", kv_latent)
            trace("  k_rot (所有头共享)", k_rot)

        kv = self.kv_b_proj(self.kv_a_norm(kv_latent))               # B,T, H*(nope+v)
        kv = kv.view(B, T, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_pass, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        if verbose:
            trace("  k_pass (nope key)", k_pass)
            trace("  v      (value)",    v)

        # k_rot 扩展到每个头
        k_rot_expanded = k_rot.unsqueeze(2).expand_as(q_rot)         # B,T,H,rope_dim

        # 拼接完整 Q 和 K
        q_full = torch.cat([q_pass, q_rot], dim=-1)                  # B,T,H,q_head_dim
        k_full = torch.cat([k_pass, k_rot_expanded], dim=-1)         # B,T,H,q_head_dim

        # ─ Attention (标准 SDPA，no flash) ───────────────────────────────────
        # 转成 [B, H, T, D] 做 batch matmul
        q_full = q_full.transpose(1, 2)   # B,H,T,q_head
        k_full = k_full.transpose(1, 2)   # B,H,T,q_head
        v      = v.transpose(1, 2)        # B,H,T,v_head

        scores = torch.einsum("bhtd,bhsd->bhts", q_full, k_full) * self.scale  # B,H,T,T
        # 因果 mask
        mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
        scores = scores.masked_fill(~mask, float("-inf"))
        attn_w = F.softmax(scores, dim=-1, dtype=torch.float32).to(x.dtype)
        if verbose: trace("  attn_weights", attn_w)

        out = torch.einsum("bhts,bhsd->bhtd", attn_w, v)            # B,H,T,v_head
        out = out.transpose(1, 2).reshape(B, T, -1)                  # B,T,H*v_head
        out = self.o_proj(out)
        if verbose: trace("MLA 输出", out)
        return out


print("\n[MLA 注意力] 演示:")
mla = TinyMLAAttention()
x_demo = torch.randn(1, 6, 32)
trace("输入", x_demo)
out_mla = mla(x_demo, verbose=True)
print(f"\n  ✅ MLA 前向成功！输入 {shape_str(x_demo)} → 输出 {shape_str(out_mla)}")

# 关键参数量比较
h, nope, rope, kv_r, v = 4, 8, 4, 8, 8
hidden = 32
q_lora = 16
mha_kv_params = 2 * hidden * h * (nope + rope)  # 传统 MHA 的 KV 参数量
mla_kv_params  = hidden * (kv_r + rope) + kv_r * h * (nope + v)  # MLA 的 KV 参数量
print(f"\n  📊 KV 参数量对比: MHA={mha_kv_params}, MLA={mla_kv_params}  "
      f"(节省 {(1 - mla_kv_params/mha_kv_params)*100:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# 第三部分：KimiDeltaAttention（线性注意力 KDA）
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*70)
print("第三部分：KimiDeltaAttention（KDA 线性注意力）")
print("═"*70)

"""
KDA = Key-Delta Attention（论文核心创新之一）

动机：标准注意力是 O(T²) 的，长序列推理慢。
KDA 是一种线性注意力的变体，复杂度 O(T)，但能保留近似全量注意力的效果。

KDA 的关键思想：
  1. 用 Short Convolution（短卷积 3~4 个 token）做 Q/K/V 的局部特征提取
  2. 维护一个"状态矩阵" S（类似 RNN hidden state）
  3. 引入 delta 更新规则：S_t = decay(A) * S_{t-1} + beta_t * k_t^T * v_t
       其中 beta_t 控制写入强度（类似 Mamba 的 selectivity）
  4. 输出 o_t = q_t @ S_t，然后通过 o_norm（门控 RMSNorm）

本 Demo 用简化版（不含 chunk_kda 的 CUDA kernel）演示数据流。
"""

class SimplifiedShortConv(nn.Module):
    """简化的短卷积，模拟 fla.modules.ShortConvolution 的行为"""
    def __init__(self, hidden_size: int, kernel_size: int = 4):
        super().__init__()
        self.conv = nn.Conv1d(
            hidden_size, hidden_size,
            kernel_size=kernel_size, padding=kernel_size - 1,
            groups=hidden_size, bias=True  # depthwise conv
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, cache=None, output_final_state=False, **kwargs):
        # x: (B, T, C) → transpose → (B, C, T) → conv → SiLU → back
        B, T, C = x.shape
        out = self.act(self.conv(x.transpose(1, 2))[..., :T].transpose(1, 2))
        if output_final_state:
            # 简化：返回最后 kernel_size 个 token 作为 conv state
            return out, x[:, -self.conv.kernel_size[0]+1:, :]
        return out, None


class TinyKDAAttention(nn.Module):
    """
    简化版 KDA，用 Python 循环模拟 chunk-wise 线性递推。
    目的是展示数据流，不是高性能实现。
    """
    def __init__(self, hidden_size: int = 32, num_heads: int = 4, head_dim: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        proj_size = num_heads * head_dim

        # 投影层
        self.q_proj = nn.Linear(hidden_size, proj_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, proj_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, proj_size, bias=False)

        # 短卷积（局部特征提取，比 RoPE 更灵活）
        self.q_conv = SimplifiedShortConv(proj_size)
        self.k_conv = SimplifiedShortConv(proj_size)
        self.v_conv = SimplifiedShortConv(proj_size)

        # A: 衰减因子（类似 Mamba 的 A）
        self.A_log = nn.Parameter(torch.log(
            torch.empty(num_heads).uniform_(1, 16)))

        # delta 相关：控制 KV 写入状态的强度
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)   # beta

        # g：类似 GRU 的遗忘门（低秩）
        self.g_a_proj = nn.Linear(hidden_size, head_dim, bias=False)
        self.g_b_proj = nn.Linear(head_dim, proj_size, bias=False)

        # 输出归一化（门控 RMSNorm）
        self.o_norm = nn.RMSNorm(head_dim)
        self.o_proj = nn.Linear(proj_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        B, T, _ = x.shape
        if verbose: trace("KDA 输入 x", x)

        # 步骤1：线性投影
        q_proj = self.q_proj(x)   # B,T,H*D
        k_proj = self.k_proj(x)
        v_proj = self.v_proj(x)

        # 步骤2：短卷积（捕获局部上下文，这是区别于 Mamba 的关键）
        q, _ = self.q_conv(q_proj)   # B,T,H*D
        k, _ = self.k_conv(k_proj)
        v, _ = self.v_conv(v_proj)
        if verbose:
            trace("  q 经短卷积后", q)
            trace("  k 经短卷积后", k)
            trace("  v 经短卷积后", v)

        # reshape 为 (B,T,H,D)
        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim)
        v = v.view(B, T, self.num_heads, self.head_dim)

        # 步骤3：归一化 q,k（L2 norm，让内积有界，类似 cosine attention）
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        # 步骤4：计算 beta（写入强度）和衰减 decay
        beta  = self.b_proj(x).sigmoid()           # B,T,H  ∈ (0,1)
        decay = (-self.A_log.exp()).exp()           # H, 衰减系数 < 1
        if verbose:
            trace("  beta (写入强度)", beta)
            trace("  decay (per head)", decay.unsqueeze(0).unsqueeze(0).expand(1,1,-1))

        # 步骤5：Python 循环模拟线性递推（演示用，真实用 chunk_kda CUDA kernel）
        #   S_t = decay_h * S_{t-1} + beta_t * k_t^T @ v_t
        #   o_t = q_t @ S_t
        S = torch.zeros(B, self.num_heads, self.head_dim, self.head_dim, device=x.device)
        outputs = []
        for t in range(T):
            q_t = q[:, t]    # B,H,D
            k_t = k[:, t]    # B,H,D
            v_t = v[:, t]    # B,H,D
            b_t = beta[:, t] # B,H

            # 更新状态：S += beta * outer(k, v)
            kv_outer = torch.einsum('bhd,bhe->bhde', k_t, v_t)  # B,H,D,D (外积)
            S = decay.view(1,-1,1,1) * S + b_t.view(B,-1,1,1) * kv_outer

            # 读取：o = q @ S
            o_t = torch.einsum('bhd,bhde->bhe', q_t, S)         # B,H,D
            outputs.append(o_t)

        o = torch.stack(outputs, dim=1)  # B,T,H,D

        # 步骤6：门控输出归一化
        g = self.g_b_proj(self.g_a_proj(x))          # B,T,H*D
        g = g.view(B, T, self.num_heads, self.head_dim).sigmoid()
        o = self.o_norm(o) * g                        # B,T,H,D (元素乘)
        if verbose: trace("  o after o_norm * gate", o)

        # 步骤7：输出投影
        o = o.reshape(B, T, -1)
        o = self.o_proj(o)
        if verbose: trace("KDA 输出", o)
        return o


print("\n[KDA 线性注意力] 演示:")
kda = TinyKDAAttention()
x_demo = torch.randn(1, 8, 32)
trace("输入", x_demo)
out_kda = kda(x_demo, verbose=True)
print(f"\n  ✅ KDA 前向成功！输入 {shape_str(x_demo)} → 输出 {shape_str(out_kda)}")
print(f"\n  💡 KDA vs Transformer: O(T) vs O(T²) 时间复杂度")
print(f"     状态矩阵 S 形状: (B={1}, H={4}, D={8}, D={8}) = {1*4*8*8} float32 元素")


# ══════════════════════════════════════════════════════════════════════════════
# 第四部分：MoE (Mixture of Experts)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*70)
print("第四部分：KimiSparseMoeBlock（MoE）")
print("═"*70)

"""
MoE 核心思想：
  - 用 N 个小 Expert（MLP）代替一个大 MLP
  - 每个 token 只激活 top-K 个 Expert
  - 总参数量大，但每次计算量 ≈ top-K 个 MLP（稀疏激活）
  - Kimi-K3 额外特性：
      1. 分组 top-k：先按组选，再在组内选，提高多样性
      2. noaux_tc：无辅助损失的 top-K，通过 e_score_correction_bias 避免 load imbalance
      3. Latent MoE（可选）：路由前把 hidden 投影到更小的 moe_hidden_size 再路由

Router 打分：
  scores = sigmoid(W_gate @ x)   （不是 softmax！）
  top-k 选择 + e_score_correction_bias 纠偏
  最终权重归一化
"""

class TinyMoEGate(nn.Module):
    """简化版 MoE 路由器"""
    def __init__(self, hidden_size: int, num_experts: int, top_k: int,
                 num_expert_groups: int = 2, topk_groups: int = 1):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.num_expert_groups = num_expert_groups
        self.topk_groups = topk_groups

        self.weight = nn.Parameter(torch.empty(num_experts, hidden_size))
        # 纠偏参数：补偿不同 expert 被选概率的差异
        self.e_score_correction_bias = nn.Parameter(torch.zeros(num_experts))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor, verbose: bool = False):
        # x: (N_tokens, hidden)
        N = x.shape[0]
        scores = F.linear(x.float(), self.weight.float()).sigmoid()  # N, E
        if verbose: trace("  路由 scores (sigmoid)", scores)

        # 加纠偏后选 top-k（分组版）
        scores_biased = scores + self.e_score_correction_bias         # N, E

        if self.num_expert_groups > 1:
            # 分组：每组取 top-2 之和作为组分数，然后选 topk_groups 个组
            E_per_group = self.num_experts // self.num_expert_groups
            group_scores = scores_biased.view(N, self.num_expert_groups, E_per_group)
            group_scores = group_scores.topk(2, dim=-1)[0].sum(-1)    # N, G
            top_groups = group_scores.topk(self.topk_groups, dim=-1)[1]  # N, topk_g
            # 构造 mask
            group_mask = torch.zeros(N, self.num_expert_groups, device=x.device)
            group_mask.scatter_(1, top_groups, 1.0)                   # N, G
            expert_mask = group_mask.unsqueeze(-1).expand(
                N, self.num_expert_groups, E_per_group).reshape(N, -1) # N, E
            scores_biased = scores_biased.masked_fill(expert_mask == 0, float('-inf'))

        _, topk_idx = scores_biased.topk(self.top_k, dim=-1)          # N, top_k
        topk_weight = scores.gather(1, topk_idx)                      # N, top_k

        # 归一化权重（使得每个 token 的权重之和 ≈ 1）
        topk_weight = topk_weight / (topk_weight.sum(-1, keepdim=True) + 1e-20)
        if verbose:
            trace("  topk_idx (选中的 expert 编号)", topk_idx)
            trace("  topk_weight (归一化权重)", topk_weight)
        return topk_idx, topk_weight


class TinySparseMoE(nn.Module):
    """简化版 Sparse MoE Block"""
    def __init__(self, hidden_size: int, num_experts: int = 8,
                 top_k: int = 2, expert_intermediate_size: int = 16):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k

        self.gate = TinyMoEGate(hidden_size, num_experts, top_k, num_expert_groups=2, topk_groups=1)

        # 每个 Expert 是一个小 MLP
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, expert_intermediate_size, bias=False),
                nn.SiLU(),
                nn.Linear(expert_intermediate_size, hidden_size, bias=False),
            )
            for _ in range(num_experts)
        ])

        # Shared Expert（所有 token 都经过的 dense 部分）
        self.shared_expert = nn.Sequential(
            nn.Linear(hidden_size, expert_intermediate_size, bias=False),
            nn.SiLU(),
            nn.Linear(expert_intermediate_size, hidden_size, bias=False),
        )

    def forward(self, x: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        B, T, D = x.shape
        x_flat = x.view(-1, D)   # N_tokens, D
        if verbose: trace("MoE 输入 (展平)", x_flat)

        topk_idx, topk_weight = self.gate(x_flat, verbose=verbose)   # N, top_k

        # 按 expert 排序并批量处理（避免 for token loop）
        y = torch.zeros_like(x_flat)
        counts = topk_idx.new_zeros(self.num_experts)
        for k in range(self.top_k):
            expert_ids = topk_idx[:, k]                                # N
            weights    = topk_weight[:, k]                             # N
            for e_id in range(self.num_experts):
                mask = (expert_ids == e_id)
                if mask.any():
                    tokens_e = x_flat[mask]
                    out_e    = self.experts[e_id](tokens_e)
                    y[mask] += weights[mask].unsqueeze(-1) * out_e
                    counts[e_id] += mask.sum()

        if verbose:
            print(f"  📊 各 Expert 处理 token 数: {counts.tolist()}")

        # Shared Expert（固定激活，不参与路由）
        y = y + self.shared_expert(x_flat)
        return y.view(B, T, D)


print("\n[MoE Block] 演示:")
moe = TinySparseMoE(hidden_size=16, num_experts=8, top_k=2)
x_demo = torch.randn(1, 4, 16)
trace("输入", x_demo)
out_moe = moe(x_demo, verbose=True)
trace("输出", out_moe)
print(f"\n  ✅ MoE 前向成功！每个 token 激活 {moe.top_k}/{moe.num_experts} 个 expert")


# ══════════════════════════════════════════════════════════════════════════════
# 第五部分：Attention Residuals（跨层残差聚合）
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*70)
print("第五部分：AttnResidual（跨层残差聚合，Kimi 独创）")
print("═"*70)

"""
AttnResidual 是 Kimi-K3 的独创机制：

传统 Transformer：
  h_i = h_{i-1} + Attn(h_{i-1}) + MLP(h_{i-1}+Attn)

AttnResidual：
  每 attn_res_block_size 层，把该层的 hidden state 存入 block_residual buffer
  在每一层的注意力之前，从 block_residual 中加权聚合历史状态：
    h = softmax_weighted_sum([h_old0, h_old4, h_old8, ..., h_current])
  
  权重由 softmax(norm(h_i) * score_proj) 决定，每层单独学习。

直觉：这类似于 DenseNet 的密集连接，让每层都能"回望"较早的表示，
有助于信息的长距离传递，尤其配合线性注意力使用时效果更好。
"""

def apply_attn_res(prefix_sum: torch.Tensor,
                   block_residual: torch.Tensor,
                   proj: nn.Linear,
                   norm: KimiRMSNorm) -> torch.Tensor:
    """
    prefix_sum:     (N, hidden)     当前层的输出
    block_residual: (N, K, hidden)  历史关键层的输出（K个）
    
    返回：加权聚合后的新 hidden state
    """
    # 拼接历史 + 当前
    v = torch.cat([block_residual, prefix_sum.unsqueeze(1)], dim=1)   # N, K+1, hidden

    # 用 norm + proj 计算每个位置的分数（这是 shared-norm attention）
    v_float = v.float()
    var = v_float.pow(2).mean(-1, keepdim=True)
    k = v_float * torch.rsqrt(var + norm.eps)                          # N, K+1, hidden

    # score_weight = norm.weight * proj.weight（合并 norm 和 proj 的参数）
    score_weight = norm.weight.float() * proj.weight.squeeze(0).float() # hidden
    scores = (k * score_weight).sum(-1)                                # N, K+1

    # softmax → 加权求和
    probs = scores.softmax(-1).unsqueeze(1)                            # N, 1, K+1
    out = torch.matmul(probs, v_float).squeeze(1)                      # N, hidden
    return out.to(v.dtype)


# 演示
print("\n[AttnResidual] 演示:")
H = 16
attn_res_norm = KimiRMSNorm(H)
attn_res_proj = nn.Linear(H, 1, bias=False)

# 模拟 3 个历史层 + 当前层
block_residual = torch.randn(4, 3, H)   # N=4, K=3, H=16
prefix_sum     = torch.randn(4, H)      # N=4, H=16

trace("block_residual (历史 K 层的 hidden)", block_residual)
trace("prefix_sum (当前层输出)", prefix_sum)
out = apply_attn_res(prefix_sum, block_residual, attn_res_proj, attn_res_norm)
trace("AttnResidual 输出", out)
print(f"\n  ✅ AttnResidual 将 {block_residual.shape[1]} 个历史状态聚合到当前层")


# ═══════════════════════════��══════════════════════════════════════════════════
# 第六部分：完整的 Tiny KimiLinear 模型（端到端）
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*70)
print("第六部分：完整的 Tiny KimiLinear 语言模型（端到端推理）")
print("═"*70)

class TinyKimiDecoderLayer(nn.Module):
    """
    单个 Decoder Layer，包含：
    - 自注意力（MLA 或 KDA，由 is_kda 决定）
    - FFN（MoE 或 dense MLP）
    - AttnResidual（可选）
    """
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        layer_idx: int,
        use_moe: bool = False,
        is_kda: bool = False,
        use_attn_residual: bool = False,
        attn_res_block_size: int = 4,
        num_heads: int = 4,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_kda = is_kda
        self.use_attn_residual = use_attn_residual
        self.attn_res_block_size = attn_res_block_size

        if is_kda:
            self.self_attn = TinyKDAAttention(hidden_size, num_heads, hidden_size // num_heads)
        else:
            # 简化 MLA 参数
            head_dim = hidden_size // num_heads
            self.self_attn = TinyMLAAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                qk_nope_head_dim=head_dim // 2,
                qk_rope_head_dim=head_dim // 2,
                kv_lora_rank=head_dim,
                v_head_dim=head_dim // 2,
                q_lora_rank=head_dim * 2,
            )

        if use_moe and layer_idx >= 1:  # 第一层 dense
            self.ffn = TinySparseMoE(hidden_size, num_experts=4, top_k=2,
                                     expert_intermediate_size=intermediate_size)
        else:
            self.ffn = KimiMLP(hidden_size, intermediate_size)

        self.input_norm  = KimiRMSNorm(hidden_size)
        self.post_norm   = KimiRMSNorm(hidden_size)

        if use_attn_residual:
            self.attn_res_norm_pre  = KimiRMSNorm(hidden_size)
            self.attn_res_proj_pre  = nn.Linear(hidden_size, 1, bias=False)
            self.attn_res_norm_post = KimiRMSNorm(hidden_size)
            self.attn_res_proj_post = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_states, block_residual=None, verbose=False):
        B, T, D = hidden_states.shape

        if self.use_attn_residual and block_residual is not None and block_residual.shape[1] > 0:
            # 注意力之前：聚合历史残差
            prefix = hidden_states.view(-1, D)
            hidden_states = apply_attn_res(
                prefix, block_residual, self.attn_res_proj_pre, self.attn_res_norm_pre
            ).view(B, T, D)
            if verbose: trace(f"  Layer {self.layer_idx} AttnRes 前聚合后", hidden_states)

        # 每 attn_res_block_size 层：把当前 hidden 存入 block_residual
        if self.use_attn_residual and self.layer_idx % self.attn_res_block_size == 0:
            new_br = hidden_states.view(-1, D).unsqueeze(1)
            block_residual = torch.cat([block_residual, new_br], dim=1) if block_residual is not None else new_br

        # Self Attention
        residual = hidden_states
        hidden_states = self.input_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states
        if verbose: trace(f"  Layer {self.layer_idx} ({'KDA' if self.is_kda else 'MLA'}) 后", hidden_states)

        if self.use_attn_residual:
            # 注意力之后：再次聚合
            prefix = hidden_states.view(-1, D)
            hidden_states = apply_attn_res(
                prefix, block_residual, self.attn_res_proj_post, self.attn_res_norm_post
            ).view(B, T, D)

        # FFN
        residual = hidden_states
        hidden_states = self.post_norm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states
        if verbose: trace(f"  Layer {self.layer_idx} FFN 后", hidden_states)

        return hidden_states, block_residual


class TinyKimiLM(nn.Module):
    """
    完整的 Tiny KimiLinear 语言模型。
    
    层结构（模拟 Kimi-K3 的混合架构）：
      Layer 0: MLA + Dense MLP    （前几层用 full attention）
      Layer 1: KDA + MoE          （中间层用线性注意力 + MoE）
      Layer 2: MLA + MoE          （全量注意力 + MoE）
      Layer 3: KDA + Dense MLP    （线性注意力 + dense）
    
    所有层都使用 AttnResidual（跨层残差聚合）。
    """
    def __init__(
        self,
        vocab_size: int = 512,
        hidden_size: int = 32,
        intermediate_size: int = 64,
        num_layers: int = 4,
        num_heads: int = 4,
        # Kimi-K3 真实参数比例（近似）：
        # ~30% 层是 KDA，~70% 是 MLA；~80% 层是 MoE，~20% 是 dense
        kda_layer_ids = (1, 3),
        moe_layer_ids = (1, 2),
        use_attn_residual: bool = True,
        attn_res_block_size: int = 2,
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.use_attn_residual = use_attn_residual

        self.layers = nn.ModuleList([
            TinyKimiDecoderLayer(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                layer_idx=i,
                use_moe=(i in moe_layer_ids),
                is_kda=(i in kda_layer_ids),
                use_attn_residual=use_attn_residual,
                attn_res_block_size=attn_res_block_size,
                num_heads=num_heads,
            )
            for i in range(num_layers)
        ])

        self.norm = KimiRMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        if use_attn_residual:
            self.output_attn_res_norm = KimiRMSNorm(hidden_size)
            self.output_attn_res_proj = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, input_ids: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        B, T = input_ids.shape
        x = self.embed_tokens(input_ids)              # B,T,H
        if verbose: trace("Embedding 输出", x)

        block_residual = None
        if self.use_attn_residual:
            block_residual = x.new_zeros(B * T, 0, x.shape[-1])

        for layer in self.layers:
            x, block_residual = layer(x, block_residual, verbose=verbose)

        if self.use_attn_residual and block_residual is not None and block_residual.shape[1] > 0:
            B2, T2, D = x.shape
            x = apply_attn_res(
                x.view(-1, D), block_residual,
                self.output_attn_res_proj, self.output_attn_res_norm
            ).view(B2, T2, D)
            if verbose: trace("最终 AttnResidual 输出聚合后", x)

        x = self.norm(x)                              # B,T,H
        logits = self.lm_head(x)                      # B,T,V
        if verbose: trace("Logits", logits)
        return logits

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 10,
                 temperature: float = 1.0) -> torch.Tensor:
        """贪心/采样生成（无 KV cache 简化版）"""
        for _ in range(max_new_tokens):
            logits = self.forward(input_ids)           # B,T,V
            next_logits = logits[:, -1, :] / temperature
            next_token = next_logits.softmax(-1).multinomial(1)  # B,1
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids


# 实例化并测试
print("\n构建 TinyKimiLM 模型...")
model = TinyKimiLM(
    vocab_size=512,
    hidden_size=32,
    intermediate_size=64,
    num_layers=4,
    num_heads=4,
    use_attn_residual=True,
    attn_res_block_size=2,
)

total_params = sum(p.numel() for p in model.parameters())
print(f"\n  📊 模型总参数量: {total_params:,}")

# 打印层级结构
print("\n  📋 层级结构:")
for i, layer in enumerate(model.layers):
    attn_type = "KDA (线性)" if layer.is_kda else "MLA (全量)"
    ffn_type  = "MoE" if isinstance(layer.ffn, TinySparseMoE) else "Dense MLP"
    print(f"     Layer {i}: {attn_type} + {ffn_type}"
          + (" + AttnResidual" if layer.use_attn_residual else ""))

# 前向测试（带 verbose）
print("\n" + "─"*60)
print("🔍 前向推理追踪 (input_ids=[1,4]):")
print("─"*60)
input_ids = torch.randint(1, 512, (1, 4))
print(f"  input_ids: {input_ids.tolist()}")
logits = model(input_ids, verbose=True)

print("\n" + "─"*60)
print("🎲 自回归生成测试:")
print("─"*60)
input_ids_gen = torch.randint(1, 100, (1, 3))
print(f"  输入 token ids: {input_ids_gen.tolist()}")
generated = model.generate(input_ids_gen, max_new_tokens=5)
print(f"  生成后 token ids: {generated.tolist()}")
print(f"  新生成了 {generated.shape[1] - input_ids_gen.shape[1]} 个 token")


# ══════════════════════════════════════════════════════════════════════════════
# 第七部分：视觉编码器 MoonViT3d
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*70)
print("第七部分：MoonViT3d 视觉编码器")
print("═"*70)

"""
MoonViT3d 处理图像/视频的流程：

输入：pixel_values (N_patches, C, pH, pW)  其中 C=3, pH=pW=patch_size=14
      grid_thws (N_images, 3)  每张图的 (T, H, W)，即帧数×高度×宽度（以 patch 为单位）

步骤1：PatchEmbed（Conv2d + 位置编码）
  - 用 Conv2d(stride=patch_size) 将每个 patch 转成一个向量
  - 加入 2D 可学习位置编码（支持双线性插值到任意分辨率）
  - 加入时间维的 sincos 位置编码

步骤2：MoonViTEncoder（标准 ViT 编码器，加 2D RoPE）
  - 多层 MoonViTEncoderLayer（Pre-LN Attention + MLP）
  - 注意力用 2D RoPE（X/Y 方向各用一半维度）
  - 所有 patch 全量互注意（非因果）

步骤3：tpool_patch_merger（时序池化 + 空间下采样）
  - 在时间维度做平均池化（T 帧 → 1）
  - 在空间维度做 2×2 的 patch 合并（H/2, W/2 个 super-patch）
  - 每个 super-patch = [2×2 个小 patch 的向量]（拼接或平均）

步骤4：PatchMergerMLPV2（投影到 LLM hidden size）
  - pre_norm + MLP(4×hidden → hidden_llm) + post_norm
"""

class TinyPatchEmbed(nn.Module):
    """简化的 3D Patch Embedding"""
    def __init__(self, vit_dim: int = 32, patch_size: int = 4, in_channels: int = 3):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, vit_dim, kernel_size=patch_size, stride=patch_size)
        # 可学习的 2D 位置编码（简化为固定大小 8×8）
        self.pos_emb = nn.Parameter(torch.randn(8, 8, vit_dim) * 0.02)

    def forward(self, pixel_values: torch.Tensor, grid_thws: torch.Tensor):
        """
        pixel_values: (N_patches_total, C, pH, pW)
                       N_patches_total = sum(T*H*W) across all images
        grid_thws: (N_images, 3)  each row = (T, H, W)
        """
        # Conv2d 提取 patch 特征
        x = self.proj(pixel_values)            # N_total, vit_dim, 1, 1
        x = x.view(x.shape[0], -1)            # N_total, vit_dim

        # 加位置编码（简化：直接切片）
        pos_embs = []
        idx = 0
        for t, h, w in grid_thws.tolist():
            n_patches = t * h * w
            # 从 8×8 的位置编码取 h×w 部分（简化版，真实代码用双线性插值）
            h_clip, w_clip = min(h, 8), min(w, 8)
            pos = self.pos_emb[:h_clip, :w_clip].reshape(-1, self.pos_emb.shape[-1])
            if pos.shape[0] < n_patches:
                pos = pos.repeat(math.ceil(n_patches / pos.shape[0]), 1)[:n_patches]
            pos_embs.append(pos)
            idx += n_patches
        pos_emb_all = torch.cat(pos_embs, dim=0)
        return x + pos_emb_all


class TinyViTBlock(nn.Module):
    """简化的 ViT 编码器层（全量双向注意力 + FFN）"""
    def __init__(self, vit_dim: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(vit_dim)
        self.norm2 = nn.LayerNorm(vit_dim)
        self.attn  = nn.MultiheadAttention(vit_dim, num_heads, batch_first=True)
        self.mlp   = nn.Sequential(
            nn.Linear(vit_dim, vit_dim * 4),
            nn.GELU(),
            nn.Linear(vit_dim * 4, vit_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN Transformer block（双向注意力，无 causal mask）
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class TinyMoonViT3d(nn.Module):
    """简化版 MoonViT3d 演示"""
    def __init__(self, vit_dim=32, num_vit_layers=2, num_heads=4,
                 merge_kernel=(2,2), patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.merge_kernel = merge_kernel
        self.patch_embed = TinyPatchEmbed(vit_dim, patch_size)
        self.encoder = nn.Sequential(*[TinyViTBlock(vit_dim, num_heads)
                                        for _ in range(num_vit_layers)])
        self.final_norm = nn.LayerNorm(vit_dim)

    def forward(self, pixel_values: torch.Tensor, grid_thws: torch.Tensor, verbose=False):
        """
        pixel_values: (N_total, 3, patch_size, patch_size)
        grid_thws:    (N_images, 3)   T, H, W per image
        """
        if verbose: trace("ViT 输入 pixel_values", pixel_values)

        # 1. Patch Embedding
        x = self.patch_embed(pixel_values, grid_thws)   # N_total, vit_dim
        if verbose: trace("Patch Embed 后", x)

        # 2. ViT Encoder（把所有 patch 当一个 batch，无跨图 attention）
        x = x.unsqueeze(0)                              # 1, N_total, vit_dim
        x = self.encoder(x)
        x = self.final_norm(x).squeeze(0)               # N_total, vit_dim
        if verbose: trace("ViT Encoder 后", x)

        # 3. Temporal Pooling + Spatial Merging
        kh, kw = self.merge_kernel
        outputs = []
        ptr = 0
        for t, h, w in grid_thws.tolist():
            n = t * h * w
            seq = x[ptr:ptr + n]                        # t*h*w, D
            # 3a. Temporal pooling：沿 T 维平均
            seq = seq.view(t, h, w, -1).mean(0)         # h, w, D
            # 3b. Spatial merging：2×2 patch 合并 → 拼接
            new_h, new_w = h // kh, w // kw
            seq = seq.view(new_h, kh, new_w, kw, -1)
            seq = seq.permute(0, 2, 1, 3, 4).contiguous()  # new_h, new_w, kh, kw, D
            seq = seq.view(new_h * new_w, kh * kw, -1)  # (new_h*new_w), (kh*kw), D
            if verbose and ptr == 0:
                trace(f"  Image 0 temporal pool后 (h={h},w={w}→{new_h},{new_w})", seq)
            outputs.append(seq)
            ptr += n

        return outputs  # list of (n_super_patches, kh*kw, D)


# ─── 演示视觉编码器 ─────────────────────────────────────────────────────────
print("\n[MoonViT3d] 演示:")
print("  模拟处理 2 张图像（分辨率不同）+ 1 段视频（2 帧）")

VIT_DIM   = 32
PATCH_SZ  = 4     # 真实是 14

# 模拟图像：batch 化（所有 patch 拼在一起）
# 图1: 1帧, 4×4 patches, 图2: 1帧, 2×4 patches, 视频: 2帧, 2×2 patches
grid_thws = torch.tensor([
    [1, 4, 4],    # image 1: 1×4×4 = 16 patches
    [1, 2, 4],    # image 2: 1×2×4 = 8 patches
    [2, 2, 2],    # video:   2×2×2 = 8 patches
])
total_patches = grid_thws.prod(dim=1).sum().item()
print(f"  total_patches = {int(total_patches)} (16+8+8)")

pixel_values = torch.randn(int(total_patches), 3, PATCH_SZ, PATCH_SZ)

vit = TinyMoonViT3d(vit_dim=VIT_DIM, num_vit_layers=2, merge_kernel=(2,2), patch_size=PATCH_SZ)
outputs_vit = vit(pixel_values, grid_thws, verbose=True)

for i, out in enumerate(outputs_vit):
    trace(f"  Visual tokens image/video {i}", out)

# ─── 投影器 PatchMergerMLPV2 ──────────────────────────────────────────────
print("\n[PatchMergerMLPV2 投影器]")
kh, kw = 2, 2
patch_token_dim = VIT_DIM * kh * kw   # 拼接后维度
lm_hidden_size  = 32

projector = nn.Sequential(
    nn.Linear(patch_token_dim, patch_token_dim, bias=False),
    nn.GELU(),
    nn.Linear(patch_token_dim, lm_hidden_size, bias=False),
)
post_norm = nn.RMSNorm(lm_hidden_size)

projected = []
for out_i in outputs_vit:
    n_super = out_i.shape[0]
    flat = out_i.view(n_super, -1)       # n_super, patch_token_dim
    proj_i = post_norm(projector(flat))  # n_super, lm_hidden
    trace(f"  投影后", proj_i)
    projected.append(proj_i)

all_visual_tokens = torch.cat(projected, dim=0)
trace("所有图像/视频 visual tokens（准备插入 LLM）", all_visual_tokens)


# ══════════════════════════════════════════════════════════════════════════════
# 第八部分：完整 VLM 数据流
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*70)
print("第八部分：VLM 完整数据流（Vision + Text 融合）")
print("═"*70)

"""
KimiK3ForConditionalGeneration 的数据流：

1. 视觉路径：
   pixel_values → MoonViT3d → PatchMergerMLPV2 → image_features
   
2. 文本路径：
   input_ids → Embedding → inputs_embeds

3. 融合（_merge_input_ids_with_image_features）：
   input_ids 里有特殊的 <image> token（media_placeholder_token_id）
   每个 <image> token 被替换为对应图像的多个 visual tokens
   结果是一个 mixed_embeds，包含文字 + 图像 tokens

4. 送入语言模型：
   mixed_embeds → KimiLinearModel → lm_head → logits
"""

print("\n  模拟 token 序列：[TEXT] [IMAGE_PLACEHOLDER] [TEXT] [TEXT]")

# 假设 image placeholder token id = 100
IMAGE_TOKEN_ID = 100
VOCAB = 512
LM_DIM = 32

text_lm = TinyKimiLM(vocab_size=VOCAB, hidden_size=LM_DIM,
                     intermediate_size=64, num_layers=2,
                     use_attn_residual=True, attn_res_block_size=2)

# 模拟 input_ids：包含一个 image placeholder
input_ids = torch.tensor([[5, 12, IMAGE_TOKEN_ID, 7, 3]])  # B=1, T=5
trace("input_ids（含 image placeholder）", input_ids)

# 模拟 image features（来自 projector）
n_visual_tokens = 3   # 假设一张图被映射成 3 个 token
image_features  = torch.randn(n_visual_tokens, LM_DIM)
trace("image_features（来自 projector）", image_features)

# 融合：把 image placeholder 替换为 image tokens
# 原 T=5 tokens → 新 T = 5-1+3 = 7 tokens
def merge_tokens(input_ids, embeddings, image_features, image_token_id):
    """简化版 _merge_input_ids_with_image_features"""
    embeds = embeddings(input_ids[0])  # T, D
    parts = []
    for tok, emb in zip(input_ids[0].tolist(), embeds):
        if tok == image_token_id:
            parts.append(image_features)     # 替换为 n_visual 个 visual tokens
        else:
            parts.append(emb.unsqueeze(0))   # 保持原文字 token
    return torch.cat(parts, dim=0).unsqueeze(0)  # 1, T_new, D

merged = merge_tokens(input_ids, text_lm.embed_tokens, image_features, IMAGE_TOKEN_ID)
trace("融合后的 inputs_embeds", merged)
print(f"  序列长度变化: {input_ids.shape[1]} → {merged.shape[1]}"
      f" (placeholder 被 {n_visual_tokens} 个 visual token 替换)")

# 直接送入语言模型（跳过 embed_tokens，直接用 inputs_embeds）
# 这里我们手工模拟，传入融合后的 embeds
# 在真实代码里，语言模型接受 inputs_embeds 参数
print("\n  ✅ 完整 VLM 数据流演示完成！")


# ══════════════════════════════════════════════════════════════════════════════
# 总结
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*70)
print("架构总结")
print("═"*70)
print("""
Kimi-K3 架构关键创新点：

┌─────────────────────────────────────────────────────────────────┐
│                  KimiK3ForConditionalGeneration                 │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────────────────┐│
│  │ MoonViT3d  │→ │PatchMerger   │→ │    KimiLinearModel       ││
│  │ 视觉编码器  │  │MLP 投影器    │  │  ┌────────────────────┐  ││
│  │            │  │              │  │  │ KimiDecoderLayer   │  ││
│  │ ViT + 2D   │  │ 4×D → D_lm  │  │  │  ┌──────────────┐  │  ││
│  │ RoPE + 3D  │  │ + RMSNorm   │  │  │  │ MLAAttention │  │  ││
│  │ 时序池化   │  │              │  │  │  │ 或 KDAAttn   │  │  ││
│  └────────────┘  └──────────────┘  │  └──────────────┘  │  ││
│                                    │  ┌──────────────┐  │  ││
│                                    │  │ MoE / Dense  │  │  ││
│                                    │  └──────────────┘  │  ││
│                                    │  ┌──────────────┐  │  ││
│                                    │  │ AttnResidual │  │  ││
│                                    │  └──────────────┘  │  ││
│                                    └────────────────────┘  ││
│                                    × N层                    ││
│                                    └──────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘

1. MLA（多潜空间注意力）：
   - KV 先压缩到低维 latent，推理时缓存占用减少 70%+
   - 来自 DeepSeek-V3，Kimi 加入了 use_nope/output_gate 等变体

2. KDA（Key-Delta 线性注意力）：
   - O(T) 复杂度，对超长序列友好
   - 短卷积 + L2 归一化 + beta 门控写入
   - 与 MLA 层交替部署（混合架构）

3. MoE（稀疏专家）：
   - 分组 top-k 路由，提高多样性
   - e_score_correction_bias 避免 load imbalance
   - Latent MoE 变体：路由前投影到小维度

4. AttnResidual（跨层残差聚合）：
   - 每 attn_res_block_size 层存一个 "checkpoint"
   - 用 soft attention 聚合历史层的输出
   - 让每层都能"回望"全局信息

5. SituAndMul 激活：
   - β·tanh(x/β)·sigmoid(x) × up
   - 比 SiLU 在大激活值处更平滑，训练稳定

6. MoonViT3d 视觉编码器：
   - 2D RoPE（X/Y 分别编码），支持任意分辨率
   - 时间位置编码（sincos），支持视频
   - 时序池化 + 2×2 空间合并压缩 token 数量
""")

print("🎉 所有演示完成！运行环境:", f"Python {sys.version.split()[0]}, PyTorch {torch.__version__}")

