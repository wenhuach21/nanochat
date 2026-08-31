# Qwen3.5 (纯文本 LLM) nanochat 训练

本文件夹是一个**自包含**的 Qwen3.5 文本模型训练包，参考 `nanochat/qwen3.py` 的训练模式实现。
它使用 Qwen3.5 的混合架构（`GatedDeltaNet` 线性注意力层 + 门控 softmax 全注意力层），
并默认使用**与 transformers 兼容的 tokenizer**。

```
nanochat/qwen35/
├── __init__.py                 # 导出 Qwen3_5TextConfig / Qwen3_5ForCausalLM ...
├── configuration_qwen3_5.py    # 自包含的文本配置（不依赖 transformers 内置 qwen3_5）
├── qwen3p5.py                  # 文本模型 + qwen3.py 风格的训练接口
├── base_train_qwen3p5.py       # 训练脚本（默认 transformers tokenizer）
├── run_qwen3p5.sh              # 单次训练启动脚本
├── run_sweep_qwen3p5.sh        # 超参 sweep 脚本（类似 run_sweep）
└── README.md                   # 本文件
```

模型的训练接口（见 `qwen3p5.py` 的 `Qwen3_5ForCausalLM`）：
- `forward(idx, targets)` → 训练返回 loss / 推理返回 logits
- `forward1(...)` → HF 风格的原始 logits
- `re_init_weights()`、logit-softcap 退火、可选 MTP heads、`get_all_hidden_states()`（DFLASH 用）

---

## 0. 环境要求

- 一个较新的 `transformers`（本包在 `transformers==5.15.x` 上验证通过；旧版本会走内置兼容 shim，
  但 Qwen3.5 的加速 kernel 需要新版）。
- PyTorch（Muon 优化器需要带 `torch.optim.Muon` 的版本；否则用 `--optimizer-mode adamw`）。
- 数据集：nanochat 默认的 fineweb-edu 数据（见仓库根 `README.md` 的数据准备说明）。

所有命令都从**仓库根目录**执行。

---

## 0.1 加速库（让训练更快，强烈建议装）

Qwen3.5 是**混合架构**：按 `full-attention-interval` 每 4 层里有 3 层是
`GatedDeltaNet` 线性注意力（即 3/4 的层）。这些层默认走的是 `qwen3p5.py` 里的
**纯 PyTorch 回退实现**（`torch_chunk_gated_delta_rule` 等带 for 循环），只保证正确性、
**不快**。装上下面这些融合 kernel 后，模型会自动切换到快路径（`qwen3p5.py` 顶部的
`use_kernel_*` 装饰器检测到就用，检测不到就回退），训练可以快好几倍。

| 库 | pip 包 | 作用（对应 `qwen3p5.py`） | 没装的后果 |
|---|---|---|---|
| **flash-linear-attention** | `flash-linear-attention`（提供 `fla`） | GatedDeltaNet 的 `chunk_gated_delta_rule` / `recurrent_gated_delta_rule` Triton kernel（3/4 的层） | 回退到纯 torch for 循环，**最影响速度** |
| **causal-conv1d** | `causal-conv1d` | GatedDeltaNet 里的短因果卷积 `causal_conv1d_fn` / `causal_conv1d_update` | 回退到 `F.conv1d`，较慢 |
| **triton** | `triton` | 上面 `fla` 及各融合 kernel 的依赖 | `fla` 无法启用 |
| **kernels** | `kernels` | 从 HF Hub 拉取融合 kernel（`RMSNormGated`、`Qwen3_5GatedDeltaNet`），以及 FA3 加载（见下） | 走 eager / 回退实现 |
| **Flash Attention 3** | 见下 | 全注意力（softmax）层的 FA3 快路径（Hopper sm90） | 回退到 PyTorch SDPA |

安装示例（CUDA 环境，先装好匹配版本的 PyTorch 和 CUDA toolkit）：

```bash
# 线性注意力 + 短卷积的融合 kernel（最重要）
pip install triton flash-linear-attention causal-conv1d
# HF kernels 加载器（FA3 / RMSNormGated 等从 Hub 拉取）
pip install kernels
# 编译型 kernel 通常需要 ninja
pip install ninja
```

**Flash Attention（全注意力层）**：`nanochat/flash_attention.py` 会自动检测。
- **Hopper（sm90，H100/H800 等）**：通过 `kernels` 从 Hub 加载 FA3
  （`get_kernel('varunneal/flash-attention-3')`），装了 `kernels` 且能联网即可，训练日志会打印
  `✓ Using Flash Attention 3`。
- **Ampere / Ada / Blackwell 等非 sm90**：FA3 不可用，自动回退到 **PyTorch SDPA**（日志会告警）。
  这些卡上可自行 `pip install flash-attn`（FA2）加速标准注意力，但本仓库的统一接口目前只在
  FA3 与 SDPA 之间切换。

> 注意：即使不装这些库，训练也能正确跑（会用纯 torch 回退），只是慢；线性注意力占多数层，
> 所以 `flash-linear-attention` + `causal-conv1d` 收益最大，优先装。

---

## 1. 训练 tokenizer（transformers 兼容）

Qwen3.5 训练默认使用 `transformers` 后端的 tokenizer。它用 `tokenizers`(Rust) 训练一个
GPT-4 风格的 ByteLevel BPE，然后包装成 `PreTrainedTokenizerFast`，可被
`AutoTokenizer.from_pretrained(...)` 直接加载。

```bash
# 训练一个 vocab=32768 的 transformers 兼容 tokenizer
python -m scripts.tok_train_transformers \
    --max-chars=2000000000 \
    --doc-cap=10000 \
    --vocab-size=32768
```

产物会保存到 `get_tokenizer_dir("transformers")`（即 `~/.cache/nanochat/tokenizer_transformers/`，
可用 `NANOCHAT_BASE_DIR` 覆盖 base 目录），包含 `tokenizer.json`、`tokenizer_config.json`
以及给 bits-per-byte 指标用的 `token_bytes.pt`。脚本会自动做一次中英文往返编解码校验，
并验证 `AutoTokenizer.from_pretrained(...)` 能对齐。

> 说明：`transformers` 本身无法从零训练 BPE，所以这里的训练发生在 `tokenizers` 后端，
> 这是把“从零训练的 tokenizer”导入 transformers 的官方标准路径。

---

## 2. 训练 Qwen3.5 LLM

训练脚本是 `nanochat/qwen35/base_train_qwen3p5.py`，作为模块运行。

### 2.1 单卡 / CPU 冒烟测试（先跑通再上量）

```bash
python -m nanochat.qwen35.base_train_qwen3p5 \
    --depth=4 --hidden-size=128 --head-dim=32 \
    --max-seq-len=512 --device-batch-size=1 \
    --total-batch-size=512 --num-iterations=20 \
    --core-metric-every=-1 --sample-every=-1
```

### 2.2 多卡训练

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=8 \
torchrun --standalone --nproc_per_node=4 \
    -m nanochat.qwen35.base_train_qwen3p5 -- \
    --depth=14 \
    --hidden-size=1024 \
    --head-dim=128 \
    --device-batch-size=16 \
    --total-batch-size=524288 \
    --target-param-data-ratio=12 \
    --model-tag=qwen3p5-d28 \
    --sample-every=-1 \
    --save-every=1000
```

也可以直接用现成脚本：

```bash
bash nanochat/qwen35/run_qwen3p5.sh
```

### 2.3 关键参数

架构（Qwen3.5 混合层）：
- `--depth`：层数。
- `--hidden-size` / `--head-dim`：`num_attention_heads = hidden_size // head_dim`。
- `--num-kv-heads`：GQA 的 KV 头数（-1 = 与注意力头数相同）。
- `--full-attention-interval`：每隔 N 层放一层**全注意力**，其余为 **GatedDeltaNet 线性注意力**
  （默认 4，即 `L L L F` 循环）。
- `--linear-conv-kernel-dim`：GatedDeltaNet 短卷积核大小。
- `--linear-num-value-mult`：`linear_num_value_heads = num_kv_heads * 该值`（默认 2）。
- `--max-seq-len`、`--rope-theta`。

训练时长（按优先级只用其一）：
- `--num-iterations`：显式步数。
- `--target-flops`。
- `--target-param-data-ratio`：按 数据:参数 比例推算步数（Chinchilla≈20，默认 10.5）。

优化器：
- `--optimizer-mode hybrid|adamw`：hybrid 是 AdamW(嵌入/lm_head/1D 参数) + Muon(权重矩阵)。
  > 注意：GatedDeltaNet 引入了非二维参数（`A_log`、`dt_bias` 为 1D，`conv1d.weight` 为 3D），
  > 这些会自动路由到 AdamW，Muon 只处理二维权重矩阵。若你的 PyTorch 没有 `torch.optim.Muon`，
  > 请使用 `--optimizer-mode adamw`。
- `--lr` / `--embedding-lr` / `--muon-lr` / `--weight-decay`。
- `--warmup-ratio` / `--warmdown-ratio` / `--final-lr-frac` / `--lr-schedule`。
- `--grad-max-norm`、`--ema-decay` / `--ema-eval`。
- `--mtp-num-heads` / `--mtp-weight`：可选的 MTP 辅助 loss。

评估 / 保存 / 断点续训：
- `--core-metric-every`、`--core-metric-max-per-task`、`--sample-every`。
- `--save-every`、`--save-format pt|hf|both`、`--model-tag`。
- `--resume-from-step`、`--init-from`/`--init-step`、`--end-step`。
- `--expand-from`/`--expand-from-step`：从更浅的 checkpoint 做深度扩展（function-preserving）。

Tokenizer：
- Qwen3.5 默认使用 transformers tokenizer（`PreTrainedTokenizerFast`）。
- `--tokenizer-backend transformers|rustbpe`（默认 `transformers`，仍可显式切回 `rustbpe`）。

checkpoint 默认写到 `~/.cache/nanochat/base_checkpoints/qwen3p5_d<depth>/`
（或 `--model-tag` 指定的目录）。

---

## 3. 超参 sweep

用 shell sweep 脚本（类似仓库里的 `run_sweep`）批量跑不同超参组合：

```bash
# 编辑脚本顶部的候选列表，然后：
CUDA_VISIBLE_DEVICES=0,1,2,3 bash nanochat/qwen35/run_sweep_qwen3p5.sh
```

它会对 `DEPTHS × LRS × WD × WARMUP × RATIO` 的笛卡尔积逐个训练，每个组合：
- 用唯一的 `--model-tag` 和 wandb run 名，
- 把日志写到 `$NANOCHAT_BASE_DIR/hparam_sweep_qwen3p5/<tag>.log`，
- **已完整训练到最后一步的组合会自动跳过**（可安全重跑/续跑）。

如果更喜欢 Python 版 sweep（带 CSV 汇总、任务级指标解析），可以直接复用仓库根的
`run_sweep.py`，把其中的 `-m scripts.base_train_qwen3` 改成
`-m nanochat.qwen35.base_train_qwen3p5` 即可。
```

