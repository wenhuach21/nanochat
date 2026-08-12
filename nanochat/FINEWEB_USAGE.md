# FineWeb Dataset Usage Guide

## 数据下载

```bash
# 下载 10B token 样本 (~27.6GB)
python -m nanochat.fineweb --subset sample-10BT

# 下载 100B token 样本 (~277.4GB)
python -m nanochat.fineweb --subset sample-100BT

# 下载 350B token 样本 (~388GB)
python -m nanochat.fineweb --subset sample-350BT

# 下载特定的 CommonCrawl dump (推荐高质量的近期 dump)
python -m nanochat.fineweb --subset CC-MAIN-2024-10 --num-files 20
python -m nanochat.fineweb --subset CC-MAIN-2023-50 --num-files 20

# 限制下载文件数量和并行数
python -m nanochat.fineweb --subset sample-10BT --num-files 10 --num-workers 8
```

## 单独使用 FineWeb 训练

### Qwen3 架构
```bash
# Linux/Mac
python -m scripts.base_train_qwen3 --data-dir ~/.cache/nanochat/fineweb_data/sample-10BT

# Windows
python -m scripts.base_train_qwen3 --data-dir %USERPROFILE%\.cache\nanochat\fineweb_data\sample-10BT

# 分布式训练
torchrun --nproc_per_node=8 -m scripts.base_train_qwen3 --data-dir ~/.cache/nanochat/fineweb_data/sample-10BT
```

### Nanochat GPT 架构
```bash
python -m scripts.base_train --data-dir ~/.cache/nanochat/fineweb_data/sample-10BT

# 分布式训练
torchrun --nproc_per_node=8 -m scripts.base_train --data-dir ~/.cache/nanochat/fineweb_data/sample-10BT
```

## 混合数据训练 (FineWeb + fineweb-edu)

使用路径分隔符连接多个数据目录（Linux 用 `:`, Windows 用 `;`）：

### Linux/Mac
```bash
# Qwen3 - 混合训练并打乱文件顺序
python -m scripts.base_train_qwen3 \
    --data-dir ~/.cache/nanochat/base_data:~/.cache/nanochat/fineweb_data/sample-10BT \
    --shuffle-files

# Nanochat GPT
python -m scripts.base_train \
    --data-dir ~/.cache/nanochat/base_data:~/.cache/nanochat/fineweb_data/sample-10BT \
    --shuffle-files
```

### Windows
```powershell
# Qwen3
python -m scripts.base_train_qwen3 `
    --data-dir "$env:USERPROFILE\.cache\nanochat\base_data;$env:USERPROFILE\.cache\nanochat\fineweb_data\sample-10BT" `
    --shuffle-files

# Nanochat GPT
python -m scripts.base_train `
    --data-dir "$env:USERPROFILE\.cache\nanochat\base_data;$env:USERPROFILE\.cache\nanochat\fineweb_data\sample-10BT" `
    --shuffle-files
```

## 数据打乱 (Shuffle) 说明

### `--shuffle-files` 参数

混合多个数据源时，**强烈建议加上 `--shuffle-files`**，否则数据会按文件名顺序依次读取（先读完一个目录的所有文件，再读下一个），导致模型在训练前期只看到一种数据。

打乱行为：
- **文件级别打乱**: 每个 epoch 开始时，所有 parquet 文件的顺序会被随机打乱
- **确定性**: 相同 epoch 号产生相同的文件顺序（seed = 42 + epoch），保证 DDP 多卡一致
- **文档级别混合**: buffer（默认 1000 个文档）提供局部的文档级别混合
- **可恢复**: 与 `--resume-from-step` 兼容

### 不加 `--shuffle-files` 时的行为

- 文件按文件名排序依次读取
- 所有目录的文件合并后统一排序
- 最后一个文件作为 validation split，其余为 training split
- 适用于单一数据源（已经预打乱的，如 fineweb-edu-100b-shuffle）

### 单数据源 vs 混合数据源

| 场景 | 建议 |
|------|------|
| 单目录（fineweb-edu 或 FineWeb 单独） | 不需要 `--shuffle-files`（数据本身已打乱） |
| 多目录混合训练 | **加上 `--shuffle-files`** |

## 混合多个 FineWeb dump 训练

```bash
# 混合多个 dump，打乱文件顺序
python -m scripts.base_train_qwen3 \
    --data-dir ~/.cache/nanochat/fineweb_data/CC-MAIN-2024-10:~/.cache/nanochat/fineweb_data/CC-MAIN-2023-50 \
    --shuffle-files
```

## 从已有 checkpoint 初始化 + FineWeb 训练

```bash
# 先用 edu 数据训练到某个 checkpoint，再用 FineWeb 从头训练（继承权重）
python -m scripts.base_train_qwen3 \
    --init-from ~/.cache/nanochat/base_checkpoints/d14 \
    --data-dir ~/.cache/nanochat/fineweb_data/sample-10BT

# 混合数据 + 初始化权重
python -m scripts.base_train_qwen3 \
    --init-from ~/.cache/nanochat/base_checkpoints/d14 --init-step 5000 \
    --data-dir ~/.cache/nanochat/base_data:~/.cache/nanochat/fineweb_data/sample-10BT \
    --shuffle-files
```

## 环境变量

可以通过 `NANOCHAT_BASE_DIR` 环境变量自定义数据存储根目录：

```bash
export NANOCHAT_BASE_DIR=/data/nanochat
python -m nanochat.fineweb --subset sample-10BT
# 数据将下载到 /data/nanochat/fineweb_data/sample-10BT/
```

## 数据格式说明

FineWeb 数据集使用 parquet 格式，包含 `text` 列，与现有 fineweb-edu-100b-shuffle 数据完全兼容。
数据加载器使用 BOS-aligned best-fit cropping 策略，自动处理文档拼接和裁剪。

## 推荐的 FineWeb 子集

| 子集 | 大小 | 适用场景 |
|------|------|----------|
| sample-10BT | ~27.6GB | 快速实验、调试 |
| sample-100BT | ~277.4GB | 中等规模训练 |
| sample-350BT | ~388GB | 大规模训练 |
| CC-MAIN-2024-10 | ~581GB | 高质量近期数据 |
| CC-MAIN-2023-50 | ~650GB | 高质量近期数据 |

