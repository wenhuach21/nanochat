#!/bin/bash
# Single training run for the Qwen3.5 (text-only) LLM.
# Run from the repository root:  bash nanochat/qwen35/run_qwen3p5.sh
set -euo pipefail

# export HF_HOME="/models/huggingface/"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# GPUs to use (override by exporting CUDA_VISIBLE_DEVICES before calling).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC="$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')"

# ----------------------------------------------------------------------------
# 1) (Optional) train the transformers-compatible tokenizer first.
#    Uncomment to (re)train it before the LLM.
# python -m scripts.tok_train_transformers --vocab-size=32768

# ----------------------------------------------------------------------------
# 2) Train the Qwen3.5 LLM.
torchrun --standalone --nproc_per_node="${NPROC}" \
    -m nanochat.qwen35.base_train_qwen3p5 -- \
    --run=qwen3p5 \
    --model-tag=qwen3p5-d28 \
    --tokenizer-backend=transformers \
    --depth=28 \
    --hidden-size=1024 \
    --head-dim=128 \
    --full-attention-interval=4 \
    --max-seq-len=2048 \
    --device-batch-size=16 \
    --total-batch-size=524288 \
    --target-param-data-ratio=12 \
    --warmup-ratio=0.1 \
    --lr=3e-3 \
    --core-metric-every=4000 \
    --core-metric-max-per-task=-1 \
    --sample-every=-1 \
    --save-every=5000 \
    2>&1 | tee -a train_qwen3p5.txt

