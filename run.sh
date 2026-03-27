#!/bin/bash

# export CUBLAS_WORKSPACE_CONFIG=':4096:8'
#export HF_HOME="/models/huggingface/"


#CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=4 -m scripts.base_train_qwen3 -- \
#    --depth=24 \
#    --run=qwen3 \
#    --model-tag=qwen3-l24 \
#    --device-batch-size=16 \
#    --sample-every=-1 \
#    --save-every=-1 \
#    --lr=3e-3 \
#    --warmup-ratio=0.05 \
#    --core-metric-max-per-task=-1 \
#    --core-metric-every=4000 \
#    --target-param-data-ratio=12 \
#    2>&1 | tee -a train_qwen3_new.txt

CUDA_VISIBLE_DEVICES=2,6 OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 -m scripts.base_train_qwen3 -- \
    --depth=12 \
    --run=qwen3 \
    --model-tag=qwen3-l12 \
    --device-batch-size=16 \
    --sample-every=-1 \
    --save-every=-1 \
    --warmup-ratio=0.05 \
    --core-metric-max-per-task=-1 \
    --core-metric-every=4000 \
    --target-param-data-ratio=12 \
    2>&1 | tee -a train_qwen3_new.txt

