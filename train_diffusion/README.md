# Qwen3 Diffusion LLM (LLaDA-style)

Self-contained code to train a Qwen3 backbone as a **masked diffusion language
model**, following `LLaDA/GUIDELINES.md`. Reuses the nanochat tokenizer/dataloader
for data; the model itself is bidirectional (causal mask removed) with a reserved
`[MASK]` token at the last vocab index.

## Files
- `diffusion_model.py` — bidirectional Qwen3 + LLaDA `forward_process` masking + loss + save/load
- `train.py` — training loop (AdamW, grad-accum, DDP, sampling, checkpoint save)
- `sample_diffusion.py` — iterative confidence-based unmasking sampler
- `eval_diffusion.py` — CORE-metric evaluation (diffusion-style masked-span scoring)

## Usage
```bash
# from project root
python -m train_diffusion.train --depth 14

# tiny CPU smoke test
python -m train_diffusion.train --depth 4 --max-seq-len 512 \
    --device-batch-size 1 --total-batch-size 512 --num-iterations 20

# multi-GPU
torchrun --nproc_per_node=8 -m train_diffusion.train --depth 14
```

## Eval (CORE metric)
Training saves a checkpoint to `<base_dir>/diffusion_checkpoints/<tag>/model.pt`.
Scoring follows official LLaDA (`get_log_likelihood.py` / `eval_llada.py`):
- **MC / schema**: pick the option with highest **Monte-Carlo log-likelihood** —
  `model.eval_loglikelihood` masks `k~U[1,len]` answer tokens, weights CE by `1/p_mask`,
  averages over `--mc-num` samples (use 1 for single-token MMLU, 128 otherwise).
- **LM tasks**: `model.suffix_greedy_match` iteratively greedy-decodes the span and
  checks exact match (`--gen-steps` rounds; 0 = one token/round).
```bash
python -m train_diffusion.eval_diffusion --model-tag diff_d14 --mc-num 128
# quick approx on a single GPU
python -m train_diffusion.eval_diffusion --model-tag diff_d14 --max-per-task 100 --mc-num 16
# or point at an explicit checkpoint
python -m train_diffusion.eval_diffusion --ckpt /path/to/model.pt
# distributed
torchrun --nproc_per_node=8 -m train_diffusion.eval_diffusion --model-tag diff_d14
```

## Objective
Each token is masked with prob `p ~ U(eps, 1)`; cross-entropy on masked tokens is
divided by `p_mask` and summed over the batch (per LLaDA). Sampling starts from an
all-mask block and unmasks the most confident predictions over N steps.

