"""
CORE-metric evaluation for the diffusion Qwen3 model.

Standard CORE (scripts.base_eval) scores continuations with autoregressive
next-token loss, which leaks for a bidirectional diffusion model. Here we instead
score each candidate the LLaDA way: mask only the answer span and let the model
predict it (every other token stays as context). For multiple-choice/schema we
pick the option with lowest masked cross-entropy; for LM we check argmax match.

Usage (from project root):

    python -m train_diffusion.eval_diffusion --model-tag diff_d14
    python -m train_diffusion.eval_diffusion --ckpt /path/model.pt --max-per-task 100
"""

import os
import csv
import json
import time
import yaml
import random
import argparse
from contextlib import nullcontext

import torch
import torch.distributed as dist

from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, autodetect_device_type, download_file_with_lock
from nanochat.tokenizer import get_tokenizer
from nanochat.core_eval import (
    render_prompts_mc, render_prompts_schema, render_prompts_lm,
    batch_sequences_mc, batch_sequences_schema, batch_sequences_lm,
)
from scripts.base_eval import EVAL_BUNDLE_URL, place_eval_bundle
from train_diffusion.diffusion_model import load_diffusion


@torch.no_grad()
def score_candidate(model, tokens, start, end, device, mc_num):
    """Log-likelihood of the [start:end) answer span via the model's MC estimator."""
    seq = torch.tensor(tokens, dtype=torch.long, device=device)
    prompt_index = torch.arange(len(seq), device=device) < start
    return model.eval_loglikelihood(seq, prompt_index, mc_num=mc_num)


@torch.no_grad()
def evaluate_example(idx, model, tokenizer, data, device, task_meta, mc_num=128, gen_steps=0):
    item = data[idx]
    tt = task_meta["task_type"]
    nf = task_meta["num_fewshot"]
    cd = task_meta["continuation_delimiter"]
    fewshot = []
    if nf > 0:
        rng = random.Random(1234 + idx)
        fewshot = [data[i] for i in rng.sample([i for i in range(len(data)) if i != idx], nf)]
    if tt == "multiple_choice":
        prompts = render_prompts_mc(item, cd, fewshot)
        toks, s, e = batch_sequences_mc(tokenizer, prompts)
    elif tt == "schema":
        prompts = render_prompts_schema(item, cd, fewshot)
        toks, s, e = batch_sequences_schema(tokenizer, prompts)
    elif tt == "language_modeling":
        prompts = render_prompts_lm(item, cd, fewshot)
        toks, s, e = batch_sequences_lm(tokenizer, prompts)
    else:
        raise ValueError(tt)
    if tt == "language_modeling":
        # greedy iterative decode of the continuation, exact-match like LLaDA suffix check
        prefix = torch.tensor(toks[0][:s[0]], dtype=torch.long, device=device)
        target = torch.tensor(toks[0][s[0]:e[0]], dtype=torch.long, device=device)
        return model.suffix_greedy_match(prefix, target, steps=gen_steps)
    # MC/schema: pick the option with highest log-likelihood
    lls = [score_candidate(model, t, s[i], e[i], device, mc_num) for i, t in enumerate(toks)]
    return lls.index(max(lls)) == item["gold"]


def evaluate_task(model, tokenizer, data, device, task_meta, mc_num=128, gen_steps=0):
    """Evaluate one task, distributing examples across ranks (mirrors nanochat core_eval)."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    # Pre-task sync: all ranks must start together so the post-task barrier is never stale.
    if world > 1:
        dist.barrier()
    correct = torch.zeros(len(data), device=device)
    for idx in range(rank, len(data), world):
        correct[idx] = float(evaluate_example(idx, model, tokenizer, data, device, task_meta, mc_num=mc_num, gen_steps=gen_steps))
    if world > 1:
        dist.barrier()
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    return correct.mean().item()


def evaluate_core_diffusion(model, tokenizer, device, max_per_task=-1, mc_num=128, gen_steps=0):
    base_dir = get_base_dir()
    bundle = os.path.join(base_dir, "eval_bundle")
    if not os.path.exists(bundle):  # reuse base_eval's downloader
        download_file_with_lock(EVAL_BUNDLE_URL, "eval_bundle.zip", postprocess_fn=place_eval_bundle)
    # Sync all ranks after bundle download so no rank races ahead while another is still
    # fetching / extracting the bundle (avoids stale barrier mismatches on task 1).
    if dist.is_initialized():
        dist.barrier()
    tasks = yaml.safe_load(open(os.path.join(bundle, "core.yaml"), encoding="utf-8"))["icl_tasks"]
    baselines = {r["Eval Task"]: float(r["Random baseline"])
                 for r in csv.DictReader(open(os.path.join(bundle, "eval_meta_data.csv"), encoding="utf-8"))}
    results, centered = {}, {}
    for task in tasks:
        label = task["label"]
        meta = {"task_type": task["icl_task_type"], "dataset_uri": task["dataset_uri"],
                "num_fewshot": task["num_fewshot"][0], "continuation_delimiter": task.get("continuation_delimiter", " ")}
        data = [json.loads(l) for l in open(os.path.join(bundle, "eval_data", meta["dataset_uri"]), encoding="utf-8")]
        random.Random(1337).shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]
        t0 = time.time()
        acc = evaluate_task(model, tokenizer, data, device, meta, mc_num=mc_num, gen_steps=gen_steps)
        results[label] = acc
        rb = baselines[label]
        centered[label] = (acc - 0.01 * rb) / (1.0 - 0.01 * rb)
        print0(f"{label}: acc {acc:.4f} centered {centered[label]:.4f} ({time.time()-t0:.1f}s)")
    core = sum(centered.values()) / len(centered)
    return {"results": results, "centered_results": centered, "core_metric": core}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--model-tag", type=str, default=None)
    p.add_argument("--max-per-task", type=int, default=-1)
    p.add_argument("--mc-num", type=int, default=128, help="Monte-Carlo masks for likelihood (MC/schema); 1 ok for single-token MMLU")
    p.add_argument("--gen-steps", type=int, default=0, help="greedy decode rounds for LM tasks (0 = 1 token/round)")
    p.add_argument("--device-type", type=str, default="")
    args = p.parse_args()
    dt = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, rank, _, _, device = compute_init(dt)
    ctx = torch.amp.autocast(device_type=dt, dtype=torch.bfloat16) if dt == "cuda" else nullcontext()
    path = args.ckpt or os.path.join(get_base_dir(), "diffusion_checkpoints", args.model_tag, "model.pt")
    model = load_diffusion(path, device)
    tokenizer = get_tokenizer()
    with ctx:
        res = evaluate_core_diffusion(model, tokenizer, device, max_per_task=args.max_per_task, mc_num=args.mc_num, gen_steps=args.gen_steps)
    print0(f"CORE metric: {res['core_metric']:.4f}")
    compute_cleanup()


if __name__ == "__main__":
    main()




