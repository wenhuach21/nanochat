import os
import time
import csv
import subprocess
from pathlib import Path

LABEL = "jan26"

FLOPS_BUDGETS = [
    1e18,
    2.15e18,
    4.64e18,
    1e19,
]

DEPTHS = [8, 10, 12, 14, 16, 18, 20]

NPROC_PER_NODE = int(os.environ.get("NPROC_PER_NODE", 8))
WANDB_RUN = os.environ.get("WANDB_RUN", f"scaling_{LABEL}")
EVAL_TOKENS = 100 * 524288  # ~100M tokens

os.environ["OMP_NUM_THREADS"] = "1"

base_dir = os.environ.get("NANOCHAT_BASE_DIR", str(Path.home() / ".cache" / "nanochat"))
results_dir = Path(base_dir) / f"scaling_laws_results_{LABEL}"
results_dir.mkdir(parents=True, exist_ok=True)

results_file = results_dir / "results.csv"


# -------------------------
# utils
# -------------------------
def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


def run_exists(flops, depth):
    if not results_file.exists():
        return False
    prefix = f"{flops},{depth},"
    with open(results_file) as f:
        return any(line.startswith(prefix) for line in f)


def grep_last(pattern, text):
    import re
    matches = re.findall(pattern, text)
    return matches[-1] if matches else ""


def extract_int(pattern, text):
    val = grep_last(pattern, text)
    if not val:
        return ""
    val = val.replace(",", "")
    try:
        return int(val)
    except:
        return ""


def extract_float(pattern, text):
    val = grep_last(pattern, text)
    try:
        return float(val)
    except:
        return ""


# -------------------------
# init csv
# -------------------------
header = [
    "flops_budget", "depth", "model_dim",
    "params_wte", "params_bigram_embed", "params_value_embeds",
    "params_lm_head", "params_transformer", "params_scalars",
    "params_total", "num_iterations", "tokens_trained",
    "val_bpb", "core_score", "train_time_sec"
]

if not results_file.exists():
    with open(results_file, "w", newline="") as f:
        csv.writer(f).writerow(header)


# -------------------------
# main loop
# -------------------------
for flops in FLOPS_BUDGETS:
    log("=" * 50)
    log(f"Compute budget: {flops} FLOPs")
    log("=" * 50)

    for d in DEPTHS:

        if run_exists(flops, d):
            log(f"Skipping d={d} at {flops} (already exists)")
            continue

        tag = f"scaling_{flops}_d{d}"
        start_time = time.time()

        cmd = [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={NPROC_PER_NODE}",
            "-m", "scripts.base_train",
            "--",
            f"--depth={d}",
            f"--target-flops={flops}",
            "--target-param-data-ratio=-1",
            f"--run={WANDB_RUN}_{tag}",
            f"--model-tag={tag}",
            f"--eval-tokens={EVAL_TOKENS}",
            "--core-metric-every=999999",
            "--core-metric-max-per-task=-1",
            "--sample-every=-1",
            "--save-every=-1",
        ]

        log_file = results_dir / f"{tag}_train.log"

        log(f"Training d={d} at {flops} FLOPs...")

        with open(log_file, "w") as lf:
            subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)

        train_time = int(time.time() - start_time)
        text = log_file.read_text(errors="ignore")

        # -------------------------
        # parse logs
        # -------------------------
        params_wte = extract_int(r"wte: ([\d,]+)", text)
        params_bigram = extract_int(r"bigram_embed: ([\d,]+)", text)
        params_ve = extract_int(r"value_embeds: ([\d,]+)", text)
        params_lm = extract_int(r"lm_head: ([\d,]+)", text)
        params_transformer = extract_int(r"transformer_matrices: ([\d,]+)", text)
        params_scalars = extract_int(r"scalars: ([\d,]+)", text)
        params_total = extract_int(r"total: ([\d,]+)", text)

        num_iters = extract_int(r"Calculated number of iterations.*: ([\d,]+)", text)
        num_iters = num_iters or 0

        tokens_trained = num_iters * 524288
        model_dim = d * 64

        val_bpb = extract_float(r"Validation bpb:\s*([\d.]+)", text)
        core_score = extract_float(r"CORE metric:\s*([\d.]+)", text) or 0.0

        log(
            f"Params: {params_total}, iters: {num_iters}, "
            f"bpb: {val_bpb}, core: {core_score}"
        )

        # -------------------------
        # write csv
        # -------------------------
        with open(results_file, "a", newline="") as f:
            csv.writer(f).writerow([
                flops, d, model_dim,
                params_wte, params_bigram, params_ve,
                params_lm, params_transformer, params_scalars,
                params_total, num_iters, tokens_trained,
                val_bpb, core_score, train_time
            ])


log("=" * 50)
log("Scaling Laws Sweep Complete")
log("=" * 50)

log(f"Results saved to: {results_file}")

# print table (simple version)
print()
print(results_file.read_text())