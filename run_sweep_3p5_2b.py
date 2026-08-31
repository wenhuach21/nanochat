"""Hyperparameter sweep for the ~2B Qwen3.5 (text-only) LLM.

Python port modeled on run_sweep_3p5.py, but the FIXED_ARGS reproduce the
architecture of the official Qwen3.5 2B text config, namely:

    hidden_size            = 2048
    head_dim               = 256   -> num_attention_heads = 8
    num_key_value_heads    = 2     (GQA)
    intermediate_size      = 6144  (= hidden_size * 3, set automatically by the trainer)
    num_hidden_layers      = 24    (swept via `depth`)
    full_attention_interval= 4     (layer_types = [L,L,L,F] repeated)
    GatedDeltaNet (linear attention):
        linear_key_head_dim   = 128
        linear_value_head_dim = 128
        linear_num_key_heads  = 16
        linear_num_value_heads= 16
        linear_conv_kernel_dim= 4
    rope:
        rope_theta            = 1e7
        partial_rotary_factor = 0.25

NOTE (per request): mtp_* and tie_word_embeddings from the reference config are
intentionally NOT reproduced here. vocab_size follows the training tokenizer.

Run from the repository root, e.g.:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python run_sweep_3p5_2b.py

For every combination in the cartesian product below it launches one training run.
Runs that already trained all the way to the final step are skipped (based on the
per-run training log), so the sweep is safe to re-run / resume.
"""

import os
import re
import sys
import csv
import time
import subprocess
import traceback
from datetime import datetime, timedelta, timezone
from itertools import product
from pathlib import Path

LABEL = "qwen3p5-2b-sweep"

# ----------------------------------------------------------------------------
# Sweep space (edit these arrays). The cartesian product is trained one by one.
SWEEP_SPACE = {
    "depth": [24],
    "lr": [3e-3],
    "weight-decay": [0.2],
    "warmup-ratio": [0.0],
    "target-param-data-ratio": [12],
}

# Fixed knobs shared by every run in the sweep. These pin the 2B architecture.
FIXED_ARGS = {
    "hidden-size": 2048,
    "head-dim": 256,
    "num-kv-heads": 2,
    "full-attention-interval": 4,
    # GatedDeltaNet linear-attention geometry (decoupled from softmax attention)
    "linear-conv-kernel-dim": 4,
    "linear-key-head-dim": 128,
    "linear-value-head-dim": 128,
    "linear-num-key-heads": 16,
    "linear-num-value-heads": 16,
    # rope
    "rope-theta": 10000000,
    "partial-rotary-factor": 0.25,
    # training
    "max-seq-len": 2048,
    "device-batch-size": 4,
    "total-batch-size": 524288,
    "tokenizer-backend": "transformers",
}

TOKENS_PER_ITER = 524288

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")


def get_nproc_per_node():
    visible_devices = str(os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    devices = [d.strip() for d in visible_devices.split(",") if d.strip()]
    return max(1, len(devices))


NPROC_PER_NODE = get_nproc_per_node()

base_dir = os.environ.get("NANOCHAT_BASE_DIR", str(Path.home() / ".cache" / "nanochat"))
results_dir = Path(base_dir) / f"hparam_sweep_{LABEL}"
results_dir.mkdir(parents=True, exist_ok=True)

results_file = results_dir / "results.csv"
SWEEP_PARAM_COLUMNS = [k.replace("-", "_") for k in SWEEP_SPACE.keys()]


# -------------------------
# utils
# -------------------------
def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


def value_for_tag(v):
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float):
        return f"{v:g}".replace(".", "p")
    return str(v)


def sanitize(v):
    return re.sub(r"[^A-Za-z0-9]", "_", str(v).replace(".", "p"))


def build_tag(sweep_values):
    return (
        f"sweep_{LABEL}"
        f"_d{sanitize(sweep_values['depth'])}"
        f"_lr{sanitize(sweep_values['lr'])}"
        f"_wd{sanitize(sweep_values['weight-decay'])}"
        f"_wu{sanitize(sweep_values['warmup-ratio'])}"
        f"_r{sanitize(sweep_values['target-param-data-ratio'])}"
    )


def extract_last_step_progress(text):
    """Parse the last 'step X/Y' training log line -> (last_step, total_iterations)."""
    matches = re.findall(r"step\s+(\d+)/(\d+)", text)
    if not matches:
        return None, None
    last_step, total = matches[-1]
    return int(last_step), int(total)


def run_completed(sweep_values):
    """A run counts as done ONLY if training fully reached the final step."""
    log_file = results_dir / f"{build_tag(sweep_values)}.log"
    if not log_file.exists():
        return False
    text = log_file.read_text(errors="ignore")
    last_step, total = extract_last_step_progress(text)
    if last_step is None or not total:
        return False
    # training logs step in [0, total-1]; reaching total-1 means the final step trained
    return last_step + 1 >= total


def grep_last(pattern, text):
    matches = re.findall(pattern, text)
    return matches[-1] if matches else ""


def extract_int(pattern, text):
    val = grep_last(pattern, text)
    if not val:
        return ""
    val = val.replace(",", "")
    try:
        return int(val)
    except Exception:
        return ""


def extract_float(pattern, text):
    val = grep_last(pattern, text)
    try:
        return float(val)
    except Exception:
        return ""


def extract_named_int(name, text):
    return extract_int(rf"{name}:\s*([\d,]+)", text) or 0


def tail_lines(text, n=40):
    return "\n".join(text.splitlines()[-n:])


def get_git_commit_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def beijing_time_str():
    bj_tz = timezone(timedelta(hours=8))
    return datetime.now(bj_tz).strftime("%Y-%m-%d %H:%M:%S")


def extract_all_task_metrics(text):
    """Extract all completed eval blocks from the log."""
    task_pattern = re.compile(
        r"Evaluating:\s+([^()]+)\s+\([^)]+\).*?accuracy:\s+([-+]?\d*\.?\d+)\s+\|\s+centered:\s+([-+]?\d*\.?\d+)"
    )
    core_pattern = re.compile(r"Step\s+(\d+)\s+\|\s+CORE metric:\s*([-+]?\d*\.?\d+)")

    all_eval_blocks = []
    current_tasks = {}

    for line in text.splitlines():
        task_match = task_pattern.search(line)
        if task_match:
            current_tasks[task_match.group(1).strip()] = {
                "accuracy": float(task_match.group(2)),
                "centered": float(task_match.group(3)),
            }
            continue

        core_match = core_pattern.search(line)
        if core_match and current_tasks:
            all_eval_blocks.append({
                "eval_step": int(core_match.group(1)),
                "core_metric": float(core_match.group(2)),
                "tasks": current_tasks.copy(),
            })
            current_tasks = {}

    return all_eval_blocks


# -------------------------
# init csv
# -------------------------
header = [
    "run_datetime_bj",
    "git_commit",
    *SWEEP_PARAM_COLUMNS,
    "params_total",
    "num_iterations",
    "tokens_trained",
    "final_loss",
    "val_bpb",
    "core_score",
    "train_time_sec",
]


def main():
    if not results_file.exists():
        with open(results_file, "w", newline="") as f:
            csv.writer(f).writerow(header)

    sweep_keys = list(SWEEP_SPACE.keys())
    sweep_combos = product(*(SWEEP_SPACE[k] for k in sweep_keys))

    log("=" * 50)
    log(f"Qwen3.5 2B sweep start (GPUs={os.environ.get('CUDA_VISIBLE_DEVICES')}, nproc={NPROC_PER_NODE})")
    log(f"Sweep space: {SWEEP_SPACE}")
    log("=" * 50)

    git_commit = get_git_commit_hash()
    log(f"Git commit: {git_commit}")

    for combo in sweep_combos:
        sweep_values = dict(zip(sweep_keys, combo))
        run_datetime_bj = beijing_time_str()

        if run_completed(sweep_values):
            log(f"SKIP {sweep_values} (already fully trained to final step)")
            continue

        tag = build_tag(sweep_values)
        start_time = time.time()

        cmd = [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={NPROC_PER_NODE}",
            "-m", "nanochat.qwen35.base_train_qwen3p5",
            "--",
            f"--run={tag}",
            f"--model-tag={tag}",
            *(f"--{k}={v}" for k, v in FIXED_ARGS.items()),
            *(f"--{k}={v}" for k, v in sweep_values.items()),
            "--core-metric-every=1000",
            "--core-metric-max-per-task=-1",
            "--sample-every=-1",
            "--save-every=5000",
        ]

        log_file = results_dir / f"{tag}.log"
        log(f"TRAIN {tag}")

        with open(log_file, "w", encoding="utf-8") as lf:
            lf.write("# cmd: " + " ".join(cmd) + "\n")
            lf.flush()
            try:
                popen = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert popen.stdout is not None
                for line in popen.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    lf.write(line)
                popen.stdout.close()
                returncode = popen.wait()
            except Exception as e:
                lf.write("\n# launcher_exception\n")
                lf.write(traceback.format_exc())
                lf.flush()
                log(f"ERROR: failed to launch training for {sweep_values}: {e}")
                log(f"See log for traceback: {log_file}")
                continue

        train_time = int(time.time() - start_time)
        text = log_file.read_text(errors="ignore")

        all_task_metrics = extract_all_task_metrics(text)

        params_total = extract_named_int("total", text)
        num_iters = extract_int(r"Calculated number of iterations.*: ([\d,]+)", text) or 0
        tokens_trained = num_iters * TOKENS_PER_ITER
        final_loss = extract_float(r"step\s+\d+/\d+\s+\([^)]+\)\s+\|\s+loss:\s+([\d.]+)", text) or 0.0
        val_bpb = extract_float(r"Validation bpb:\s*([\d.]+)", text)
        core_score = extract_float(r"CORE metric:\s*([-+]?\d*\.?\d+)", text) or 0.0

        if returncode != 0:
            log(f"WARNING: training process exited with code {returncode} for {sweep_values}")
            if text:
                log("Last log lines:")
                print(tail_lines(text, 60))
            log(f"Full log path: {log_file}")

        log(f"Params: {params_total}, iters: {num_iters}, bpb: {val_bpb}, core: {core_score}")

        with open(results_file, "a", newline="") as f:
            csv.writer(f).writerow([
                run_datetime_bj,
                git_commit,
                *(sweep_values[k] for k in sweep_keys),
                params_total, num_iters, tokens_trained,
                final_loss, val_bpb, core_score, train_time,
            ])

        if all_task_metrics:
            tasks_csv_file = results_dir / f"{tag}_tasks.csv"
            with open(tasks_csv_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["eval_index", "eval_step", "core_metric", "task_name", "accuracy", "centered"])
                for eval_index, eval_block in enumerate(all_task_metrics, start=1):
                    for task_name, metrics in eval_block["tasks"].items():
                        writer.writerow([
                            eval_index,
                            eval_block["eval_step"],
                            eval_block["core_metric"],
                            task_name,
                            metrics["accuracy"],
                            metrics["centered"],
                        ])
            log(f"Saved {len(all_task_metrics)} eval rounds to CSV: {tasks_csv_file}")

    log("=" * 50)
    log(f"Qwen3.5 2B sweep complete. Results in: {results_dir}")
    log("=" * 50)

    print()
    print(results_file.read_text())


if __name__ == "__main__":
    main()

