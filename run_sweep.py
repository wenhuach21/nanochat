import os
import sys
import time
import csv
import json
import re
import subprocess
import traceback
from datetime import datetime, timedelta, timezone
from itertools import product
from pathlib import Path

LABEL = "Qwen"

SWEEP_SPACE = {
    "depth": [14],
    "lr": [3e-3,3e-4],
    "weight-decay": [0.02],
    "warmup-ratio":[0.1],
    "muon-lr": [0.02,0.002],
    "hidden-size":[1024],
    "embedding-lr":[0.3,0.03],
    "target-param-data-ratio":[12]
}

TOKENS_PER_ITER = 524288

NPROC_PER_NODE = int(len(os.environ.get("CUDA_VISIBLE_DEVICES", 1).split(",")))
WANDB_RUN = os.environ.get("WANDB_RUN", f"sweep_{LABEL}")
EVAL_TOKENS = 100 * 524288  # ~100M tokens

os.environ["OMP_NUM_THREADS"] = "1"

base_dir = os.environ.get("NANOCHAT_BASE_DIR", str(Path.home() / ".cache" / "nanochat"))
results_dir = Path(base_dir) / f"hparam_sweep_results_{LABEL}"
results_dir.mkdir(parents=True, exist_ok=True)

results_file = results_dir / "results.csv"


# -------------------------
# utils
# -------------------------
def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


def approximately_equal(csv_val, target):
    try:
        return abs(float(csv_val) - float(target)) <= max(1e-12, abs(float(target)) * 1e-9)
    except Exception:
        return str(csv_val) == str(target)


def run_exists(sweep_values):
    if not results_file.exists():
        return False

    with open(results_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue

            matched = True
            for k, v in sweep_values.items():
                col = k.replace("-", "_")
                if not approximately_equal(row.get(col, ""), v):
                    matched = False
                    break

            if matched:
                return True
    return False


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


def extract_named_int(name, text):
    return extract_int(rf"{name}:\s*([\d,]+)", text) or 0


def value_for_tag(v):
    if isinstance(v, float):
        return f"{v:g}".replace(".", "p")
    return str(v)


def tail_lines(text, n=40):
    lines = text.splitlines()
    return "\n".join(lines[-n:])


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


def extract_task_metrics(text):
    """
    Extract all task evaluations from log.
    Returns dict: { task_name: { "accuracy": float, "centered": float }, ... }
    """
    tasks = {}
    # Pattern: "Evaluating: task_name (...) ... accuracy: X.XXXX | centered: Y.YYYY"
    pattern = r"Evaluating:\s+([^\(]+)\s+\([^\)]+\).*?accuracy:\s+([\d.]+)\s+\|\s+centered:\s+([\d.]+)"
    for match in re.finditer(pattern, text):
        task_name = match.group(1).strip()
        accuracy = float(match.group(2))
        centered = float(match.group(3))
        tasks[task_name] = {"accuracy": accuracy, "centered": centered}
    return tasks


# -------------------------
# init csv
# -------------------------
header = [
    "run_datetime_bj",
    "git_commit",
    "depth",
    "lr",
    "weight_decay",
    "model_dim",
    "params_total",
    "num_iterations",
    "tokens_trained",
    "final_loss",
    "val_bpb",
    "core_score",
    "train_time_sec",
]

if not results_file.exists():
    with open(results_file, "w", newline="") as f:
        csv.writer(f).writerow(header)


# -------------------------
# main loop
# -------------------------
sweep_keys = list(SWEEP_SPACE.keys())
sweep_combos = product(*(SWEEP_SPACE[k] for k in sweep_keys))

log("=" * 50)
log(f"Hyperparameter sweep start: {SWEEP_SPACE}")
log("=" * 50)

git_commit = get_git_commit_hash()
log(f"Git commit: {git_commit}")

for combo in sweep_combos:
    sweep_values = dict(zip(sweep_keys, combo))
    d = int(sweep_values["depth"])
    run_datetime_bj = beijing_time_str()

    # if run_exists(sweep_values):
    #     log(f"Skipping {sweep_values} (already exists)")
    #     continue

    tag_suffix = "_".join(
        f"{k.replace('-', '')}{value_for_tag(v)}"
        for k, v in sweep_values.items()
    )
    tag = f"sweep_{LABEL}_{tag_suffix}"
    start_time = time.time()

    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={NPROC_PER_NODE}",
        "-m", "scripts.base_train_qwen3",
        "--",
        *(f"--{k}={v}" for k, v in sweep_values.items()),
        f"--run={WANDB_RUN}_{tag}",
        f"--model-tag={tag}",
        "--core-metric-every=999999",
        "--core-metric-max-per-task=-1",
        "--sample-every=-1",
        "--save-every=-1",
    ]

    log_file = results_dir / f"{tag}_train.log"

    log(f"Training {sweep_values}...")

    with open(log_file, "w", encoding="utf-8") as lf:
        # Save the full launch command for reproducibility/debugging.
        lf.write("# cmd: " + " ".join(cmd) + "\n")
        lf.flush()
        try:
            # Tee child output to both terminal and log file for realtime debug.
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

    class _ProcResult:
        def __init__(self, rc):
            self.returncode = rc

    proc = _ProcResult(returncode)

    train_time = int(time.time() - start_time)
    text = log_file.read_text(errors="ignore")

    # Extract task-level metrics
    task_metrics = extract_task_metrics(text)

    params_total = extract_named_int("total", text)

    num_iters = extract_int(r"Calculated number of iterations.*: ([\d,]+)", text)
    num_iters = num_iters or 0

    tokens_trained = num_iters * TOKENS_PER_ITER
    model_dim = int(sweep_values["hidden-size"])

    # Extract final step loss (last occurrence)
    final_loss = extract_float(r"step\s+\d+/\d+\s+\([^)]+\)\s+\|\s+loss:\s+([\d.]+)", text) or 0.0

    val_bpb = extract_float(r"Validation bpb:\s*([\d.]+)", text)
    core_score = extract_float(r"CORE metric:\s*([\d.]+)", text) or 0.0

    if proc.returncode != 0:
        log(f"WARNING: training process exited with code {proc.returncode} for {sweep_values}")
        if text:
            log("Last log lines:")
            print(tail_lines(text, 60))
        log(f"Full log path: {log_file}")

    log(
        f"Params: {params_total}, iters: {num_iters}, "
        f"bpb: {val_bpb}, core: {core_score}"
    )

    # -------------------------
    # write csv
    # -------------------------
    with open(results_file, "a", newline="") as f:
        csv.writer(f).writerow([
            run_datetime_bj,
            git_commit,
            d,
            sweep_values["lr"],
            sweep_values["weight-decay"],
            model_dim,
            params_total, num_iters, tokens_trained,
            final_loss,
            val_bpb, core_score, train_time
        ])

    # Save task metrics as JSON alongside CSV results
    if task_metrics:
        tasks_json_file = results_dir / f"{tag}_tasks.json"
        with open(tasks_json_file, "w") as f:
            json.dump(task_metrics, f, indent=2)
        log(f"Saved {len(task_metrics)} task metrics to: {tasks_json_file}")

        # Also save as CSV for easier analysis
        tasks_csv_file = results_dir / f"{tag}_tasks.csv"
        with open(tasks_csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["task_name", "accuracy", "centered"])
            for task_name, metrics in task_metrics.items():
                writer.writerow([
                    task_name,
                    metrics["accuracy"],
                    metrics["centered"],
                ])
        log(f"Saved {len(task_metrics)} task metrics to CSV: {tasks_csv_file}")


log("=" * 50)
log("Hyperparameter Sweep Complete")
log("=" * 50)

log(f"Results saved to: {results_file}")

# print table (simple version)
print()
print(results_file.read_text())