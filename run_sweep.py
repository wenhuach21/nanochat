import os
import sys
import time
import csv
import re
import subprocess
import traceback
from datetime import datetime, timedelta, timezone
from itertools import product
from pathlib import Path

LABEL = "Qwen-0801-adamw"

SWEEP_SPACE = {
    "depth": [28],
    "lr": [3e-3],
    "weight-decay": [0.02],
    "warmup-ratio": [0.00],
    "hidden-size": [1024],
    "embedding-lr": [0.3],
    "target-param-data-ratio": [100],
    "grad-max-norm": [-1.0],
}

TOKENS_PER_ITER = 524288


def get_nproc_per_node():
    visible_devices = str(os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    devices = [d.strip() for d in visible_devices.split(",") if d.strip()]
    return max(1, len(devices))


NPROC_PER_NODE = get_nproc_per_node()
WANDB_RUN = os.environ.get("WANDB_RUN", f"sweep_{LABEL}")
EVAL_TOKENS = 100 * 524288  # ~100M tokens

os.environ["OMP_NUM_THREADS"] = "1"

base_dir = os.environ.get("NANOCHAT_BASE_DIR", str(Path.home() / ".cache" / "nanochat"))
results_dir = Path(base_dir) / f"hparam_sweep_results_{LABEL}"
results_dir.mkdir(parents=True, exist_ok=True)

results_file = results_dir / "results.csv"
SWEEP_PARAM_COLUMNS = [k.replace("-", "_") for k in SWEEP_SPACE.keys()]
BOOL_FLAG_KEYS = {"ema-eval"}


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
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float):
        return f"{v:g}".replace(".", "p")
    return str(v)


def build_cli_arg(k, v):
    if k in BOOL_FLAG_KEYS:
        return f"--{k}" if bool(v) else None
    return f"--{k}={v}"


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


def extract_all_task_metrics(text):
    """
    Extract all completed eval blocks from the log.
    Returns a list like:
    [
        {
            "eval_step": 5000,
            "core_metric": -0.05,
            "tasks": {
                "arc_easy": {"accuracy": 0.2, "centered": -0.01},
                ...
            },
        },
        ...
    ]
    """
    task_pattern = re.compile(
        r"Evaluating:\s+([^()]+)\s+\([^)]+\).*?accuracy:\s+([-+]?\d*\.?\d+)\s+\|\s+centered:\s+([-+]?\d*\.?\d+)"
    )
    core_pattern = re.compile(r"Step\s+(\d+)\s+\|\s+CORE metric:\s*([-+]?\d*\.?\d+)")

    all_eval_blocks = []
    current_tasks = {}

    for line in text.splitlines():
        task_match = task_pattern.search(line)
        if task_match:
            task_name = task_match.group(1).strip()
            accuracy = float(task_match.group(2))
            centered = float(task_match.group(3))
            current_tasks[task_name] = {"accuracy": accuracy, "centered": centered}
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


def extract_task_metrics(text):
    """
    Extract task evaluations from the last completed eval block in the log.
    This is used for summary purposes when there are multiple eval rounds.
    Returns dict: { task_name: { "accuracy": float, "centered": float }, ... }
    """
    all_eval_blocks = extract_all_task_metrics(text)
    if all_eval_blocks:
        return all_eval_blocks[-1]["tasks"]
    return {}


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

        # Reduce duplicate combinations:
        # - AdamW mode ignores muon-lr, keep only the first value
        # - EMA eval is meaningful only when ema-decay > 0
        if sweep_values.get("optimizer-mode") == "adamw":
            muon_candidates = SWEEP_SPACE.get("muon-lr", [])
            if muon_candidates and sweep_values.get("muon-lr") != muon_candidates[0]:
                continue
        if float(sweep_values.get("ema-decay", 0.0)) <= 0.0 and bool(sweep_values.get("ema-eval", False)):
            continue

        if run_exists(sweep_values):
            log(f"Skipping {sweep_values} (already exists)")
            continue

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
            *(arg for k, v in sweep_values.items() for arg in [build_cli_arg(k, v)] if arg is not None),
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
        all_task_metrics = extract_all_task_metrics(text)
        task_metrics = extract_task_metrics(text)

        params_total = extract_named_int("total", text)

        num_iters = extract_int(r"Calculated number of iterations.*: ([\d,]+)", text)
        num_iters = num_iters or 0

        tokens_trained = num_iters * TOKENS_PER_ITER

        # Extract final step loss (last occurrence)
        final_loss = extract_float(r"step\s+\d+/\d+\s+\([^)]+\)\s+\|\s+loss:\s+([\d.]+)", text) or 0.0

        val_bpb = extract_float(r"Validation bpb:\s*([\d.]+)", text)
        core_score = extract_float(r"CORE metric:\s*([-+]?\d*\.?\d+)", text) or 0.0

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
                *(sweep_values[k] for k in sweep_keys),
                params_total, num_iters, tokens_trained,
                final_loss,
                val_bpb, core_score, train_time
            ])

        # Save task metrics to a single standalone CSV file.
        # If there are multiple eval rounds, keep them all here;
        # summary metrics above still come from the last completed eval block.
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
    log("Hyperparameter Sweep Complete")
    log("=" * 50)

    log(f"Results saved to: {results_file}")

    # print table (simple version)
    print()
    print(results_file.read_text())


if __name__ == "__main__":
    main()

