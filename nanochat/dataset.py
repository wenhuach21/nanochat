"""
The base/pretraining dataset is a set of parquet files.
This file contains utilities for:
- iterating over the parquet files and yielding documents from it
- download the files on demand if they are not on disk

For details of how the dataset was prepared, see `repackage_data_reference.py`.
"""

import os
import argparse
import time
import requests
import pyarrow.parquet as pq
from multiprocessing import Pool

from nanochat.common import get_base_dir

# -----------------------------------------------------------------------------
# The specifics of the current pretraining dataset

# The URL on the internet where the data is hosted and downloaded from on demand
BASE_URL = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
MAX_SHARD = 1822 # the last datashard is shard_01822.parquet
index_to_filename = lambda index: f"shard_{index:05d}.parquet" # format of the filenames
base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "base_data")
os.makedirs(DATA_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# These functions are useful utilities to other modules, can/should be imported

def list_parquet_files(data_dir=None):
    """
    Looks into a data dir (or multiple dirs) and returns full paths to all parquet files.
    Supports colon-separated multiple directories for mixing datasets, e.g.:
        data_dir = "/path/to/base_data:/path/to/fineweb_data"
    """
    data_dir = DATA_DIR if data_dir is None else data_dir

    # Support multiple directories separated by colon (or semicolon on Windows)
    separator = ";" if os.name == "nt" else ":"
    if separator in str(data_dir):
        dirs = [d.strip() for d in data_dir.split(separator) if d.strip()]
    else:
        dirs = [data_dir]

    parquet_paths = []
    for d in dirs:
        d = os.path.expanduser(d)
        if not os.path.isdir(d):
            print(f"Warning: data directory '{d}' does not exist, skipping.")
            continue
        files = sorted([
            f for f in os.listdir(d)
            if f.endswith('.parquet') and not f.endswith('.tmp')
        ])
        parquet_paths.extend([os.path.join(d, f) for f in files])

    return parquet_paths


def print_dataset_summary(data_dir=None, split="train"):
    """
    Print a summary of the dataset: number of files, total rows, and source directories.
    Useful for verifying that mixed datasets (e.g. fineweb + edu) are loaded correctly.
    """
    parquet_paths = list_parquet_files(data_dir)
    all_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]

    total_rows = 0
    total_row_groups = 0
    dir_stats = {}  # directory -> (num_files, num_rows)

    for filepath in all_paths:
        pf = pq.ParquetFile(filepath)
        num_rows = pf.metadata.num_rows
        total_rows += num_rows
        total_row_groups += pf.num_row_groups

        parent_dir = os.path.dirname(filepath)
        if parent_dir not in dir_stats:
            dir_stats[parent_dir] = [0, 0]
        dir_stats[parent_dir][0] += 1
        dir_stats[parent_dir][1] += num_rows

    print(f"  Dataset summary ({split}):")
    print(f"    Total files: {len(all_paths)}, Total samples: {total_rows:,}, Row groups: {total_row_groups:,}")
    if len(dir_stats) > 1:
        print(f"    Sources ({len(dir_stats)} directories):")
        for d, (nf, nr) in sorted(dir_stats.items()):
            pct = nr / total_rows * 100 if total_rows > 0 else 0
            print(f"      {os.path.basename(d)}: {nf} files, {nr:,} samples ({pct:.1f}%)")
    else:
        for d in dir_stats:
            print(f"    Source: {os.path.basename(d)}")
    print()
    return total_rows

def parquets_iter_batched(split, start=0, step=1, data_dir=None):
    """
    Iterate through the dataset, in batches of underlying row_groups for efficiency.
    - split can be "train" or "val". the last parquet file will be val.
    - start/step are useful for skipping rows in DDP. e.g. start=rank, step=world_size
    - data_dir: optional data directory (supports colon-separated multiple dirs)
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files(data_dir)
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column('text').to_pylist()
            yield texts

# -----------------------------------------------------------------------------
def download_single_file(index):
    """ Downloads a single file index, with some backoff """

    # Construct the local filepath for this file and skip if it already exists
    filename = index_to_filename(index)
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        print(f"Skipping {filepath} (already exists)")
        return True

    # Construct the remote URL for this file
    url = f"{BASE_URL}/{filename}"
    print(f"Downloading {filename}...")

    # Download with retries
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            # Write to temporary file first
            temp_path = filepath + f".tmp"
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
            # Move temp file to final location
            os.rename(temp_path, filepath)
            print(f"Successfully downloaded {filename}")
            return True

        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            # Clean up any partial files
            for path in [filepath + f".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
            # Try a few times with exponential backoff: 2^attempt seconds
            if attempt < max_attempts:
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {filename} after {max_attempts} attempts")
                return False

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download FineWeb-Edu 100BT dataset shards")
    parser.add_argument("-n", "--num-files", type=int, default=-1, help="Number of shards to download (default: -1), -1 = disable")
    parser.add_argument("-w", "--num-workers", type=int, default=4, help="Number of parallel download workers (default: 4)")
    args = parser.parse_args()

    num = MAX_SHARD + 1 if args.num_files == -1 else min(args.num_files, MAX_SHARD + 1)
    ids_to_download = list(range(num))
    print(f"Downloading {len(ids_to_download)} shards using {args.num_workers} workers...")
    print(f"Target directory: {DATA_DIR}")
    print()
    with Pool(processes=args.num_workers) as pool:
        results = pool.map(download_single_file, ids_to_download)

    # Report results
    successful = sum(1 for success in results if success)
    print(f"Done! Downloaded: {successful}/{len(ids_to_download)} shards to {DATA_DIR}")
