"""
FineWeb dataset support for nanochat pretraining.

Supports loading data from HuggingFaceFW/fineweb dataset (parquet format).
Can be used standalone or mixed with the default fineweb-edu-100b-shuffle dataset.

Usage:
    # Download a sample (e.g. 10BT subset)
    python -m nanochat.fineweb --subset sample-10BT --num-files 10

    # Download a specific dump
    python -m nanochat.fineweb --subset CC-MAIN-2024-10 --num-files 20

    # Use with training (standalone):
    python -m scripts.base_train_qwen3 --data-dir ~/.cache/nanochat/fineweb_data

    # Use with training (mixed with default data):
    python -m scripts.base_train_qwen3 --data-dir ~/.cache/nanochat/base_data:~/.cache/nanochat/fineweb_data
"""

import os
import argparse
import time
import requests
from multiprocessing import Pool

from nanochat.common import get_base_dir

# The HuggingFace dataset repository
HF_REPO = "HuggingFaceFW/fineweb"
HF_BASE_URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main"

# Available subsets/configs
AVAILABLE_SUBSETS = [
    "sample-10BT",   # ~10B tokens, ~27.6GB
    "sample-100BT",  # ~100B tokens, ~277.4GB
    "sample-350BT",  # ~350B tokens, ~388GB
    # Individual dumps like CC-MAIN-2024-10, CC-MAIN-2023-50, etc.
]

base_dir = get_base_dir()
FINEWEB_DATA_DIR = os.path.join(base_dir, "fineweb_data")
os.makedirs(FINEWEB_DATA_DIR, exist_ok=True)


def get_fineweb_data_dir(subset=None):
    """Get (or create) the data directory for a specific FineWeb subset."""
    if subset is None:
        return FINEWEB_DATA_DIR
    subset_dir = os.path.join(FINEWEB_DATA_DIR, subset.replace("/", "_"))
    os.makedirs(subset_dir, exist_ok=True)
    return subset_dir


def list_remote_parquet_files(subset="sample-10BT"):
    """
    List parquet files available for a given subset using the HuggingFace API.
    Returns list of relative file paths.
    """
    # Use HuggingFace Hub API to list files in the dataset
    api_url = f"https://huggingface.co/api/datasets/{HF_REPO}/tree/main"

    # Determine the path prefix based on subset
    if subset.startswith("sample-"):
        path_prefix = f"sample/{subset.replace('sample-', '')}"
    elif subset.startswith("CC-MAIN-"):
        path_prefix = f"data/{subset}"
    else:
        path_prefix = f"data/{subset}"

    params = {"path": path_prefix, "recursive": "true"}

    try:
        response = requests.get(api_url, params=params, timeout=30)
        response.raise_for_status()
        files = response.json()
        parquet_files = [
            f["path"] for f in files
            if f["path"].endswith(".parquet") and f["type"] == "file"
        ]
        return sorted(parquet_files)
    except Exception as e:
        print(f"Warning: Could not list remote files via API: {e}")
        print("Falling back to sequential file discovery...")
        return []


def download_fineweb_file(args_tuple):
    """Download a single FineWeb parquet file. args_tuple = (remote_path, local_dir)"""
    remote_path, local_dir = args_tuple

    # Construct local filepath
    filename = os.path.basename(remote_path)
    filepath = os.path.join(local_dir, filename)

    if os.path.exists(filepath):
        print(f"Skipping {filename} (already exists)")
        return True

    url = f"{HF_BASE_URL}/{remote_path}"
    print(f"Downloading {filename} from {url}...")

    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            temp_path = filepath + ".tmp"
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            os.rename(temp_path, filepath)
            print(f"Successfully downloaded {filename}")
            return True
        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            for path in [filepath + ".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
            if attempt < max_attempts:
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {filename} after {max_attempts} attempts")
                return False
    return False


def download_with_huggingface_hub(subset="sample-10BT", num_files=-1):
    """
    Download FineWeb parquet files using huggingface_hub (preferred, supports hf_transfer).
    Falls back to direct HTTP if huggingface_hub is not available.
    """
    local_dir = get_fineweb_data_dir(subset)

    try:
        from huggingface_hub import snapshot_download

        # Determine allow_patterns
        if subset.startswith("sample-"):
            token_size = subset.replace("sample-", "")
            allow_patterns = f"sample/{token_size}/*"
        elif subset.startswith("CC-MAIN-"):
            allow_patterns = f"data/{subset}/*"
        else:
            allow_patterns = f"data/{subset}/*"

        print(f"Downloading FineWeb subset '{subset}' using huggingface_hub...")
        print(f"Pattern: {allow_patterns}")
        print(f"Target directory: {local_dir}")

        # Download to a temporary location, then move parquets to our dir
        import tempfile
        tmp_dir = tempfile.mkdtemp()
        snapshot_download(
            HF_REPO,
            repo_type="dataset",
            local_dir=tmp_dir,
            allow_patterns=allow_patterns,
        )

        # Move parquet files to our target directory
        import glob
        parquet_files = sorted(glob.glob(os.path.join(tmp_dir, "**/*.parquet"), recursive=True))
        if num_files > 0:
            parquet_files = parquet_files[:num_files]

        for pf in parquet_files:
            dest = os.path.join(local_dir, os.path.basename(pf))
            if not os.path.exists(dest):
                os.rename(pf, dest)
                print(f"Moved {os.path.basename(pf)} -> {local_dir}")

        # Cleanup
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

        print(f"Done! Files available in: {local_dir}")
        return True

    except ImportError:
        print("huggingface_hub not installed, falling back to direct HTTP download.")
        print("For faster downloads: pip install huggingface_hub[hf_transfer]")
        return False


def download_fineweb(subset="sample-10BT", num_files=-1, num_workers=4):
    """
    Download FineWeb dataset files.

    Args:
        subset: Which subset to download. Options:
            - "sample-10BT": ~10B token sample (~27.6GB)
            - "sample-100BT": ~100B token sample (~277.4GB)
            - "sample-350BT": ~350B token sample (~388GB)
            - "CC-MAIN-YYYY-WW": A specific CommonCrawl dump
        num_files: Number of parquet files to download (-1 = all)
        num_workers: Number of parallel download workers
    """
    local_dir = get_fineweb_data_dir(subset)

    # Try huggingface_hub first
    if download_with_huggingface_hub(subset, num_files):
        return local_dir

    # Fallback: list and download via HTTP
    remote_files = list_remote_parquet_files(subset)
    if not remote_files:
        print(f"No remote files found for subset '{subset}'.")
        print("Please check the subset name or install huggingface_hub:")
        print("  pip install huggingface_hub[hf_transfer]")
        return local_dir

    if num_files > 0:
        remote_files = remote_files[:num_files]

    print(f"Downloading {len(remote_files)} files for subset '{subset}'...")
    print(f"Target directory: {local_dir}")

    download_args = [(rf, local_dir) for rf in remote_files]

    with Pool(processes=num_workers) as pool:
        results = pool.map(download_fineweb_file, download_args)

    successful = sum(1 for s in results if s)
    print(f"Done! Downloaded: {successful}/{len(remote_files)} files to {local_dir}")
    return local_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download FineWeb dataset for nanochat pretraining")
    parser.add_argument("--subset", type=str, default="sample-10BT",
                        help="FineWeb subset to download: sample-10BT, sample-100BT, sample-350BT, or CC-MAIN-YYYY-WW")
    parser.add_argument("-n", "--num-files", type=int, default=-1,
                        help="Number of parquet files to download (-1 = all)")
    parser.add_argument("-w", "--num-workers", type=int, default=4,
                        help="Number of parallel download workers (default: 4)")
    args = parser.parse_args()

    print(f"FineWeb Dataset Downloader")
    print(f"=" * 50)
    print(f"Subset: {args.subset}")
    print(f"Num files: {'all' if args.num_files == -1 else args.num_files}")
    print(f"Workers: {args.num_workers}")
    print()

    data_dir = download_fineweb(
        subset=args.subset,
        num_files=args.num_files,
        num_workers=args.num_workers,
    )
    print(f"\nData directory: {data_dir}")
    print(f"Use with training: python -m scripts.base_train_qwen3 --data-dir {data_dir}")

