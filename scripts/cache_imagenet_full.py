#!/usr/bin/env python3
"""Download and cache the full ImageNet-1k dataset (train + validation).

Saves a DatasetDict with both 'train' (~1.28M images) and 'validation' (50k)
splits to a local Arrow cache. Train script (train_phase2.py) auto-detects
the DatasetDict format and uses the splits directly — no random 80/20 needed.

IMPORTANT: do NOT cache to the project home directory on the cluster.
The home filesystem is shared/slow and ImageNet is ~150GB. Stage to fast
local NVMe at /mnt/local_learning/data/<username>/imagenet_full instead.

Usage on Neptun:
    python3 scripts/cache_imagenet_full.py \
        --output /mnt/local_learning/data/$USER/imagenet_full

The HuggingFace ILSVRC/imagenet-1k repo is gated — make sure you have
accepted the dataset license and run `huggingface-cli login` first.
"""

import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Cache full ImageNet-1k to disk.")
    parser.add_argument(
        '--output', type=str,
        default=f"/mnt/local_learning/data/{os.environ.get('USER', 'user')}/imagenet_full",
        help="Output directory for the cached DatasetDict (default: /mnt/local_learning/data/$USER/imagenet_full)",
    )
    parser.add_argument(
        '--hf-cache-dir', type=str, default=None,
        help="HuggingFace download cache dir (default: HF default ~/.cache/huggingface). "
             "Set this to a path on local NVMe to avoid filling the home directory.",
    )
    parser.add_argument(
        '--num-proc', type=int, default=8,
        help="Parallel download workers (default: 8).",
    )
    args = parser.parse_args()

    target = Path(args.output)
    if target.exists():
        print(f"Cache already exists at {target}")
        print("Delete it first if you want to re-download.")
        return

    target.parent.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset

    load_kwargs = {}
    if args.hf_cache_dir:
        load_kwargs['cache_dir'] = args.hf_cache_dir

    print("Downloading ImageNet-1k (train + validation)...")
    print("This will take a while (~150GB total). Make sure you have:")
    print("  1. Accepted the dataset license on huggingface.co/datasets/ILSVRC/imagenet-1k")
    print("  2. Run `huggingface-cli login`")
    print()

    ds = load_dataset("ILSVRC/imagenet-1k", num_proc=args.num_proc, **load_kwargs)

    print(f"\nDownloaded splits: {list(ds.keys())}")
    for split, data in ds.items():
        print(f"  {split}: {len(data):,} images")

    print(f"\nSaving DatasetDict to {target}...")
    ds.save_to_disk(str(target))
    print(f"Done. Use --data-dir {target} when launching train_phase2.py")


if __name__ == "__main__":
    main()
