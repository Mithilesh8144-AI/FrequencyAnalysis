#!/usr/bin/env python3
"""Download and cache 100k ImageNet training images for offline use."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CACHE_PATH = PROJECT_ROOT / "data" / "imagenet_100k_cache"
SUBSET_SIZE = 100_000


def main():
    from datasets import load_dataset

    if CACHE_PATH.exists():
        print(f"Cache already exists at {CACHE_PATH}")
        return

    print(f"Downloading {SUBSET_SIZE:,} ImageNet training images via streaming...")
    print("This will download only the required samples, not the full dataset.\n")

    streaming_ds = load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True)
    subset = list(streaming_ds.take(SUBSET_SIZE))

    print(f"Downloaded {len(subset):,} images. Converting to Dataset...")

    from datasets import Dataset
    ds = Dataset.from_list(subset)
    ds.save_to_disk(str(CACHE_PATH))

    print(f"Saved to {CACHE_PATH}")


if __name__ == "__main__":
    main()
