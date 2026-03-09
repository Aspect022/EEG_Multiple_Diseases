#!/usr/bin/env python3
"""
Precompute CWT Scalograms — Memory-Mapped (Zero RAM Pressure).

Uses numpy memmap to write each sample DIRECTLY to a file on disk.
Peak RAM = raw dataset (~12 GB) + 1 sample (negligible).
Never loads/concatenates the full output tensor.

During training, CachedScalogramDataset opens the memmap in read-only mode
and the OS page cache serves data on demand — only active pages are in RAM.

Usage:
    python precompute_scalograms.py --data-dir data/ds005555
"""

import os
import sys
import gc
import time
import json
import argparse
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.data.boas_dataset import BOASDataset
from src.data.transforms import create_scalogram_transform


def precompute_split(
    data_dir: str,
    split: str,
    transform,
    cache_dir: Path,
    max_subjects: int = None,
):
    """Precompute scalograms for one split using memory-mapped files."""
    print(f"\n{'='*60}")
    print(f"  Precomputing: {split.upper()}")
    print(f"{'='*60}")

    # ── Load raw dataset ──
    dataset = BOASDataset(
        data_dir=data_dir,
        split=split,
        transform=None,
        max_subjects=max_subjects,
    )

    n = len(dataset)
    if n == 0:
        print(f"  [WARN] No epochs for split '{split}', skipping.")
        return 0

    print(f"  Samples: {n}")
    print(f"  Class dist: {dataset.get_class_distribution()}")

    # Get shape from first sample
    raw_signal, _ = dataset[0]
    sample_out = transform(raw_signal)
    C, H, W = sample_out.shape
    print(f"  Scalogram shape: ({C}, {H}, {W})")
    size_gb = n * C * H * W * 2 / 1e9
    print(f"  Output file size: {size_gb:.2f} GB (float16)")

    # ── Create memory-mapped file ──
    # This allocates space on DISK, not in RAM!
    data_path = cache_dir / f"{split}_data.npy"
    label_path = cache_dir / f"{split}_labels.npy"

    print(f"  Creating memmap file: {data_path}")
    data_mmap = np.memmap(
        str(data_path), dtype='float16', mode='w+', shape=(n, C, H, W)
    )
    labels = np.zeros(n, dtype='int64')

    # ── Write samples one by one ──
    # Peak RAM: raw dataset + 1 sample transform = ~12 GB total
    t0 = time.time()
    flush_every = 200  # Flush to disk periodically

    for i in tqdm(range(n), desc=f"  [{split}]", ncols=80):
        raw_signal, label = dataset[i]
        scalogram = transform(raw_signal)
        data_mmap[i] = scalogram.numpy().astype('float16')
        labels[i] = label

        if (i + 1) % flush_every == 0:
            data_mmap.flush()

    # Final flush
    data_mmap.flush()

    elapsed = time.time() - t0
    rate = n / elapsed if elapsed > 0 else 0
    print(f"  Done: {elapsed:.1f}s ({rate:.0f} samples/sec)")

    # Save labels (small — just N int64 values)
    np.save(str(label_path), labels)
    print(f"  Saved: {data_path} ({size_gb:.2f} GB)")
    print(f"  Saved: {label_path}")

    # Save shape metadata for this split
    meta = {'n': n, 'C': C, 'H': H, 'W': W, 'dtype': 'float16'}
    with open(cache_dir / f"{split}_meta.json", 'w') as f:
        json.dump(meta, f)

    # ── Free everything ──
    del data_mmap, labels, dataset
    gc.collect()

    return n


def main():
    parser = argparse.ArgumentParser(description="Precompute CWT scalograms")
    parser.add_argument('--data-dir', type=str, default='data/ds005555')
    parser.add_argument('--cache-dir', type=str, default=None)
    parser.add_argument('--max-subjects', type=int, default=None)
    parser.add_argument('--output-size', type=int, default=224)
    args = parser.parse_args()

    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
    else:
        cache_dir = Path(args.data_dir).parent / 'ds005555_cache'

    cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  SCALOGRAM PRECOMPUTATION (Memory-Mapped)")
    print("=" * 60)
    print(f"  Data dir:    {args.data_dir}")
    print(f"  Cache dir:   {cache_dir}")
    print(f"  Output size: {args.output_size}x{args.output_size}")
    print(f"  Strategy:    numpy memmap (writes to disk, ~0 RAM)")
    print(f"  Max subjects: {args.max_subjects or 'all'}")
    print("=" * 60)

    transform = create_scalogram_transform(
        output_size=(args.output_size, args.output_size),
        sampling_rate=100,
    )

    total_time = time.time()
    total_samples = 0

    for split in ['train', 'val', 'test']:
        n = precompute_split(
            data_dir=args.data_dir,
            split=split,
            transform=transform,
            cache_dir=cache_dir,
            max_subjects=args.max_subjects,
        )
        total_samples += n
        gc.collect()

    elapsed = time.time() - total_time
    print(f"\n{'='*60}")
    print(f"  PRECOMPUTATION COMPLETE")
    print(f"  Total samples: {total_samples}")
    print(f"  Total time:    {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Cache dir:     {cache_dir}")
    print(f"{'='*60}")

    # Global metadata
    meta = {
        'total_samples': total_samples,
        'output_size': args.output_size,
        'dtype': 'float16',
        'format': 'numpy_memmap',
        'precompute_time_seconds': round(elapsed, 1),
        'data_dir': str(args.data_dir),
    }
    with open(cache_dir / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)


if __name__ == '__main__':
    main()
