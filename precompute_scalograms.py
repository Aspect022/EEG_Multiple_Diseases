#!/usr/bin/env python3
"""
Precompute CWT Scalograms for BOAS Dataset.

Converts all raw EEG epochs → 2D scalogram tensors and saves as .pt files.
This eliminates the ~15ms/sample CWT cost during training.

Usage:
    python precompute_scalograms.py --data-dir data/ds005555
    python precompute_scalograms.py --data-dir data/ds005555 --max-subjects 10  # quick test
"""

import os
import sys
import time
import argparse
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

# Add project root to path
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
    """Precompute scalograms for one split and save to disk."""
    print(f"\n{'='*60}")
    print(f"  Precomputing: {split.upper()}")
    print(f"{'='*60}")

    # Load raw dataset (no transform — we'll apply it manually with progress bar)
    dataset = BOASDataset(
        data_dir=data_dir,
        split=split,
        transform=None,  # raw signals
        max_subjects=max_subjects,
    )

    n = len(dataset)
    if n == 0:
        print(f"  [WARN] No epochs for split '{split}', skipping.")
        return

    print(f"  Samples: {n}")
    print(f"  Class dist: {dataset.get_class_distribution()}")

    # Preallocate tensors (float16 to save memory)
    # First sample to get shape
    raw_signal, _ = dataset[0]
    sample_scalogram = transform(raw_signal)
    shape = sample_scalogram.shape  # (3, 224, 224)

    print(f"  Scalogram shape: {shape}")
    print(f"  Storage: {n * np.prod(shape) * 2 / 1e9:.2f} GB (float16)")

    data_tensor = torch.zeros((n, *shape), dtype=torch.float16)
    label_tensor = torch.zeros(n, dtype=torch.long)

    # Transform all samples with progress bar
    t0 = time.time()
    for i in tqdm(range(n), desc=f"  [{split}] CWT", ncols=80):
        raw_signal, label = dataset[i]
        scalogram = transform(raw_signal)
        data_tensor[i] = scalogram.half()  # float32 → float16
        label_tensor[i] = label

    elapsed = time.time() - t0
    rate = n / elapsed
    print(f"  Done: {elapsed:.1f}s ({rate:.0f} samples/sec)")

    # Save to disk
    data_path = cache_dir / f"{split}_data.pt"
    label_path = cache_dir / f"{split}_labels.pt"

    torch.save(data_tensor, data_path)
    torch.save(label_tensor, label_path)

    file_size_gb = data_path.stat().st_size / 1e9
    print(f"  Saved: {data_path} ({file_size_gb:.2f} GB)")
    print(f"  Saved: {label_path}")

    return n


def main():
    parser = argparse.ArgumentParser(description="Precompute CWT scalograms")
    parser.add_argument('--data-dir', type=str, default='data/ds005555',
                        help='Path to BOAS ds005555 root')
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='Output cache directory (default: data/ds005555_cache)')
    parser.add_argument('--max-subjects', type=int, default=None,
                        help='Limit subjects per split (for testing)')
    parser.add_argument('--output-size', type=int, default=224,
                        help='Scalogram output size (default: 224)')
    args = parser.parse_args()

    # Cache directory
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
    else:
        cache_dir = Path(args.data_dir).parent / 'ds005555_cache'

    cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  SCALOGRAM PRECOMPUTATION")
    print("=" * 60)
    print(f"  Data dir:    {args.data_dir}")
    print(f"  Cache dir:   {cache_dir}")
    print(f"  Output size: {args.output_size}×{args.output_size}")
    print(f"  Max subjects: {args.max_subjects or 'all'}")
    print("=" * 60)

    # Create transform
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
        if n:
            total_samples += n

    elapsed = time.time() - total_time
    print(f"\n{'='*60}")
    print(f"  PRECOMPUTATION COMPLETE")
    print(f"  Total samples: {total_samples}")
    print(f"  Total time:    {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Cache dir:     {cache_dir}")
    print(f"{'='*60}")

    # Write metadata
    import json
    meta = {
        'total_samples': total_samples,
        'output_size': args.output_size,
        'dtype': 'float16',
        'precompute_time_seconds': elapsed,
        'data_dir': str(args.data_dir),
    }
    with open(cache_dir / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)


if __name__ == '__main__':
    main()
