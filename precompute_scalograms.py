#!/usr/bin/env python3
"""
Precompute CWT Scalograms for BOAS Dataset (Memory-Safe).

Strategy: Save each chunk to DISK immediately (never accumulate in RAM).
After all chunks are saved and raw data is freed, concatenate from disk.

Peak RAM during precompute: ~12 GB (raw dataset) + 150 MB (1 chunk) = ~12.2 GB
Peak RAM during concatenation: ~25 GB (output tensor only, raw data freed)

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


CHUNK_SIZE = 500


def precompute_split(
    data_dir: str,
    split: str,
    transform,
    cache_dir: Path,
    max_subjects: int = None,
):
    """Precompute scalograms for one split. Saves chunks to disk to avoid OOM."""
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

    # Get output shape from first sample
    raw_signal, _ = dataset[0]
    sample_out = transform(raw_signal)
    shape = sample_out.shape
    print(f"  Scalogram shape: {shape}")
    print(f"  Estimated final size: {n * np.prod(shape) * 2 / 1e9:.2f} GB (float16)")

    # ── Shard directory ──
    shard_dir = cache_dir / f"{split}_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    num_chunks = (n + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"  Processing {num_chunks} chunks (saving each to disk immediately)...")

    t0 = time.time()

    # ── Phase 1: Transform + save each chunk to disk ──
    # Peak RAM: raw_dataset (~12GB) + one chunk (~150MB) = ~12.2 GB
    for chunk_idx in range(num_chunks):
        start = chunk_idx * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, n)
        chunk_n = end - start

        chunk_data = torch.zeros((chunk_n, *shape), dtype=torch.float16)
        chunk_labels = torch.zeros(chunk_n, dtype=torch.long)

        for i in tqdm(range(start, end), desc=f"  [{split}] chunk {chunk_idx+1}/{num_chunks}", ncols=80):
            raw_signal, label = dataset[i]
            scalogram = transform(raw_signal)
            chunk_data[i - start] = scalogram.half()
            chunk_labels[i - start] = label

        # Save chunk to disk IMMEDIATELY — don't keep in RAM
        torch.save(chunk_data, shard_dir / f"data_{chunk_idx:04d}.pt")
        torch.save(chunk_labels, shard_dir / f"labels_{chunk_idx:04d}.pt")

        # Free the chunk from RAM
        del chunk_data, chunk_labels
        gc.collect()

    elapsed = time.time() - t0
    rate = n / elapsed if elapsed > 0 else 0
    print(f"  Phase 1 done: {elapsed:.1f}s ({rate:.0f} samples/sec)")

    # ── Free raw dataset to reclaim ~12 GB ──
    del dataset
    gc.collect()
    print(f"  Raw dataset freed from memory.")

    # ── Phase 2: Concatenate shards into final files ──
    # Peak RAM: ~25 GB (output tensor only, raw data is gone)
    print(f"  Phase 2: Concatenating {num_chunks} shards from disk...")

    data_chunks = []
    label_chunks = []
    for chunk_idx in range(num_chunks):
        d = torch.load(shard_dir / f"data_{chunk_idx:04d}.pt", weights_only=True)
        l = torch.load(shard_dir / f"labels_{chunk_idx:04d}.pt", weights_only=True)
        data_chunks.append(d)
        label_chunks.append(l)

    data_tensor = torch.cat(data_chunks, dim=0)
    label_tensor = torch.cat(label_chunks, dim=0)
    del data_chunks, label_chunks
    gc.collect()

    # Save final files
    data_path = cache_dir / f"{split}_data.pt"
    label_path = cache_dir / f"{split}_labels.pt"
    torch.save(data_tensor, data_path)
    torch.save(label_tensor, label_path)

    file_size_gb = data_path.stat().st_size / 1e9
    print(f"  Saved: {data_path} ({file_size_gb:.2f} GB)")
    print(f"  Saved: {label_path}")

    # Cleanup shards
    del data_tensor, label_tensor
    gc.collect()

    for f in shard_dir.iterdir():
        f.unlink()
    shard_dir.rmdir()
    print(f"  Shard files cleaned up.")

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
    print("  SCALOGRAM PRECOMPUTATION (Memory-Safe)")
    print("=" * 60)
    print(f"  Data dir:    {args.data_dir}")
    print(f"  Cache dir:   {cache_dir}")
    print(f"  Chunk size:  {CHUNK_SIZE} (saved to disk, NOT accumulated in RAM)")
    print(f"  Output size: {args.output_size}x{args.output_size}")
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

    meta = {
        'total_samples': total_samples,
        'output_size': args.output_size,
        'dtype': 'float16',
        'chunk_size': CHUNK_SIZE,
        'precompute_time_seconds': elapsed,
        'data_dir': str(args.data_dir),
    }
    with open(cache_dir / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)


if __name__ == '__main__':
    main()
