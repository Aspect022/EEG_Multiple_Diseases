#!/usr/bin/env python3
"""
Paper-focused BOAS sleep-staging pipeline.

Runs only the important models for paper comparison:
1. TCANet (NEW - runs FIRST, fast ~10 min/epoch)
2. Core baselines (SNN, 1D SNN, Transformers)
3. Quantum baselines (best performers)
4. Fusion models (best performers)
5. fusion_c (LAST - slow 1hr/epoch)

Skips: QIF variants, broken 1D ViT, failed quantum-spiking fusion
"""

import argparse
from pathlib import Path

import torch

from pipeline import EXPERIMENT_DEFS, generate_summary, run_experiment, verify_dataset


MODEL_PRESETS = {
    # Minimal: TCANet + core baselines + fusion_c (6 models, ~6 hours)
    'minimal': [
        'tcanet',  # NEW - runs first
        'snn_lif_resnet',
        'snn_1d_attn',
        'convnext',
        'quantum_ring_RXY',
        'fusion_c',  # LAST - slow
    ],
    # Core: TCANet + all important baselines + fusion models (10 models, ~10 hours)
    'core': [
        'tcanet',  # NEW - runs first
        'snn_lif_resnet',
        'snn_lif_vit',
        'snn_1d_lif',
        'snn_1d_attn',
        'convnext',
        'quantum_ring_RXY',
        'quantum_full_RXY',
        'fusion_a',  # Already done, will skip fast
        'fusion_c',  # LAST - slow, crashed before
    ],
    # Extended: Core + transformer baselines + SNN fusion (15 models, ~15 hours)
    'extended': [
        'tcanet',  # NEW - runs first
        'snn_lif_resnet',
        'snn_lif_vit',
        'snn_1d_lif',
        'snn_1d_attn',
        'swin',
        'vit',
        'deit',
        'efficientnet',
        'convnext',
        'quantum_ring_RXY',
        'quantum_full_RXY',
        'fusion_a',  # Already done
        'snn_fusion_gated',  # NEW
        'fusion_c',  # LAST - slow, crashed before
    ],
    # Fusion ablation: All fusion models for comparison
    'fusion_ablation': [
        'snn_1d_attn',  # 1D baseline
        'snn_lif_resnet',  # 2D baseline
        'fusion_a',  # 2D fusion (already done)
        'snn_fusion_early',  # Rerun (failed before)
        'snn_fusion_late',  # Rerun (crashed)
        'snn_fusion_gated',  # NEW
        'fusion_c',  # LAST - slow, crashed before
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description='Paper-focused BOAS experiment runner')
    parser.add_argument('--preset', choices=sorted(MODEL_PRESETS), default='core')
    parser.add_argument('--no-pretrained', action='store_true',
                        help='Disable pretrained timm backbones to avoid weight downloads on offline servers')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--output-dir', type=str, default='outputs/paper_runs')
    parser.add_argument('--skip-download', action='store_true')
    parser.add_argument('--max-subjects', type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("  PAPER-FOCUSED EEG SLEEP STAGING PIPELINE")
    print("=" * 70)
    print(f"  Preset:       {args.preset}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Batch Size:   {args.batch_size}")
    print(f"  LR:           {args.lr}")
    print(f"  Data Dir:     {args.data_dir}")
    print(f"  Output Dir:   {args.output_dir}")
    print(f"  Max Subjects: {args.max_subjects or 'all'}")
    print(f"  Device:       {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 70)

    if not args.skip_download and not verify_dataset(args.data_dir):
        raise SystemExit(
            "Dataset verification failed. Prepare BOAS first or rerun with --skip-download if already present."
        )

    selected = MODEL_PRESETS[args.preset]
    experiments = {}
    for key in selected:
        exp = dict(EXPERIMENT_DEFS[key])
        if args.no_pretrained and exp['type'] in {
            'swin', 'vit', 'deit', 'efficientnet', 'convnext',
            'fusion_a', 'fusion_b', 'fusion_c',
        }:
            exp['pretrained'] = False
        experiments[key] = exp

    print("\nModels in this run:")
    for key in selected:
        print(f"  - {key}: {experiments[key]['name']}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    all_results = []
    for exp_key, exp_config in experiments.items():
        result = run_experiment(
            exp_key=exp_key,
            exp_config=exp_config,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            max_subjects=args.max_subjects,
        )
        all_results.append(result)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    generate_summary(all_results, args.output_dir)


if __name__ == '__main__':
    main()
