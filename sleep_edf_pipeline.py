#!/usr/bin/env python3
"""
Dedicated Sleep-EDF experiment pipeline — FINAL configuration.

Runs 13 finalized models on PhysioNet Sleep-EDF Expanded dataset.
All runs use fold-0 with configurable epochs, batch size, and learning rate.
Results logged to W&B project "eeg-sleep-apnea" with full metadata.

Phase 1 (Priority 0 — 1D signal models):
    snn_1d_attn, snn_1d_lif, quantum_1d_ring_RYZ,
    quantum_1d_full_RXY, tcanet, conditional_routing

Phase 2 (Priority 1 — 2D baselines & fusion):
    swin, convnext, efficientnet, vit, deit, fusion_a, snn_fusion_early

Usage:
    # Run all 13 models (fold-0):
    python sleep_edf_pipeline.py --epochs 30

    # Run a specific phase:
    python sleep_edf_pipeline.py --models phase1

    # Run specific models:
    python sleep_edf_pipeline.py --models snn_1d_attn,conditional_routing

    # Validate models without training:
    python sleep_edf_pipeline.py --validate-only
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from pipeline import EXPERIMENT_DEFS, create_model, generate_summary
from src.data.sleep_edf_dataset import (
    MIN_RECOMMENDED_RECORD_PAIRS,
    create_sleep_edf_dataloaders,
    create_sleep_edf_multimodal_dataloaders,
    verify_sleep_edf_dataset,
)
from src.data.transforms import create_scalogram_transform
from src.training.multimodal_trainer import MultiModalFoldTrainer
from src.training.research_trainer import FoldTrainer, ResearchConfig


# ─────────────────── Final Model Lists ───────────────────

PHASE1_MODELS = [
    "snn_1d_attn",           # Best 1D model (94.74%)
    "snn_1d_lif",            # SNN baseline (ablation)
    "quantum_1d_ring_RYZ",   # Best quantum variant (94.56%)
    "quantum_1d_full_RXY",   # Entanglement comparison
    "tcanet",                # Established 1D baseline
    "conditional_routing",   # Novel: SNN→Quantum conditional routing
]

PHASE2_MODELS = [
    "swin",                  # 2D transformer baseline
    "convnext",              # 2D CNN baseline
    "efficientnet",          # Lightweight 2D
    "vit",                   # ViT baseline
    "deit",                  # DeiT baseline
    "fusion_a",              # Multi-backbone fusion (Swin+ConvNeXt)
    "snn_fusion_early",      # Multi-modal SNN (1D+2D)
]

ALL_MODELS = PHASE1_MODELS + PHASE2_MODELS

# W&B configuration
WANDB_PROJECT = "eeg-sleep-apnea"
DATASET_NAME = "physionet-sleep-edf-expanded"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sleep-EDF benchmarking pipeline (Final 13 models)",
    )
    parser.add_argument("--data-dir", type=str, default="data/sleep-edf")
    parser.add_argument("--output-dir", type=str, default="outputs/sleep_edf_results")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--models", type=str, default="all",
        help=(
            "Model selector: 'all' | 'phase1' | 'phase2' | "
            "comma-separated model keys (e.g. snn_1d_attn,tcanet)"
        ),
    )
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--validate-models", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def resolve_models(selector: str) -> Dict[str, Dict[str, Any]]:
    """Resolve model selector to experiment definitions."""
    if selector == "all":
        keys = ALL_MODELS
    elif selector == "phase1":
        keys = PHASE1_MODELS
    elif selector == "phase2":
        keys = PHASE2_MODELS
    else:
        keys = [k.strip() for k in selector.split(",") if k.strip()]

    experiments = {}
    for k in keys:
        if k in EXPERIMENT_DEFS:
            experiments[k] = dict(EXPERIMENT_DEFS[k])
        else:
            print(f"  [WARN] Model '{k}' not found in EXPERIMENT_DEFS, skipping.")

    if not experiments:
        raise ValueError(f"No valid models resolved from '{selector}'")
    return experiments


def maybe_disable_pretrained(experiments: Dict[str, Dict[str, Any]]) -> None:
    """Disable pretrained weights for 2D models (for fair comparison)."""
    for exp in experiments.values():
        if exp["type"] in {
            "swin", "vit", "deit", "efficientnet", "convnext",
            "fusion_a", "fusion_b", "fusion_c",
        }:
            exp["pretrained"] = False


def build_sleep_edf_loaders(
    data_mode: str,
    data_dir: str,
    batch_size: int,
    num_workers: int,
    max_records: int | None,
):
    """Build data loaders for the specified data mode (1d/2d/both)."""
    if data_mode == "1d":
        return create_sleep_edf_dataloaders(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            transform=None,
            max_records=max_records,
        )

    transform = create_scalogram_transform(output_size=(224, 224), sampling_rate=100)
    if data_mode == "both":
        return create_sleep_edf_multimodal_dataloaders(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            scalogram_transform=transform,
            max_records=max_records,
        )

    return create_sleep_edf_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        transform=transform,
        max_records=max_records,
    )


def compute_class_weights(dataset) -> torch.Tensor | None:
    """Compute inverse-frequency class weights for imbalanced datasets."""
    labels = getattr(dataset, "labels", None)
    if labels is None or len(labels) == 0:
        return None

    unique, counts = np.unique(labels, return_counts=True)
    freq = counts / counts.sum()
    weights = 1.0 / (freq * len(unique))
    weights = weights / weights.mean()
    full_weights = np.ones(5, dtype=np.float32)
    for cls, weight in zip(unique.tolist(), weights.tolist()):
        full_weights[int(cls)] = weight
    return torch.tensor(full_weights, dtype=torch.float32)


def validate_model_forward(exp_key: str, exp_config: Dict[str, Any]) -> Tuple[str, str]:
    """Smoke-test a model's forward pass with dummy data."""
    try:
        model = create_model(exp_config, num_classes=5)
        model.eval()

        data_mode = exp_config.get("data_mode", "2d")
        with torch.no_grad():
            if data_mode == "1d":
                x = torch.randn(2, 6, 3000)
                out = model(x)
            elif data_mode == "both":
                x1d = torch.randn(2, 6, 3000)
                x2d = torch.randn(2, 3, 224, 224)
                out = model(raw_signal=x1d, scalogram=x2d)
            else:
                x = torch.randn(2, 3, 224, 224)
                out = model(x)

        if isinstance(out, (tuple, list)):
            logits = out[0]
        else:
            logits = out

        if tuple(logits.shape) != (2, 5):
            return "FAIL", f"unexpected output shape {tuple(logits.shape)}"
        return "OK", "ok"
    except ImportError as exc:
        return "MISSING_DEP", str(exc)
    except Exception as exc:
        return "FAIL", str(exc)


def validate_models(experiments: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """Validate all model forward passes."""
    print("\n[Validation] Checking model forward passes...")
    results: Dict[str, str] = {}
    for key, config in experiments.items():
        status, message = validate_model_forward(key, config)
        results[key] = f"{status}: {message}"
        print(f"  - {key:<28} {status}  {message}")
    return results


def determine_phase(exp_key: str) -> str:
    """Tag which phase a model belongs to (for W&B grouping)."""
    if exp_key in PHASE1_MODELS:
        return "phase1-1d"
    elif exp_key in PHASE2_MODELS:
        return "phase2-2d-fusion"
    return "custom"


def run_sleep_edf_experiment(
    exp_key: str,
    exp_config: Dict[str, Any],
    data_dir: str,
    output_dir: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_records: int | None,
    num_workers: int,
) -> Dict[str, Any]:
    """Train a single model on Sleep-EDF and return results."""
    exp_name = exp_config["name"]
    exp_output = Path(output_dir) / exp_key
    exp_output.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    print(f"\n{'='*70}")
    print(f"  Sleep-EDF Experiment: {exp_name}")
    print(f"  Model Key: {exp_key}")
    print(f"  Output: {exp_output}")
    print(f"{'='*70}")

    try:
        data_mode = exp_config.get("data_mode", "2d")
        train_loader, val_loader, test_loader = build_sleep_edf_loaders(
            data_mode=data_mode,
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            max_records=max_records,
        )
        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        test_size = len(test_loader.dataset)
        if train_size == 0:
            raise RuntimeError(
                "Sleep-EDF training split is empty. The dataset is likely incomplete "
                "or does not have enough matched PSG/Hypnogram pairs."
            )
        if val_size == 0:
            raise RuntimeError(
                "Sleep-EDF validation split is empty. Download more matched record pairs "
                f"(recommended minimum: {MIN_RECOMMENDED_RECORD_PAIRS}) before training."
            )
        if test_size == 0:
            print(
                "  [WARN] Sleep-EDF test split is empty. Training can continue, but "
                "final evaluation will be incomplete."
            )

        model = create_model(exp_config, num_classes=5)
        class_weights = compute_class_weights(train_loader.dataset)

        # Build W&B metadata
        wandb_extra = {
            "dataset_name": DATASET_NAME,
            "dataset_dir": data_dir,
            "model_key": exp_key,
            "model_name": exp_name,
            "model_type": exp_config.get("type", "unknown"),
            "data_mode": data_mode,
            "phase": determine_phase(exp_key),
            "train_samples": train_size,
            "val_samples": val_size,
            "test_samples": test_size,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "num_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }
        # Add model-specific config (e.g., gate_type, rotation, entanglement)
        for key_extra in ["gate_type", "quantum_rotation", "quantum_entanglement",
                          "attention", "neuron_type", "backbone", "entanglement",
                          "rotation", "confidence_threshold"]:
            if key_extra in exp_config:
                wandb_extra[key_extra] = exp_config[key_extra]

        wandb_tags = [
            DATASET_NAME,
            data_mode,
            determine_phase(exp_key),
            exp_config.get("type", "unknown"),
        ]

        config = ResearchConfig(
            experiment_name=exp_key,
            output_dir=output_dir,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            num_classes=5,
            mixed_precision=True,
            gradient_accumulation_steps=4,
            early_stopping=True,
            patience=10,
            save_best=True,
            seed=42,
            warmup_epochs=3,
            max_grad_norm=1.0,
            wandb_project=WANDB_PROJECT,
            wandb_tags=wandb_tags,
            wandb_config_extra=wandb_extra,
        )

        if data_mode == "both":
            trainer = MultiModalFoldTrainer(
                model=model,
                config=config,
                train_loader=train_loader,
                val_loader=val_loader,
                fold=0,
                class_weights=class_weights,
            )
        else:
            trainer = FoldTrainer(
                model=model,
                config=config,
                train_loader=train_loader,
                val_loader=val_loader,
                fold=0,
                class_weights=class_weights,
            )

        metrics = trainer.fit()
        duration = time.time() - start_time
        result = {
            "experiment": exp_key,
            "name": exp_name,
            "config": exp_config,
            "metrics": metrics,
            "duration_seconds": duration,
            "timestamp": datetime.now().isoformat(),
        }
        with open(exp_output / "results.json", "w") as f:
            json.dump(result, f, indent=2, default=str)
        return result

    except Exception as exc:
        duration = time.time() - start_time
        result = {
            "experiment": exp_key,
            "name": exp_name,
            "error": str(exc),
            "duration_seconds": duration,
            "timestamp": datetime.now().isoformat(),
        }
        with open(exp_output / "error.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"  [FAILED] {exp_name}: {exc}")
        import traceback
        traceback.print_exc()
        return result


def main():
    args = parse_args()
    if not verify_sleep_edf_dataset(args.data_dir):
        raise SystemExit(
            "Sleep-EDF dataset not found or incomplete at "
            f"{args.data_dir}. Expected at least "
            f"{MIN_RECOMMENDED_RECORD_PAIRS} matched PSG/Hypnogram pairs."
        )

    experiments = resolve_models(args.models)
    if args.no_pretrained:
        maybe_disable_pretrained(experiments)

    print("=" * 70)
    print("  SLEEP-EDF EEG BENCHMARK PIPELINE (FINAL)")
    print("=" * 70)
    print(f"  Dataset:      {DATASET_NAME}")
    print(f"  W&B Project:  {WANDB_PROJECT}")
    print(f"  Data Dir:     {args.data_dir}")
    print(f"  Output Dir:   {args.output_dir}")
    print(f"  Models ({len(experiments)}):  {', '.join(experiments.keys())}")
    print(f"  Batch Size:   {args.batch_size}")
    print(f"  Num Workers:  {args.num_workers}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Max Records:  {args.max_records or 'all'}")
    print(f"  Device:       {'CUDA (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 70)

    validation_results = {}
    if args.validate_models or args.validate_only:
        validation_results = validate_models(experiments)

    if args.validate_only:
        failed = [k for k, v in validation_results.items() if v.startswith("FAIL")]
        missing = [k for k, v in validation_results.items() if v.startswith("MISSING_DEP")]
        print(f"\nValidation complete. Failed: {failed or 'none'}")
        print(f"Missing deps: {missing or 'none'}")
        return

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    all_results: List[Dict[str, Any]] = []
    total = len(experiments)

    for idx, (exp_key, exp_config) in enumerate(experiments.items(), 1):
        print(f"\n{'#'*70}")
        print(f"  [{idx}/{total}] Starting: {exp_key}")
        print(f"{'#'*70}")

        result = run_sleep_edf_experiment(
            exp_key=exp_key,
            exp_config=exp_config,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            max_records=args.max_records,
            num_workers=args.num_workers,
        )
        all_results.append(result)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Print progress summary
        completed = len([r for r in all_results if "error" not in r])
        failed = len([r for r in all_results if "error" in r])
        print(f"\n  Progress: {idx}/{total} done | {completed} ✓ | {failed} ✗")

    generate_summary(all_results, args.output_dir)

    # Final summary
    print(f"\n{'='*70}")
    print("  ALL EXPERIMENTS COMPLETE")
    print(f"{'='*70}")
    for r in all_results:
        status = "✓" if "error" not in r else "✗"
        duration = r.get("duration_seconds", 0)
        hours = duration / 3600
        print(f"  {status} {r['experiment']:<30} {hours:.1f}h")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
