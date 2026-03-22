#!/usr/bin/env python3
"""
Dedicated Sleep-EDF experiment pipeline.

This runner mirrors the BOAS pipeline, but targets PhysioNet Sleep-EDF
Expanded so the existing model zoo can be benchmarked on a second EEG
dataset without touching the BOAS-specific path.
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
    create_sleep_edf_dataloaders,
    create_sleep_edf_multimodal_dataloaders,
    verify_sleep_edf_dataset,
)
from src.data.transforms import create_scalogram_transform
from src.training.multimodal_trainer import MultiModalFoldTrainer
from src.training.research_trainer import FoldTrainer, ResearchConfig


STABLE_MODELS = [
    "tcanet",
    "snn_1d_lif",
    "snn_1d_attn",
    "quantum_1d_ring_RYZ",
    "quantum_1d_full_RXY",
    "snn_lif_resnet",
    "snn_lif_vit",
    "swin",
    "vit",
    "deit",
    "efficientnet",
    "convnext",
    "quantum_ring_RXY",
    "quantum_full_RXY",
    "fusion_a",
    "fusion_b",
]

RISKY_MODELS = [
    "snn_qif_resnet",
    "snn_qif_vit",
    "spiking_vit_1d",
    "snn_fusion_early",
    "snn_fusion_late",
    "snn_fusion_gated",
    "quantum_snn_fusion_early",
    "quantum_snn_fusion_gated",
    "fusion_c",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Sleep-EDF benchmarking pipeline")
    parser.add_argument("--data-dir", type=str, default="data/sleep-edf")
    parser.add_argument("--output-dir", type=str, default="outputs/sleep_edf_results")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--models", type=str, default="stable",
                        help='Preset: stable|risky|all or comma-separated model keys')
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--validate-models", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def resolve_models(selector: str) -> Dict[str, Dict[str, Any]]:
    if selector == "stable":
        keys = STABLE_MODELS
    elif selector == "risky":
        keys = RISKY_MODELS
    elif selector == "all":
        keys = [k for k in EXPERIMENT_DEFS if k not in {"snn_qif_resnet"}] + ["snn_qif_resnet"]
    else:
        keys = [k.strip() for k in selector.split(",") if k.strip()]

    experiments = {k: dict(EXPERIMENT_DEFS[k]) for k in keys if k in EXPERIMENT_DEFS}
    if not experiments:
        raise ValueError(f"No valid models resolved from '{selector}'")
    return experiments


def maybe_disable_pretrained(experiments: Dict[str, Dict[str, Any]]) -> None:
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
    print("\n[Validation] Checking model forward passes...")
    results: Dict[str, str] = {}
    for key, config in experiments.items():
        status, message = validate_model_forward(key, config)
        results[key] = f"{status}: {message}"
        print(f"  - {key:<28} {status}  {message}")
    return results


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
    exp_name = exp_config["name"]
    exp_output = Path(output_dir) / exp_key
    exp_output.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    print(f"\n{'='*70}")
    print(f"  Sleep-EDF Experiment: {exp_name}")
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

        model = create_model(exp_config, num_classes=5)
        class_weights = compute_class_weights(train_loader.dataset)

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
        return result


def main():
    args = parse_args()
    if not verify_sleep_edf_dataset(args.data_dir):
        raise SystemExit(f"Sleep-EDF dataset not found or incomplete at {args.data_dir}")

    experiments = resolve_models(args.models)
    if args.no_pretrained:
        maybe_disable_pretrained(experiments)

    print("=" * 70)
    print("  SLEEP-EDF EEG BENCHMARK PIPELINE")
    print("=" * 70)
    print(f"  Data Dir:     {args.data_dir}")
    print(f"  Output Dir:   {args.output_dir}")
    print(f"  Models:       {', '.join(experiments.keys())}")
    print(f"  Batch Size:   {args.batch_size}")
    print(f"  Num Workers:  {args.num_workers}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Max Records:  {args.max_records or 'all'}")
    print(f"  Device:       {'CUDA' if torch.cuda.is_available() else 'CPU'}")
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
    for exp_key, exp_config in experiments.items():
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

    generate_summary(all_results, args.output_dir)


if __name__ == "__main__":
    main()
