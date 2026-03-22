#!/usr/bin/env python3
"""
Unified EEG Classification Pipeline.

Orchestrates dataset download/verification and sequential training of all 19
model experiments on the BOAS (ds005555) sleep EEG dataset:
    - SNN ResNet-18 (LIF + QIF)
    - Spiking ViT (LIF + QIF)
    - Quantum CNN (7 rotations × 2 entanglements = 14 combos)
    - Swin Transformer

Usage:
    python pipeline.py --epochs 30 --batch-size 16
    python pipeline.py --epochs 1 --models snn_lif_resnet --max-subjects 5
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

import torch
import numpy as np


# =========================================================================
# Configuration
# =========================================================================

BOAS_S3_URI = "s3://openneuro.org/ds005555"
BOAS_DIR_NAME = "ds005555"

QUANTUM_ENTANGLEMENTS = ['ring', 'full']
QUANTUM_ROTATIONS = ['RX', 'RY', 'RZ', 'RXY', 'RXZ', 'RYZ', 'RXYZ']

# All experiment definitions
EXPERIMENT_DEFS = {
    'snn_lif_resnet': {
        'type': 'snn', 'backbone': 'resnet18', 'neuron_type': 'lif',
        'name': 'SNN-ResNet18-LIF',
    },
    'snn_qif_resnet': {
        'type': 'snn', 'backbone': 'resnet18', 'neuron_type': 'qif',
        'name': 'SNN-ResNet18-QIF',
    },
    'snn_lif_vit': {
        'type': 'snn_vit', 'neuron_type': 'lif',
        'name': 'SNN-ViT-LIF',
    },
    'snn_qif_vit': {
        'type': 'snn_vit', 'neuron_type': 'qif',
        'name': 'SNN-ViT-QIF',
    },
    'swin': {
        'type': 'swin',
        'name': 'Swin-Transformer',
    },
    'vit': {
        'type': 'vit',
        'name': 'ViT-Small',
    },
    'deit': {
        'type': 'deit',
        'name': 'DeiT-Small',
    },
    'efficientnet': {
        'type': 'efficientnet',
        'name': 'EfficientNet-B0',
    },
    'convnext': {
        'type': 'convnext',
        'name': 'ConvNeXt-Tiny',
    },
}

# Add quantum experiments dynamically (2D scalogram-based)
for ent in QUANTUM_ENTANGLEMENTS:
    for rot in QUANTUM_ROTATIONS:
        key = f'quantum_{ent}_{rot}'
        EXPERIMENT_DEFS[key] = {
            'type': 'quantum',
            'entanglement': ent,
            'rotation': rot,
            'name': f'Quantum-{ent}-{rot}',
            'data_mode': '2d',
        }

# ── Phase 2: 1D Signal Models ──
EXPERIMENT_DEFS['snn_1d_lif'] = {
    'type': 'snn_1d', 'attention': False,
    'name': 'SNN-1D-LIF', 'data_mode': '1d',
}
EXPERIMENT_DEFS['snn_1d_attn'] = {
    'type': 'snn_1d', 'attention': True,
    'name': 'SNN-1D-Attention', 'data_mode': '1d',
}
EXPERIMENT_DEFS['spiking_vit_1d'] = {
    'type': 'spiking_vit_1d',
    'name': 'Spiking-ViT-1D', 'data_mode': '1d',
}

# Add Quantum 1D experiments dynamically (raw signal-based)
for ent in QUANTUM_ENTANGLEMENTS:
    for rot in QUANTUM_ROTATIONS:
        key = f'quantum_1d_{ent}_{rot}'
        EXPERIMENT_DEFS[key] = {
            'type': 'quantum_1d',
            'entanglement': ent,
            'rotation': rot,
            'name': f'Q1D-{ent}-{rot}',
            'data_mode': '1d',
        }

# ── NEW: TCANet Model (Priority #1) ──
EXPERIMENT_DEFS['tcanet'] = {
    'type': 'tcanet',
    'name': 'TCANet-Clean', 'data_mode': '1d',
}

# Transformer baselines
EXPERIMENT_DEFS['swin'] = {
    'type': 'swin',
    'name': 'Swin-Transformer',
}
EXPERIMENT_DEFS['vit'] = {
    'type': 'vit',
    'name': 'ViT-Small',
}
EXPERIMENT_DEFS['deit'] = {
    'type': 'deit',
    'name': 'DeiT-Small',
}
EXPERIMENT_DEFS['efficientnet'] = {
    'type': 'efficientnet',
    'name': 'EfficientNet-B0',
}
EXPERIMENT_DEFS['convnext'] = {
    'type': 'convnext',
    'name': 'ConvNeXt-Tiny',
}

# Add quantum experiments dynamically (2D scalogram-based)
for ent in QUANTUM_ENTANGLEMENTS:
    for rot in QUANTUM_ROTATIONS:
        key = f'quantum_{ent}_{rot}'
        EXPERIMENT_DEFS[key] = {
            'type': 'quantum',
            'entanglement': ent,
            'rotation': rot,
            'name': f'Quantum-{ent}-{rot}',
            'data_mode': '2d',
        }

# Fusion experiments (2D only - already completed: fusion_a, fusion_b)
# fusion_c moved to LAST position (slow training)
EXPERIMENT_DEFS['fusion_a'] = {
    'type': 'fusion_a',
    'name': 'Fusion-Swin-ConvNeXt', 'data_mode': '2d',
}
EXPERIMENT_DEFS['fusion_b'] = {
    'type': 'fusion_b',
    'name': 'Fusion-Hybrid-4Way', 'data_mode': '2d',
}

# SNN Fusion models (1D + 2D)
EXPERIMENT_DEFS['snn_fusion_early'] = {
    'type': 'snn_fusion_early',
    'name': 'SNN-EarlyFusion-1D+2D',
    'data_mode': 'both',
    'dim_1d': 128,
    'dim_2d': 512,
    'fusion_dim': 256,
}
EXPERIMENT_DEFS['snn_fusion_late'] = {
    'type': 'snn_fusion_late',
    'name': 'SNN-LateFusion-Ensemble',
    'data_mode': 'both',
}
EXPERIMENT_DEFS['snn_fusion_gated'] = {
    'type': 'snn_fusion_gated',
    'name': 'SNN-GatedFusion-Adaptive',
    'data_mode': 'both',
    'confidence_threshold': 0.7,
    'gate_type': 'adaptive',
}

# Quantum-SNN Fusion (Best Quantum: RXY + Full)
EXPERIMENT_DEFS['quantum_snn_fusion_early'] = {
    'type': 'quantum_snn_fusion_early',
    'name': 'Quantum-SNN-EarlyFusion-RXY-full',
    'data_mode': 'both',
    'quantum_rotation': 'RXY',
    'quantum_entanglement': 'full',
    'dim_quantum': 512,
    'dim_1d': 128,
    'fusion_dim': 256,
}
EXPERIMENT_DEFS['quantum_snn_fusion_gated'] = {
    'type': 'quantum_snn_fusion_gated',
    'name': 'Quantum-SNN-GatedFusion-RXY-full',
    'data_mode': 'both',
    'quantum_rotation': 'RXY',
    'quantum_entanglement': 'full',
    'confidence_threshold': 0.7,
    'gate_type': 'adaptive',
}

# fusion_c placed LAST due to slow training (1 hour/epoch)
EXPERIMENT_DEFS['fusion_c'] = {
    'type': 'fusion_c',
    'name': 'Fusion-MultiModal (SNN-1D + Swin)', 'data_mode': 'both',
}


# =========================================================================
# Dataset Download & Verification
# =========================================================================

def download_dataset(data_dir: str, max_retries: int = 3) -> bool:
    """
    Download BOAS ds005555 from OpenNeuro S3 with retry logic.

    Uses: aws s3 sync --no-sign-request (parallel, resumable).
    """
    data_path = Path(data_dir)
    boas_path = data_path / BOAS_DIR_NAME

    for attempt in range(1, max_retries + 1):
        if verify_dataset(data_dir):
            print(f"[Dataset] BOAS verified at {boas_path}")
            return True

        print(f"\n[Dataset] Download attempt {attempt}/{max_retries}...")
        data_path.mkdir(parents=True, exist_ok=True)

        try:
            # aws s3 sync (parallel + resumable)
            result = subprocess.run([
                'aws', 's3', 'sync', '--no-sign-request',
                BOAS_S3_URI, str(boas_path),
            ], check=True, timeout=7200)  # 2hr timeout for ~33GB

            if verify_dataset(data_dir):
                print("[Dataset] Download and verification complete!")
                return True
            else:
                print(f"[Dataset] Verification failed (attempt {attempt})")

        except FileNotFoundError:
            print("[Dataset] ERROR: aws CLI not found! Install: pip install awscli")
            return False
        except subprocess.TimeoutExpired:
            print(f"[Dataset] Download timeout (attempt {attempt}), will retry...")
        except Exception as e:
            print(f"[Dataset] Download error: {e}")

    print("[Dataset] ERROR: Failed to download after all retries!")
    return False


def verify_dataset(data_dir: str) -> bool:
    """
    Verify BOAS dataset integrity.

    Checks for:
    - At least 10 subject directories (sub-XXX/)
    - At least 1 EDF file in any subject
    """
    data_path = Path(data_dir)
    boas_path = data_path / BOAS_DIR_NAME

    if not boas_path.exists():
        return False

    # Check subject directories
    subjects = [d for d in boas_path.iterdir() if d.is_dir() and d.name.startswith('sub-')]
    if len(subjects) < 10:
        return False

    # Check for at least one EDF file
    for sub in subjects[:5]:
        edf_files = list(sub.rglob("*.edf"))
        if edf_files:
            return True

    return False


# =========================================================================
# Model Factories
# =========================================================================

def create_model(exp_config: Dict, num_classes: int = 5) -> torch.nn.Module:
    """Create a model based on experiment configuration."""
    exp_type = exp_config['type']

    if exp_type == 'snn':
        from src.models.snn import create_spiking_resnet
        return create_spiking_resnet(
            model_name=exp_config['backbone'],
            num_classes=num_classes,
            neuron_type=exp_config['neuron_type'],
            num_timesteps=8,  # Optimized for speed (8 timesteps gives similar accuracy with 3x speedup)
        )

    elif exp_type == 'snn_vit':
        from src.models.snn import create_spiking_vit
        return create_spiking_vit(
            num_classes=num_classes,
            neuron_type=exp_config['neuron_type'],
            variant='small',
            num_timesteps=8,  # Optimized for speed (8 timesteps gives similar accuracy with 3x speedup)
        )

    elif exp_type == 'quantum':
        from src.models.quantum import create_hybrid_quantum_cnn
        return create_hybrid_quantum_cnn(
            model_name='efficient',
            num_classes=num_classes,
            entanglement_type=exp_config['entanglement'],
            rotation_type=exp_config['rotation'],
        )

    elif exp_type == 'swin':
        from src.models.transformer import create_swin_classifier
        return create_swin_classifier(
            model_name='swin_tiny_patch4_window7_224',
            num_classes=num_classes,
            pretrained=exp_config.get('pretrained', True),
        )

    elif exp_type == 'vit':
        from src.models.transformer import create_vit_classifier
        return create_vit_classifier(
            num_classes=num_classes,
            pretrained=exp_config.get('pretrained', True),
        )

    elif exp_type == 'deit':
        from src.models.transformer import create_deit_classifier
        return create_deit_classifier(
            num_classes=num_classes,
            pretrained=exp_config.get('pretrained', True),
        )

    elif exp_type == 'efficientnet':
        from src.models.transformer import create_efficientnet_classifier
        return create_efficientnet_classifier(
            num_classes=num_classes,
            pretrained=exp_config.get('pretrained', True),
        )

    elif exp_type == 'convnext':
        from src.models.transformer import create_convnext_classifier
        return create_convnext_classifier(
            num_classes=num_classes,
            pretrained=exp_config.get('pretrained', True),
        )

    elif exp_type == 'snn_1d':
        from src.models.snn_1d import create_snn_1d_lif, create_snn_1d_attention
        if exp_config.get('attention', False):
            return create_snn_1d_attention(num_classes=num_classes)
        else:
            return create_snn_1d_lif(num_classes=num_classes)

    elif exp_type == 'spiking_vit_1d':
        from src.models.snn_1d import create_spiking_vit_1d
        return create_spiking_vit_1d(num_classes=num_classes)

    elif exp_type == 'quantum_1d':
        from src.models.quantum.quantum_1d import create_quantum_1d
        return create_quantum_1d(
            num_classes=num_classes,
            rotation=exp_config['rotation'],
            entanglement=exp_config['entanglement'],
        )

    elif exp_type == 'tcanet':
        # NEW: TCANet model (Priority #1)
        from tcanet_clean import TCANetClean
        return TCANetClean(
            channels=6,  # BOAS has 6 EEG channels
            time=3000,   # 30 seconds at 100 Hz
            num_classes=num_classes,
            f1=40,
            filters=64,
            pooling_size=75,
            tcn_depth=4,
            num_heads=8,
            dropout=0.5,
        )

    elif exp_type == 'fusion_a':
        from src.models.fusion import create_fusion_a
        return create_fusion_a(
            num_classes=num_classes,
            pretrained=exp_config.get('pretrained', True),
        )

    elif exp_type == 'fusion_b':
        from src.models.fusion import create_fusion_b
        return create_fusion_b(
            num_classes=num_classes,
            pretrained=exp_config.get('pretrained', True),
        )

    elif exp_type == 'fusion_c':
        from src.models.fusion import create_fusion_c
        return create_fusion_c(
            num_classes=num_classes,
            pretrained=exp_config.get('pretrained', True),
        )
    
    elif exp_type == 'snn_fusion_early':
        from src.models.fusion import create_early_fusion_complete
        # Create 1D and 2D feature extractors
        from src.models.snn_1d import create_snn_1d_lif
        from src.models.snn import create_spiking_resnet
        model_1d = create_snn_1d_lif(num_classes=num_classes)
        model_2d = create_spiking_resnet('resnet18', num_classes=num_classes, neuron_type='lif')
        return create_early_fusion_complete(
            model_1d=model_1d,
            model_2d=model_2d,
            num_classes=num_classes,
            dim_1d=exp_config.get('dim_1d', 128),
            dim_2d=exp_config.get('dim_2d', 512),
            fusion_dim=exp_config.get('fusion_dim', 256),
        )

    elif exp_type == 'snn_fusion_late':
        from src.models.fusion import create_late_fusion_complete
        from src.models.snn_1d import create_snn_1d_lif
        from src.models.snn import create_spiking_resnet
        model_1d = create_snn_1d_lif(num_classes=num_classes)
        model_2d = create_spiking_resnet('resnet18', num_classes=num_classes, neuron_type='lif')
        return create_late_fusion_complete(
            model_1d=model_1d,
            model_2d=model_2d,
            num_classes=num_classes,
        )

    elif exp_type == 'snn_fusion_gated':
        from src.models.fusion import create_gated_fusion_complete
        from src.models.snn_1d import create_snn_1d_lif
        from src.models.snn import create_spiking_resnet
        model_1d = create_snn_1d_lif(num_classes=num_classes)
        model_2d = create_spiking_resnet('resnet18', num_classes=num_classes, neuron_type='lif')
        return create_gated_fusion_complete(
            model_1d=model_1d,
            model_2d=model_2d,
            num_classes=num_classes,
            dim_1d=128,
            confidence_threshold=exp_config.get('confidence_threshold', 0.7),
            gate_type=exp_config.get('gate_type', 'adaptive'),
        )
    
    elif exp_type == 'quantum_snn_fusion_early':
        from src.models.fusion import create_quantum_snn_fusion_early
        return create_quantum_snn_fusion_early(
            num_classes=num_classes,
            dim_1d=exp_config.get('dim_1d', 128),
            dim_quantum=exp_config.get('dim_quantum', 512),
            fusion_dim=exp_config.get('fusion_dim', 256),
            quantum_rotation=exp_config.get('quantum_rotation', 'RXY'),
            quantum_entanglement=exp_config.get('quantum_entanglement', 'full'),
        )
    
    elif exp_type == 'quantum_snn_fusion_gated':
        from src.models.fusion import create_quantum_snn_fusion_gated
        return create_quantum_snn_fusion_gated(
            num_classes=num_classes,
            dim_1d=128,
            dim_quantum=exp_config.get('dim_quantum', 512),
            confidence_threshold=exp_config.get('confidence_threshold', 0.7),
            gate_type=exp_config.get('gate_type', 'adaptive'),
            quantum_rotation=exp_config.get('quantum_rotation', 'RXY'),
            quantum_entanglement=exp_config.get('quantum_entanglement', 'full'),
        )

    else:
        raise ValueError(f"Unknown experiment type: {exp_type}")


# =========================================================================
# Training Runner
# =========================================================================

def run_experiment(
    exp_key: str,
    exp_config: Dict,
    data_dir: str,
    output_dir: str,
    epochs: int = 30,
    batch_size: int = 128,
    num_classes: int = 5,
    learning_rate: float = 1e-3,
    max_subjects: int = None,
) -> Dict[str, Any]:
    """
    Run a single training experiment.

    Returns:
        Dict with experiment results and metrics.
    """
    from src.training.research_trainer import ResearchConfig, FoldTrainer

    exp_name = exp_config['name']
    exp_output = Path(output_dir) / exp_key
    exp_output.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  EXPERIMENT: {exp_name}")
    print(f"  Output: {exp_output}")
    print(f"{'='*70}")

    start_time = time.time()

    try:
        # Find dataset path
        boas_path = Path(data_dir) / BOAS_DIR_NAME
        cache_path = Path(data_dir) / 'ds005555_cache'

        if not boas_path.exists():
            raise FileNotFoundError(f"BOAS dataset not found at {boas_path}")

        # Enable cuDNN auto-tuning for faster convolutions
        torch.backends.cudnn.benchmark = True

        # Import data utilities
        from src.data.boas_dataset import (
            create_cached_dataloaders,
            create_boas_dataloaders,
            create_boas_multimodal_dataloaders,
        )
        from src.data.transforms import create_scalogram_transform

        # Check for scalogram cache
        has_cache = (
            cache_path.exists() and (
                (cache_path / 'train_meta.json').exists() or
                (cache_path / 'train_data.pt').exists()
            )
        )

        # Determine data mode
        data_mode = exp_config.get('data_mode', '2d')

        # === Load data based on mode ===
        if data_mode == '1d':
            # 1D raw signal mode — no scalograms needed
            print("  [DATA] Loading raw 1D EEG signals (no CWT)")
            train_loader, val_loader, test_loader = create_boas_dataloaders(
                data_dir=str(boas_path),
                batch_size=batch_size,
                transform=None,  # Raw signal: (6, 3000)
                num_workers=4,
                max_subjects=max_subjects,
            )
        elif data_mode == 'both':
            print("  [DATA] Loading paired raw 1D + 2D scalograms")
            transform = create_scalogram_transform(
                output_size=(224, 224), sampling_rate=100
            )
            train_loader, val_loader, test_loader = create_boas_multimodal_dataloaders(
                data_dir=str(boas_path),
                batch_size=batch_size,
                num_workers=4,
                scalogram_transform=transform,
                max_subjects=max_subjects,
            )
        else:
            # Default 2D scalogram mode
            if has_cache:
                print("  [CACHE] Loading memory-mapped scalograms")
                train_loader, val_loader, test_loader = create_cached_dataloaders(
                    cache_dir=str(cache_path),
                    batch_size=batch_size,
                )
            else:
                print("  [WARN] No scalogram cache found. Using slow CWT path.")
                transform = create_scalogram_transform(
                    output_size=(224, 224), sampling_rate=100
                )
                train_loader, val_loader, test_loader = create_boas_dataloaders(
                    data_dir=str(boas_path),
                    batch_size=batch_size,
                    transform=transform,
                    num_workers=4,
                    max_subjects=max_subjects,
                )

        # Create model
        model = create_model(exp_config, num_classes=num_classes)

        # Note: torch.compile disabled — hangs on SNN/quantum dynamic graphs
        # The A100 with AMP is fast enough without it

        # --- Compute class weights (inverse frequency) for imbalanced data ---
        # Sleep staging is heavily imbalanced: N2 ~ 60%. Without weighting,
        # models (especially from-scratch SNN/quantum) collapse to majority class.
        class_weights = None
        try:
            ds = train_loader.dataset
            if hasattr(ds, 'labels'):
                labels = ds.labels
                if hasattr(labels, 'numpy'):
                    labels = labels.numpy()
                unique, counts = np.unique(labels, return_counts=True)
                # Inverse frequency: rare classes get higher weight
                freq = counts / counts.sum()
                weights = 1.0 / (freq * num_classes)
                # Normalize so mean weight = 1
                weights = weights / weights.mean()
                class_weights = torch.tensor(weights, dtype=torch.float32)
                print(f"  [WEIGHTS] Class weights: {dict(zip(unique.tolist(), [f'{w:.2f}' for w in weights]))}")
        except Exception as e:
            print(f"  [WEIGHTS] Could not compute class weights: {e}")

        # SNN-specific configuration
        is_snn = exp_config['type'] in ['snn', 'snn_vit', 'snn_1d', 'spiking_vit_1d']
        is_snn_1d = exp_config['type'] in ['snn_1d', 'spiking_vit_1d']
        
        # Use SNN-optimized hyperparameters
        if is_snn:
            snn_lr = 3e-4  # Reduced learning rate for SNN stability
            snn_warmup = 10  # Longer warmup for SNNs
            snn_grad_clip = 5.0  # Higher clip for BPTT
            snn_epochs = 50  # SNNs need more epochs
            mixed_precision = True  # Enabled for 3x speedup on A100

            if is_snn_1d:
                print(f"  [SNN-1D MODE] Using 1D SNN-optimized hyperparameters:")
                print(f"    - Model type: {exp_config['type']} (raw EEG signal processing)")
                print(f"    - Learning rate: {snn_lr} (reduced for BPTT stability)")
                print(f"    - Warmup epochs: {snn_warmup} (longer for spike dynamics)")
                print(f"    - Gradient clip: {snn_grad_clip} (higher for BPTT)")
                print(f"    - Epochs: {snn_epochs} (more for convergence)")
                print(f"    - Mixed precision: Enabled (AMP for 3x speedup)")
                print(f"    - Timesteps: 8 (optimized for speed/accuracy tradeoff)")
                print(f"    - Input: Poisson spike encoding (continuous → binary spikes)")
            else:
                print(f"  [SNN-2D MODE] Using SNN-optimized hyperparameters:")
                print(f"    - Learning rate: {snn_lr} (reduced for BPTT stability)")
                print(f"    - Warmup epochs: {snn_warmup} (longer for spike dynamics)")
                print(f"    - Gradient clip: {snn_grad_clip} (higher for BPTT)")
                print(f"    - Epochs: {snn_epochs} (more for convergence)")
                print(f"    - Mixed precision: Enabled (AMP for 3x speedup)")
                print(f"    - Timesteps: 8 (optimized for speed/accuracy tradeoff)")
        else:
            snn_lr = learning_rate
            snn_warmup = 3
            snn_grad_clip = 1.0
            snn_epochs = epochs
            mixed_precision = True

        # Training config
        config = ResearchConfig(
            experiment_name=exp_key,
            output_dir=str(output_dir),
            epochs=snn_epochs,
            learning_rate=snn_lr,
            batch_size=batch_size,
            num_classes=num_classes,
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=4,
            early_stopping=True,
            patience=10,
            save_best=True,
            seed=42,
            warmup_epochs=snn_warmup,
            max_grad_norm=snn_grad_clip,
        )

        # Train (single fold for speed)
        if data_mode == 'both':
            from src.training.multimodal_trainer import MultiModalFoldTrainer
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

        # Save results
        result = {
            'experiment': exp_key,
            'name': exp_name,
            'config': exp_config,
            'metrics': metrics,
            'duration_seconds': duration,
            'timestamp': datetime.now().isoformat(),
        }

        result_path = exp_output / 'results.json'
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)

        print(f"\n  [{exp_name}] Completed in {duration:.1f}s")
        print(f"  Results saved to {result_path}")

        return result

    except Exception as e:
        duration = time.time() - start_time
        error_result = {
            'experiment': exp_key,
            'name': exp_name,
            'error': str(e),
            'duration_seconds': duration,
            'timestamp': datetime.now().isoformat(),
        }

        error_path = exp_output / 'error.json'
        with open(error_path, 'w') as f:
            json.dump(error_result, f, indent=2)

        print(f"\n  [{exp_name}] FAILED: {e}")
        import traceback
        traceback.print_exc()

        return error_result


# =========================================================================
# Results Summary
# =========================================================================

def generate_summary(results: List[Dict], output_dir: str):
    """Generate a consolidated results summary."""
    summary_path = Path(output_dir) / 'experiment_summary.json'
    md_path = Path(output_dir) / 'experiment_summary.md'

    # JSON summary
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Markdown summary
    lines = [
        "# EEG Sleep Staging Pipeline — Results Summary",
        f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Dataset**: BOAS (ds005555) — 5-class Sleep Staging (W/N1/N2/N3/REM)",
        f"**Total experiments**: {len(results)}",
        "",
        "## Results Table",
        "",
        "| # | Experiment | Accuracy | F1 (macro) | MCC | Cohen's κ | Duration |",
        "|---|-----------|----------|-----------|-----|-----------|----------|",
    ]

    for i, r in enumerate(results, 1):
        if r is None:
            lines.append(f"| {i} | Experiment {i} | ERROR (None result) | — | — | — | — |")
            continue
        
        name = r.get('name', r.get('experiment', '?'))
        if 'error' in r:
            lines.append(f"| {i} | {name} | ERROR | — | — | — | {r.get('duration_seconds', 0):.0f}s |")
        else:
            m = r.get('metrics', {})
            acc = m.get('accuracy', 0)
            f1 = m.get('f1_macro', 0)
            mcc = m.get('MCC', 0)
            kappa = m.get('cohens_kappa', 0)
            dur = r.get('duration_seconds', 0)
            lines.append(f"| {i} | {name} | {acc:.4f} | {f1:.4f} | {mcc:.4f} | {kappa:.4f} | {dur:.0f}s |")

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

    print(f"\nSummary saved to {summary_path}")
    print(f"Markdown report saved to {md_path}")


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="EEG Sleep Staging Pipeline")
    parser.add_argument('--epochs', type=int, default=30, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--data-dir', type=str, default='data', help='Dataset directory')
    parser.add_argument('--output-dir', type=str, default='outputs/results', help='Output directory')
    parser.add_argument('--skip-download', action='store_true', help='Skip dataset download')
    parser.add_argument('--models', type=str, default='all',
                        help='Comma-separated model keys or "all". E.g. snn_lif_resnet,swin')
    parser.add_argument('--max-subjects', type=int, default=None,
                        help='Limit subjects per split (for quick testing)')
    args = parser.parse_args()

    print("=" * 70)
    print("  EEG SLEEP STAGING PIPELINE")
    print("  Dataset: BOAS ds005555 (W/N1/N2/N3/REM)")
    print("  Models: Phase1(23) + SNN-1D(3) + Q-1D(14) + Fusion(3) = 43")
    print("=" * 70)
    print(f"  Epochs:       {args.epochs}")
    print(f"  Batch Size:   {args.batch_size}")
    print(f"  LR:           {args.lr}")
    print(f"  Data Dir:     {args.data_dir}")
    print(f"  Output Dir:   {args.output_dir}")
    print(f"  Max Subjects: {args.max_subjects or 'all'}")
    print(f"  Device:       {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"  GPU:          {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:         {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 70)

    # Step 1: Dataset
    if not args.skip_download:
        print("\n[Step 1/3] Downloading and verifying BOAS dataset...")
        if not download_dataset(args.data_dir):
            print("FATAL: Could not prepare dataset. Exiting.")
            sys.exit(1)
    else:
        print("\n[Step 1/3] Skipping download (--skip-download)")
        if not verify_dataset(args.data_dir):
            print("WARNING: Dataset verification failed. Training may fail.")

    # Step 2: Determine experiments to run
    if args.models == 'all':
        experiments = EXPERIMENT_DEFS
    else:
        keys = [k.strip() for k in args.models.split(',')]
        experiments = {k: EXPERIMENT_DEFS[k] for k in keys if k in EXPERIMENT_DEFS}
        if not experiments:
            print(f"No valid models found in: {args.models}")
            print(f"Available: {', '.join(EXPERIMENT_DEFS.keys())}")
            sys.exit(1)

    print(f"\n[Step 2/3] Running {len(experiments)} experiments...")
    for k in experiments:
        print(f"  - {experiments[k]['name']}")

    # Step 3: Run experiments
    all_results = []
    for exp_key, exp_config in experiments.items():
        # Try with full batch size, halve on OOM (128 → 64 → 32)
        batch = args.batch_size
        result = None
        while batch >= 16:
            result = run_experiment(
                exp_key=exp_key,
                exp_config=exp_config,
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                epochs=args.epochs,
                batch_size=batch,
                learning_rate=args.lr,
                max_subjects=args.max_subjects,
            )

            # Check if it was an OOM failure
            if 'error' in result and 'out of memory' in str(result['error']).lower():
                old_batch = batch
                batch = batch // 2
                print(f"\n  [OOM RETRY] {exp_config['name']}: batch {old_batch} → {batch}")
                # Aggressive VRAM cleanup before retry
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                import gc; gc.collect()
                continue
            else:
                break  # Success or non-OOM error

        all_results.append(result)

        # Free GPU memory between experiments
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc; gc.collect()

    # Step 4: Summary
    print(f"\n[Step 3/3] Generating summary...")
    generate_summary(all_results, args.output_dir)

    # Final report
    successes = sum(1 for r in all_results if 'error' not in r)
    failures = len(all_results) - successes
    print(f"\n{'='*70}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Successful: {successes}/{len(all_results)}")
    if failures > 0:
        print(f"  Failed: {failures}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
