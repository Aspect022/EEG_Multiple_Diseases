#!/usr/bin/env python3
"""
Comprehensive SNN Training Pipeline - All Variants

This script runs ALL SNN variants systematically:
1. 1D SNNs (LIF, QIF, with/without attention)
2. 2D SNNs (LIF, QIF, ViT variants)
3. Fusion Models (Early, Late, Gated)
4. Quantum-SNN Fusion (using best quantum variant)

Best Quantum Variant (from W&B results):
- Rotation: RXY (R-X-Y gates)
- Entanglement: Full (all-to-all connectivity)
- Accuracy: 83.4%, F1: 65.7%
- Alternative: RXYZ with Full entanglement (83.5%, F1: 63.3%)

Usage:
    python comprehensive_snn_pipeline.py
    python comprehensive_snn_pipeline.py --stage 1  # Run only stage 1
    python comprehensive_snn_pipeline.py --dry-run  # Print what would run
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict


# =============================================================================
# Configuration
# =============================================================================

class Config:
    """Training configuration."""
    
    # Paths (from run.sh - properly configured)
    DATA_DIR = "./data"
    OUTPUT_DIR = "./outputs/results"
    LOG_DIR = "./logs"
    CACHE_DIR = "./data/ds005555_cache"  # Precomputed scalograms
    
    # A100 Optimization
    CUDA_VISIBLE_DEVICES = "0"
    BATCH_SIZE = 128
    NUM_WORKERS = 8
    EPOCHS = 50
    LEARNING_RATE = 3e-4
    
    # Best quantum variant (from W&B analysis)
    BEST_QUANTUM_ROTATION = "RXY"  # 83.4% acc, 65.7% F1
    BEST_QUANTUM_ENTANGLEMENT = "full"  # Full connectivity


# =============================================================================
# Experiment Definitions
# =============================================================================

EXPERIMENTS = {
    # =========================================================================
    # STAGE 1: 1D SNN Variants (Raw EEG)
    # =========================================================================
    "stage_1_1d_snn": [
        {
            "name": "snn_1d_lif",
            "type": "snn_1d",
            "config": {"attention": False, "neuron_type": "lif"},
            "description": "1D SNN with LIF neurons (baseline)",
        },
        {
            "name": "snn_1d_qif",
            "type": "snn_1d",
            "config": {"attention": False, "neuron_type": "qif"},
            "description": "1D SNN with QIF neurons (nonlinear dynamics)",
        },
        {
            "name": "snn_1d_lif_attn",
            "type": "snn_1d",
            "config": {"attention": True, "neuron_type": "lif"},
            "description": "1D SNN with LIF + attention",
        },
        {
            "name": "snn_1d_qif_attn",
            "type": "snn_1d",
            "config": {"attention": True, "neuron_type": "qif"},
            "description": "1D SNN with QIF + attention",
        },
        {
            "name": "spiking_vit_1d",
            "type": "spiking_vit_1d",
            "config": {},
            "description": "Spiking ViT for 1D EEG",
        },
    ],
    
    # =========================================================================
    # STAGE 2: 2D SNN Variants (Scalograms)
    # =========================================================================
    "stage_2_2d_snn": [
        {
            "name": "snn_lif_resnet",
            "type": "snn",
            "config": {"backbone": "resnet18", "neuron_type": "lif"},
            "description": "2D SNN with LIF + ResNet-18",
        },
        {
            "name": "snn_qif_resnet",
            "type": "snn",
            "config": {"backbone": "resnet18", "neuron_type": "qif"},
            "description": "2D SNN with QIF + ResNet-18",
        },
        {
            "name": "snn_lif_vit",
            "type": "snn_vit",
            "config": {"neuron_type": "lif", "variant": "small"},
            "description": "2D SNN with LIF + ViT",
        },
        {
            "name": "snn_qif_vit",
            "type": "snn_vit",
            "config": {"neuron_type": "qif", "variant": "small"},
            "description": "2D SNN with QIF + ViT",
        },
    ],
    
    # =========================================================================
    # STAGE 3: Original Fusion Models (fusion_b, fusion_c)
    # =========================================================================
    "stage_3_original_fusion": [
        {
            "name": "fusion_b",
            "type": "fusion_b",
            "config": {},
            "description": "4-way hybrid fusion (original)",
        },
        {
            "name": "fusion_c",
            "type": "fusion_c",
            "config": {},
            "description": "Multi-modal fusion (original)",
        },
    ],
    
    # =========================================================================
    # STAGE 4: New SNN Fusion Models
    # =========================================================================
    "stage_4_snn_fusion": [
        {
            "name": "snn_fusion_early",
            "type": "snn_fusion_early",
            "config": {
                "dim_1d": 128,
                "dim_2d": 512,
                "fusion_dim": 256,
            },
            "description": "Early fusion (1D + 2D features)",
        },
        {
            "name": "snn_fusion_late",
            "type": "snn_fusion_late",
            "config": {},
            "description": "Late fusion (ensemble averaging)",
        },
        {
            "name": "snn_fusion_gated",
            "type": "snn_fusion_gated",
            "config": {
                "confidence_threshold": 0.7,
                "gate_type": "adaptive",
            },
            "description": "Gated fusion (confidence-based routing) ⭐ MAIN",
        },
    ],
    
    # =========================================================================
    # STAGE 5: Quantum-SNN Fusion (Best Quantum Variant)
    # =========================================================================
    "stage_5_quantum_snn_fusion": [
        {
            "name": "quantum_snn_fusion_early",
            "type": "quantum_snn_fusion_early",
            "config": {
                "quantum_rotation": Config.BEST_QUANTUM_ROTATION,
                "quantum_entanglement": Config.BEST_QUANTUM_ENTANGLEMENT,
                "dim_quantum": 512,
                "dim_1d": 128,
                "fusion_dim": 256,
            },
            "description": f"Quantum (RXY-full) + SNN-1D early fusion",
        },
        {
            "name": "quantum_snn_fusion_gated",
            "type": "quantum_snn_fusion_gated",
            "config": {
                "quantum_rotation": Config.BEST_QUANTUM_ROTATION,
                "quantum_entanglement": Config.BEST_QUANTUM_ENTANGLEMENT,
                "confidence_threshold": 0.7,
                "gate_type": "adaptive",
            },
            "description": f"Quantum (RXY-full) + SNN-1D gated fusion ⭐ NOVEL",
        },
    ],
}


# =============================================================================
# Training Functions
# =============================================================================

def run_experiment(exp_config: Dict, args: argparse.Namespace) -> bool:
    """Run a single experiment."""
    
    cmd = [
        sys.executable, "pipeline.py",
        "--experiment", exp_config["name"],
        "--data-dir", Config.DATA_DIR,
        "--output-dir", Config.OUTPUT_DIR,
        "--epochs", str(Config.EPOCHS),
        "--batch-size", str(Config.BATCH_SIZE),
        "--num-workers", str(Config.NUM_WORKERS),
        "--learning-rate", str(Config.LEARNING_RATE),
    ]
    
    if args.dry_run:
        print(f"  [DRY RUN] Would run: {' '.join(cmd)}")
        return True
    
    print(f"\n{'='*70}")
    print(f"Running: {exp_config['name']}")
    print(f"Description: {exp_config['description']}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")
    
    # Run experiment
    log_file = Path(Config.LOG_DIR) / f"{exp_config['name']}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(log_file, 'w') as f:
        f.write(f"# Experiment: {exp_config['name']}\n")
        f.write(f"# Description: {exp_config['description']}\n")
        f.write(f"# Started: {datetime.now()}\n")
        f.write(f"# Command: {' '.join(cmd)}\n\n")
        f.flush()
        
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    
    success = result.returncode == 0
    status = "✅ SUCCESS" if success else "❌ FAILED"
    print(f"\n{exp_config['name']}: {status}")
    
    return success


def run_stage(stage_name: str, args: argparse.Namespace) -> bool:
    """Run all experiments in a stage."""
    
    if stage_name not in EXPERIMENTS:
        print(f"Unknown stage: {stage_name}")
        print(f"Available stages: {list(EXPERIMENTS.keys())}")
        return False
    
    experiments = EXPERIMENTS[stage_name]
    print(f"\n{'='*70}")
    print(f"STAGE: {stage_name}")
    print(f"Experiments: {len(experiments)}")
    print(f"{'='*70}")
    
    results = []
    for exp in experiments:
        success = run_experiment(exp, args)
        results.append(success)
        
        if not args.dry_run and success:
            # Wait a bit between experiments
            time.sleep(5)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"STAGE {stage_name} COMPLETE")
    print(f"Success: {sum(results)}/{len(results)}")
    print(f"{'='*70}")
    
    return all(results)


def run_all(args: argparse.Namespace) -> bool:
    """Run all stages in sequence."""
    
    print(f"\n{'='*70}")
    print("COMPREHENSIVE SNN TRAINING PIPELINE")
    print(f"Started: {datetime.now()}")
    print(f"Data: {Config.DATA_DIR}")
    print(f"Output: {Config.OUTPUT_DIR}")
    print(f"Best Quantum: {Config.BEST_QUANTUM_ROTATION} + {Config.BEST_QUANTUM_ENTANGLEMENT}")
    print(f"{'='*70}\n")
    
    results = {}
    for stage_name in EXPERIMENTS.keys():
        success = run_stage(stage_name, args)
        results[stage_name] = success
        
        if not success and not args.dry_run:
            print(f"\n⚠️  Stage {stage_name} failed. Continue to next stage?")
            if not args.force_continue:
                response = input("Continue? (y/n): ").strip().lower()
                if response != 'y':
                    break
    
    # Final Summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    for stage, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {stage}")
    print(f"{'='*70}")
    
    return all(results.values())


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Comprehensive SNN Training Pipeline")
    parser.add_argument("--stage", type=str, default="all",
                       help="Stage to run (or 'all' for everything)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Print commands without running")
    parser.add_argument("--force-continue", action="store_true",
                       help="Continue even if a stage fails")
    
    args = parser.parse_args()
    
    # Create directories
    Path(Config.LOG_DIR).mkdir(parents=True, exist_ok=True)
    Path(Config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Run
    if args.stage == "all":
        success = run_all(args)
    else:
        success = run_stage(args.stage, args)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
