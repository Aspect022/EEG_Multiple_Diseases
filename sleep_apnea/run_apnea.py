#!/usr/bin/env python3
"""
Sleep Apnea Severity Classification Pipeline.

Main entry point for training and evaluation.

Usage:
    # Train Custom CNN baseline
    python -m sleep_apnea.run_apnea --model cnn_baseline --data-dir /path/to/shhs --epochs 30
    
    # Train ResNet18 with transfer learning
    python -m sleep_apnea.run_apnea --model resnet18_transfer --data-dir /path/to/shhs --epochs 30 --pretrained
    
    # Train ViT+BiLSTM with SSL pretraining
    python -m sleep_apnea.run_apnea --model vit_bilstm --data-dir /path/to/shhs --epochs 30 --ssl-pretrain
    
    # Cross-dataset evaluation
    python -m sleep_apnea.run_apnea --model vit_bilstm --checkpoint path/to/ckpt.pt --eval-only --target-dataset apnea_ecg
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Sleep Apnea Severity Classification Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Model selection
    parser.add_argument(
        '--model', type=str, required=True,
        choices=['cnn_baseline', 'resnet18_transfer', 'vit_bilstm'],
        help='Model architecture to train',
    )
    
    # Dataset
    parser.add_argument(
        '--dataset', type=str, default='shhs',
        choices=['shhs', 'apnea_ecg'],
        help='Dataset to use',
    )
    parser.add_argument(
        '--data-dir', type=str, required=True,
        help='Path to dataset directory',
    )
    
    # Training
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1.0e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1.0e-4, help='Weight decay')
    
    # Model-specific
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained weights')
    parser.add_argument('--in-channels', type=int, default=6, help='Input channels')
    
    # SSL pretraining
    parser.add_argument('--ssl-pretrain', action='store_true', help='Use self-supervised pretraining')
    parser.add_argument('--ssl-epochs', type=int, default=100, help='SSL pretraining epochs')
    parser.add_argument('--mask-ratio', type=float, default=0.75, help='MAE mask ratio')
    
    # Evaluation
    parser.add_argument('--eval-only', action='store_true', help='Skip training, only evaluate')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint for evaluation')
    parser.add_argument('--target-dataset', type=str, help='Target dataset for cross-dataset evaluation')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='./sleep_apnea/outputs', help='Output directory')
    parser.add_argument('--experiment-name', type=str, default=None, help='Experiment name')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--config', type=str, default=None, help='Path to config YAML')
    
    return parser.parse_args()


# =============================================================================
# Helper Functions
# =============================================================================

def seed_everything(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"  🌱 Random seed: {seed}")


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        # Load default config
        default_config = project_root / 'sleep_apnea' / 'configs' / 'default_config.yaml'
        if default_config.exists():
            with open(default_config) as f:
                return yaml.safe_load(f)
    else:
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def create_model(model_name: str, args, config: Dict) -> torch.nn.Module:
    """Create model based on arguments."""
    from sleep_apnea.models import (
        create_cnn_baseline,
        create_resnet18_transfer,
        create_vit_bilstm,
        create_mae_pretrainer,
    )
    
    num_classes = 4  # Fixed for apnea severity
    
    if model_name == 'cnn_baseline':
        return create_cnn_baseline(
            in_channels=args.in_channels,
            num_classes=num_classes,
        )
    
    elif model_name == 'resnet18_transfer':
        return create_resnet18_transfer(
            in_channels=args.in_channels,
            num_classes=num_classes,
            pretrained=args.pretrained,
        )
    
    elif model_name == 'vit_bilstm':
        return create_vit_bilstm(
            vit_pretrained=args.pretrained,
            num_classes=num_classes,
        )
    
    else:
        raise ValueError(f"Unknown model: {model_name}")


def create_dataloaders(dataset_name: str, data_dir: str, batch_size: int, num_workers: int):
    """Create data loaders for specified dataset."""
    from sleep_apnea.data import (
        create_shhs_dataloaders,
        create_apnea_ecg_dataloaders,
        create_apnea_transform,
    )
    
    transform = create_apnea_transform()
    
    if dataset_name == 'shhs':
        return create_shhs_dataloaders(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            transform=transform,
        )
    
    elif dataset_name == 'apnea_ecg':
        return create_apnea_ecg_dataloaders(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            transform=transform,
        )
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


# =============================================================================
# Main Training Function
# =============================================================================

def train(args):
    """Main training function."""
    print("\n" + "="*70)
    print("  🫁 Sleep Apnea Severity Classification Pipeline")
    print("="*70 + "\n")
    
    # Set seed
    seed_everything(args.seed)
    
    # Load config
    config = load_config(args.config)
    
    # Create model
    print(f"  📦 Creating model: {args.model}")
    model = create_model(args.model, args, config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  📊 Model parameters: {num_params:,}")
    
    # Create data loaders
    print(f"\n  📂 Loading dataset: {args.dataset}")
    train_loader, val_loader, test_loader = create_dataloaders(
        args.dataset, args.data_dir, args.batch_size, args.num_workers
    )
    
    # SSL pretraining (optional)
    if args.ssl_pretrain:
        print("\n  🔄 Starting SSL pretraining...")
        from sleep_apnea.models import create_mae_pretrainer
        from sleep_apnea.training import SSLTrainer, SSLConfig
        
        # Create MAE pretrainer
        pretrainer = create_mae_pretrainer(
            mask_ratio=args.mask_ratio,
        )
        
        # SSL config
        ssl_config = SSLConfig(
            experiment_name=f'{args.model}_ssl_pretrain',
            output_dir=args.output_dir.replace('outputs', 'checkpoints'),
            epochs=args.ssl_epochs,
            batch_size=args.batch_size,
            learning_rate=1.5e-4,
            seed=args.seed,
        )
        
        # Pretrain
        ssl_trainer = SSLTrainer(pretrainer, ssl_config, train_loader)
        ssl_results = ssl_trainer.fit()
        
        # Get classifier for fine-tuning
        model = pretrainer.finetune_classifier(num_classes=4)
        print("  ✅ SSL pretraining complete, ready for fine-tuning\n")
    
    # Training
    if not args.eval_only:
        from sleep_apnea.training import ApneaTrainer, ApneaConfig
        
        # Training config
        train_config = ApneaConfig(
            experiment_name=args.experiment_name or f'{args.model}_{args.dataset}',
            output_dir=args.output_dir,
            epochs=args.epochs,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
        )
        
        # Create trainer
        trainer = ApneaTrainer(
            model=model,
            config=train_config,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
        )
        
        # Train
        results = trainer.fit()
        
        print("\n  ✅ Training complete!")
        print(f"  📈 Best validation metric: {results['best_metric']:.4f}")
    
    # Evaluation only
    if args.eval_only and args.checkpoint:
        from sleep_apnea.training import ApneaTrainer, ApneaConfig
        
        print(f"\n  📥 Loading checkpoint: {args.checkpoint}")
        
        # Create dummy trainer for evaluation
        train_config = ApneaConfig(
            experiment_name=args.experiment_name or f'{args.model}_eval',
            output_dir=args.output_dir,
        )
        
        trainer = ApneaTrainer(
            model=model,
            config=train_config,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
        )
        trainer.load_checkpoint(args.checkpoint)
        
        # Evaluate
        test_results = trainer.evaluate(test_loader)
        
        print("\n  📊 Evaluation Results:")
        for metric, value in test_results.items():
            if isinstance(value, float):
                print(f"    {metric}: {value:.4f}")
    
    # Cross-dataset evaluation
    if args.target_dataset:
        print(f"\n  🔄 Cross-dataset evaluation: {args.dataset} → {args.target_dataset}")
        # TODO: Implement cross-dataset evaluation
    
    print("\n" + "="*70)
    print("  Pipeline complete!")
    print("="*70 + "\n")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == '__main__':
    args = parse_args()
    train(args)
