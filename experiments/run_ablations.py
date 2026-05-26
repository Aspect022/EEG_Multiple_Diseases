import os
import argparse
import torch
from qspikexai.data.unified_loader import create_unified_dataloaders
from qspikexai.models.baselines.ablations import SNNOnlyModel, QuantumOnlyModel, ConcatFusionModel
from qspikexai.training.trainer import Trainer, TrainingConfig

def get_ablation_model(model_name: str, task: str):
    if model_name == 'snn_only':
        return SNNOnlyModel(task=task)
    elif model_name == 'quantum_only':
        return QuantumOnlyModel(task=task)
    elif model_name == 'concat_fusion':
        return ConcatFusionModel(task=task)
    else:
        raise ValueError(f"Unknown ablation model: {model_name}")

def main():
    parser = argparse.ArgumentParser(description="Run Ablation Models")
    parser.add_argument('--task', type=str, required=True, choices=['sleep_apnea', 'schizophrenia', 'mci', 'depression'])
    parser.add_argument('--model', type=str, required=True, choices=['snn_only', 'quantum_only', 'concat_fusion'])
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--use-wandb', action='store_true', help='Log to Weights & Biases')
    args = parser.parse_args()

    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        n_folds=args.folds,
        results_csv='results/canonical_results.csv',
        use_wandb=args.use_wandb
    )

    print(f"Running ablation {args.model} on task {args.task} for {args.folds} folds...")
    
    for fold in range(args.folds):
        print(f"\n--- Fold {fold+1}/{args.folds} ---")
        train_loader, val_loader = create_unified_dataloaders(
            task=args.task,
            data_dir=args.data_dir,
            fold=fold,
            n_folds=args.folds,
            batch_size=args.batch_size,
            seed=config.seed
        )
        
        model = get_ablation_model(args.model, args.task)
        trainer = Trainer(model, args.task, config)
        
        metrics = trainer.fit(train_loader, val_loader, fold=fold)
        print(f"Fold {fold+1} validation metrics: {metrics}")

if __name__ == '__main__':
    main()
