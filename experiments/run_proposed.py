import os
import argparse
import torch
from qspikexai.data.unified_loader import create_unified_dataloaders
from qspikexai.models.qspikexai_net import QSpikeXAINet
from qspikexai.training.trainer import Trainer, TrainingConfig

def main():
    parser = argparse.ArgumentParser(description="Run QSpikeXAI-Net Proposed Model")
    parser.add_argument('--task', type=str, required=True, choices=['sleep_apnea', 'schizophrenia', 'mci', 'depression'])
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        n_folds=args.folds,
        results_csv='results/canonical_results.csv'
    )

    print(f"Running proposed QSpikeXAI-Net on task {args.task} for {args.folds} folds...")
    
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
        
        # Instantiate proposed model: 4 qubits, 4 heads
        model = QSpikeXAINet(task=args.task, n_qubits=4, n_heads_vqc=4)
        trainer = Trainer(model, args.task, config)
        
        metrics = trainer.fit(train_loader, val_loader, fold=fold)
        print(f"Fold {fold+1} validation metrics: {metrics}")

if __name__ == '__main__':
    main()
