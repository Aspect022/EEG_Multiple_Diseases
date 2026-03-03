"""
Training script for baseline CNN models
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from src.data.dataset import get_dataloaders
from src.models.baseline.cnn_baseline import SimpleCNN, DeepCNN
from src.training.trainer import create_trainer


def main():
    print("=" * 70)
    print("🚀 Training Baseline CNN for ECG Classification")
    print("=" * 70)
    
    # Configuration
    config = {
        'experiment_name': 'baseline_simple_cnn',
        'output_dir': 'outputs',
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'use_scheduler': True,
        'use_class_weights': True,  # Handle class imbalance
    }
    
    # Hyperparameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    IMG_SIZE = (224, 224)
    NUM_WORKERS = 0  # CPU only, set to 0
    NUM_CLASSES = 4
    
    print("\n📋 Configuration:")
    print(f"   Experiment: {config['experiment_name']}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Epochs: {NUM_EPOCHS}")
    print(f"   Image size: {IMG_SIZE}")
    print(f"   Learning rate: {config['learning_rate']}")
    print(f"   Optimizer: {config['optimizer']}")
    
    # Create dataloaders
    print("\n📊 Loading data...")
    data_path = Path("data/processed/kaggle_ecg_grayscale")
    
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_path=data_path,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        img_size=IMG_SIZE
    )
    
    print(f"   Classes: {class_names}")
    
    # Create model
    print("\n🏗️  Creating model...")
    model = SimpleCNN(num_classes=NUM_CLASSES)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Model: SimpleCNN")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: {model.get_model_size():.2f} MB")
    
    # Create trainer
    print("\n👨‍🏫 Initializing trainer...")
    trainer = create_trainer(model, train_loader, val_loader, config)
    
    # Train
    print("\n🎯 Starting training...\n")
    history = trainer.train(num_epochs=NUM_EPOCHS)
    
    # Print final results
    print("\n" + "=" * 70)
    print("📊 Final Results:")
    print("=" * 70)
    print(f"Best Validation Accuracy: {trainer.best_val_acc:.2f}%")
    print(f"Best Validation Loss: {trainer.best_val_loss:.4f}")
    print(f"Final Train Accuracy: {history['train_acc'][-1]:.2f}%")
    print(f"Final Train Loss: {history['train_loss'][-1]:.4f}")
    
    # Save model architecture info
    model_info = {
        'model_name': 'SimpleCNN',
        'num_classes': NUM_CLASSES,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'input_size': IMG_SIZE,
        'best_val_acc': trainer.best_val_acc,
        'best_val_loss': trainer.best_val_loss,
    }
    
    import json
    info_path = trainer.checkpoint_dir / 'model_info.json'
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=4)
    
    print(f"\n✓ Model info saved to {info_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()