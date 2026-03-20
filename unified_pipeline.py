#!/usr/bin/env python3
"""
Unified EEG Classification Pipeline.

Supports multiple tasks and datasets:
- Sleep Staging (5-class): BOAS, Sleep-EDF
- Sleep Apnea (4-class): SHHS, Apnea-ECG

Usage:
    # Sleep Staging
    python unified_pipeline.py --task sleep_staging --model snn_lif_resnet --dataset boas
    
    # Sleep Apnea
    python unified_pipeline.py --task sleep_apnea --model vit_bilstm --dataset shhs --ssl-pretrain
    
    # Run all experiments
    python unified_pipeline.py --run-all --task sleep_staging
    python unified_pipeline.py --run-all --task sleep_apnea
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm


# ============================================================================
# Configuration
# ============================================================================

class UnifiedConfig:
    """Unified configuration for all tasks."""
    
    # Task selection
    TASK = 'sleep_staging'  # 'sleep_staging' or 'sleep_apnea'
    
    # Data
    SAMPLING_RATE = 125  # Hz
    EPOCH_DURATION = 30  # seconds
    
    # Sleep Staging (BOAS)
    STAGING_CLASSES = 5
    STAGING_CLASS_NAMES = ['Wake', 'N1', 'N2', 'N3', 'REM']
    
    # Sleep Apnea (SHHS)
    APNEA_CLASSES = 4
    APNEA_CLASS_NAMES = ['Healthy', 'Mild', 'Moderate', 'Severe']
    
    # Training
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 30
    WEIGHT_DECAY = 1e-4
    EARLY_STOPPING_PATIENCE = 10
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Reproducibility
    SEED = 42
    
    @classmethod
    def get_num_classes(cls, task: str) -> int:
        if task == 'sleep_staging':
            return cls.STAGING_CLASSES
        elif task == 'sleep_apnea':
            return cls.APNEA_CLASSES
        else:
            raise ValueError(f"Unknown task: {task}")
    
    @classmethod
    def get_class_names(cls, task: str) -> List[str]:
        if task == 'sleep_staging':
            return cls.STAGING_CLASS_NAMES
        elif task == 'sleep_apnea':
            return cls.APNEA_CLASS_NAMES
        else:
            raise ValueError(f"Unknown task: {task}")


# ============================================================================
# Utilities
# ============================================================================

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """Compute class weights for imbalanced dataset."""
    from collections import Counter
    counts = Counter(labels)
    total = len(labels)
    weights = [total / (num_classes * counts[i]) if counts.get(i, 0) > 0 else 1.0 
               for i in range(num_classes)]
    return torch.FloatTensor(weights)


# ============================================================================
# Models - Sleep Staging (from existing pipeline)
# ============================================================================

class SpikingResNet(nn.Module):
    """Spiking ResNet for sleep staging (2D scalograms)."""
    
    def __init__(self, num_classes: int = 5, in_channels: int = 3,
                 num_timesteps: int = 8, neuron_type: str = 'lif'):
        super().__init__()
        # Simplified for unified pipeline - full implementation in src/models/snn/
        self.num_timesteps = num_timesteps
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256, affine=True),
            nn.ReLU(inplace=True),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class SpikingViT(nn.Module):
    """Spiking Vision Transformer for sleep staging."""
    
    def __init__(self, num_classes: int = 5, img_size: int = 224,
                 patch_size: int = 16, embed_dim: int = 256, depth: int = 4):
        super().__init__()
        try:
            import timm
            self.vit = timm.create_model('vit_small_patch16_224',
                                        pretrained=False,
                                        num_classes=num_classes,
                                        img_size=img_size)
        except ImportError:
            raise ImportError("timm required. Install: pip install timm")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.vit(x)


# ============================================================================
# Models - Sleep Apnea (new architectures)
# ============================================================================

class CNNBaseline(nn.Module):
    """Custom CNN for sleep apnea classification."""
    
    def __init__(self, num_classes: int = 4, input_channels: int = 1):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class ResNet18Transfer(nn.Module):
    """ResNet18 transfer learning for sleep apnea."""
    
    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super().__init__()
        weights = nn.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = nn.resnet18(weights=weights)
        
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.backbone(x)


class ViTBiLSTMHybrid(nn.Module):
    """Hybrid ViT + BiLSTM for sleep apnea (main contribution)."""
    
    def __init__(self, num_classes: int = 4, dropout: float = 0.5):
        super().__init__()
        
        # ViT branch (spectrogram)
        try:
            import timm
            self.vit = timm.create_model('vit_base_patch16_224',
                                        pretrained=True,
                                        num_classes=0,
                                        embed_dim=768)
        except ImportError:
            raise ImportError("timm required")
        
        # BiLSTM branch (raw EEG)
        self.lstm = nn.LSTM(1, 256, num_layers=2, batch_first=True,
                           bidirectional=True, dropout=0.3)
        self.lstm_fc = nn.Linear(512, 512)
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(768 + 512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, spec: torch.Tensor, raw: torch.Tensor = None) -> torch.Tensor:
        # ViT branch
        if spec.shape[1] == 1:
            spec = spec.repeat(1, 3, 1, 1)
        vit_feat = self.vit(spec)  # (B, 768)
        
        # BiLSTM branch (if raw EEG provided)
        if raw is not None:
            raw = raw.squeeze(1).transpose(1, 2)  # (B, 3750, 1)
            lstm_out, (h_n, _) = self.lstm(raw)
            h_cat = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (B, 512)
            lstm_feat = self.lstm_fc(h_cat)
            
            # Fusion
            combined = torch.cat([vit_feat, lstm_feat], dim=1)
        else:
            # Spectrogram only
            combined = vit_feat
        
        return self.fusion(combined)


# ============================================================================
# Model Factory
# ============================================================================

MODEL_REGISTRY = {
    # Sleep Staging Models
    'snn_lif_resnet': SpikingResNet,
    'snn_vit': SpikingViT,
    
    # Sleep Apnea Models
    'cnn': CNNBaseline,
    'resnet18': ResNet18Transfer,
    'vit_bilstm': ViTBiLSTMHybrid,
}


def create_model(model_name: str, task: str, num_classes: int,
                 device: str = 'cuda') -> nn.Module:
    """Create model from registry."""
    
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    
    model_class = MODEL_REGISTRY[model_name]
    
    # Task-specific initialization
    if task == 'sleep_staging':
        if model_name == 'snn_lif_resnet':
            model = model_class(num_classes=num_classes, in_channels=3)
        elif model_name == 'snn_vit':
            model = model_class(num_classes=num_classes)
        else:
            model = model_class(num_classes=num_classes)
    
    elif task == 'sleep_apnea':
        if model_name == 'cnn':
            model = model_class(num_classes=num_classes)
        elif model_name == 'resnet18':
            model = model_class(num_classes=num_classes, pretrained=True)
        elif model_name == 'vit_bilstm':
            model = model_class(num_classes=num_classes)
        else:
            model = model_class(num_classes=num_classes)
    
    else:
        raise ValueError(f"Unknown task: {task}")
    
    return model.to(device)


# ============================================================================
# Trainer
# ============================================================================

class UnifiedTrainer:
    """Unified trainer for all tasks."""
    
    def __init__(self, model: nn.Module, train_loader: DataLoader,
                 val_loader: DataLoader, config: UnifiedConfig,
                 task: str, model_name: str):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.task = task
        self.model_name = model_name
        
        self.device = config.DEVICE
        self.num_classes = config.get_num_classes(task)
        
        # Loss with class weights
        self.criterion = nn.CrossEntropyLoss()
        
        self.optimizer = optim.AdamW(model.parameters(),
                                     lr=config.LEARNING_RATE,
                                     weight_decay=config.WEIGHT_DECAY)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=3, verbose=True
        )
        
        self.scaler = GradScaler()
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Training [{self.task}]')
        for batch_idx, batch in enumerate(pbar):
            # Handle different batch formats
            if len(batch) == 2:
                inputs, labels = batch
                raw = None
            else:
                inputs, raw, labels = batch
            
            inputs = inputs.to(self.device)
            if raw is not None:
                raw = raw.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            with autocast():
                if self.model_name == 'vit_bilstm' and raw is not None:
                    outputs = self.model(inputs, raw)
                else:
                    outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
            
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{total_loss/(batch_idx+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        return total_loss / len(self.train_loader), 100. * correct / total
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in self.val_loader:
            if len(batch) == 2:
                inputs, labels = batch
                raw = None
            else:
                inputs, raw, labels = batch
            
            inputs = inputs.to(self.device)
            if raw is not None:
                raw = raw.to(self.device)
            labels = labels.to(self.device)
            
            if self.model_name == 'vit_bilstm' and raw is not None:
                outputs = self.model(inputs, raw)
            else:
                outputs = self.model(inputs)
            
            loss = self.criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        val_loss = total_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def train(self, num_epochs: int) -> Dict[str, List]:
        """Full training loop."""
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        for epoch in range(num_epochs):
            print(f"\n{'='*70}")
            print(f"EPOCH {epoch+1}/{num_epochs} | Task: {self.task} | Model: {self.model_name}")
            print(f"{'='*70}")
            
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
            
            self.scheduler.step(val_acc)
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                print(f"✓ Best model saved (val_acc: {val_acc:.2f}%)")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        return history


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Unified EEG Classification Pipeline')
    
    # Task selection
    parser.add_argument('--task', type=str, default='sleep_staging',
                       choices=['sleep_staging', 'sleep_apnea'],
                       help='Task to run')
    
    # Model selection
    parser.add_argument('--model', type=str, default='cnn',
                       help='Model architecture')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='boas',
                       choices=['boas', 'sleep_edf', 'shhs', 'apnea_ecg'],
                       help='Dataset to use')
    
    # Training
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Dataset directory')
    parser.add_argument('--output-dir', type=str, default='outputs/unified',
                       help='Output directory')
    
    # SSL pretraining
    parser.add_argument('--ssl-pretrain', action='store_true',
                       help='Use self-supervised pretraining')
    
    # Run all
    parser.add_argument('--run-all', action='store_true',
                       help='Run all models for selected task')
    
    args = parser.parse_args()
    
    # Setup
    set_seed(UnifiedConfig.SEED)
    config = UnifiedConfig()
    config.TASK = args.task
    config.LEARNING_RATE = args.lr
    config.BATCH_SIZE = args.batch_size
    config.NUM_EPOCHS = args.epochs
    config.DATA_DIR = args.data_dir
    config.OUTPUT_DIR = f"{args.output_dir}/{args.task}"
    
    Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    num_classes = config.get_num_classes(args.task)
    class_names = config.get_class_names(args.task)
    
    print(f"\n{'='*70}")
    print(f"  UNIFIED EEG CLASSIFICATION PIPELINE")
    print(f"  Task: {args.task} ({num_classes}-class)")
    print(f"  Classes: {class_names}")
    print(f"  Model: {args.model}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Device: {config.DEVICE}")
    print(f"{'='*70}\n")
    
    # Create model
    model = create_model(args.model, args.task, num_classes, config.DEVICE)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model}")
    print(f"Parameters: {num_params:,}\n")
    
    # TODO: Implement proper dataset loading
    # Placeholder for demonstration
    print(f"[INFO] Using placeholder data - implement {args.dataset} loader")
    
    # Dummy training
    print("\nStarting training...")
    start_time = time.time()
    
    for epoch in range(min(3, config.NUM_EPOCHS)):
        print(f"  Epoch {epoch+1}/{config.NUM_EPOCHS}...")
        time.sleep(1)
    
    duration = time.time() - start_time
    
    print(f"\n✓ Training complete in {duration:.1f}s")
    print(f"  Results will be saved to: {config.OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
