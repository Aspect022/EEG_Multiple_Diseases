"""
Self-supervised pretraining loop.

Implements training for MAE (Masked Autoencoder) pretraining.
"""

import os
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler


@dataclass
class SSLConfig:
    """Configuration for SSL pretraining."""
    
    # Experiment
    experiment_name: str = 'ssl_pretrain'
    output_dir: str = './sleep_apnea/checkpoints'
    seed: int = 42
    
    # Pretraining
    epochs: int = 100
    learning_rate: float = 1.5e-4
    weight_decay: float = 1.0e-4
    batch_size: int = 128
    warmup_epochs: int = 10
    
    # MAE-specific
    mask_ratio: float = 0.75
    
    # Regularization
    mixed_precision: bool = True
    gradient_clip: float = 1.0
    
    # Checkpointing
    save_every: int = 10
    
    # Device
    device: str = 'auto'


class SSLTrainer:
    """
    Trainer for self-supervised pretraining (MAE).
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: SSLConfig,
        train_loader: DataLoader,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        
        # Setup device
        if config.device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(config.device)
        
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        # Setup scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.epochs - self.config.warmup_epochs,
        )
        
        # Mixed precision
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # Output directory
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history = {'loss': []}
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for x, _ in self.train_loader:
            x = x.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed precision
            if self.scaler is not None:
                with autocast('cuda'):
                    loss, _ = self.model(x)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss, _ = self.model(x)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def fit(self) -> Dict[str, Any]:
        """Full pretraining loop."""
        print(f"\n{'='*60}")
        print(f"  SSL Pretraining: {self.config.experiment_name}")
        print(f"  Device: {self.device}")
        print(f"  Epochs: {self.config.epochs}")
        print(f"  Mask ratio: {self.config.mask_ratio}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            # Train
            avg_loss = self.train_epoch()
            self.history['loss'].append(avg_loss)
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{self.config.epochs} - Loss: {avg_loss:.4f}")
            
            # Scheduler step
            if self.scheduler is not None and epoch >= self.config.warmup_epochs:
                self.scheduler.step()
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self._save_checkpoint(epoch, f'epoch_{epoch+1}')
        
        # Save final checkpoint
        self._save_checkpoint(self.config.epochs - 1, 'final')
        
        duration = time.time() - start_time
        
        # Save history
        history_path = self.output_dir / 'pretrain_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"  Pretraining complete in {duration:.1f}s")
        print(f"  Final loss: {self.history['loss'][-1]:.4f}")
        print(f"{'='*60}\n")
        
        return {
            'config': vars(self.config),
            'history': self.history,
            'duration_seconds': duration,
        }
    
    def _save_checkpoint(self, epoch: int, name: str):
        """Save checkpoint."""
        checkpoint_path = self.output_dir / f'pretrain_{name}.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': vars(self.config),
        }, checkpoint_path)
        print(f"  💾 Checkpoint saved: {checkpoint_path}")
