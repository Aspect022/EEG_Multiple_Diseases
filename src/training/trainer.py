"""
Training Utilities for ECG Classification Models.

This module provides trainers for different model types:
- StandardTrainer: For classical CNNs and Transformers
- SNNTrainer: For Spiking Neural Networks (requires time-step loop)

Supports mixed precision training, gradient accumulation, and various
learning rate schedules.
"""

import os
import time
from typing import Optional, Dict, Any, Callable, List, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Optimization
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    optimizer: str = 'adamw'  # 'adam', 'adamw', 'sgd'
    scheduler: str = 'cosine'  # 'cosine', 'step', 'plateau', 'none'
    warmup_epochs: int = 5
    
    # Training tricks
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.1
    
    # Checkpointing
    save_dir: str = './checkpoints'
    save_every: int = 10
    save_best: bool = True
    
    # Logging
    log_every: int = 10
    eval_every: int = 1
    
    # Early stopping
    early_stopping_patience: int = 20
    early_stopping_metric: str = 'val_acc'  # 'val_loss' or 'val_acc'
    
    # Device
    device: str = 'cuda'


@dataclass
class TrainingState:
    """Mutable training state."""
    
    epoch: int = 0
    global_step: int = 0
    best_metric: float = 0.0
    patience_counter: int = 0
    history: Dict[str, List[float]] = field(default_factory=lambda: {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'lr': [],
    })


class StandardTrainer:
    """
    Standard trainer for CNN and Transformer models.
    
    Handles:
    - Mixed precision training
    - Learning rate scheduling with warmup
    - Gradient clipping and accumulation
    - Checkpointing and early stopping
    - Metric logging
    
    Args:
        model: PyTorch model to train.
        config: TrainingConfig instance.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        criterion: Loss function. Default: CrossEntropyLoss.
        
    Example:
        >>> trainer = StandardTrainer(model, config, train_loader, val_loader)
        >>> trainer.train()
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: Optional[nn.Module] = None,
    ):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Device setup
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        
        # Loss function
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        else:
            self.criterion = criterion
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # State
        self.state = TrainingState()
        
        # Create save directory
        Path(config.save_dir).mkdir(parents=True, exist_ok=True)
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on config."""
        if self.config.optimizer == 'adam':
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == 'adamw':
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == 'sgd':
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _create_scheduler(self) -> Optional[_LRScheduler]:
        """Create learning rate scheduler."""
        if self.config.scheduler == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs - self.config.warmup_epochs,
            )
        elif self.config.scheduler == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1,
            )
        elif self.config.scheduler == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max' if 'acc' in self.config.early_stopping_metric else 'min',
                factor=0.5,
                patience=10,
            )
        return None
    
    def _warmup_lr(self, epoch: int, step: int, total_steps: int):
        """Apply linear warmup to learning rate."""
        if epoch >= self.config.warmup_epochs:
            return
        
        warmup_steps = self.config.warmup_epochs * total_steps
        current_step = epoch * total_steps + step
        
        lr_scale = min(1.0, current_step / warmup_steps)
        for pg in self.optimizer.param_groups:
            pg['lr'] = self.config.learning_rate * lr_scale
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average_loss, accuracy).
        """
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        loader = tqdm(self.train_loader, desc=f"Epoch {self.state.epoch}") if HAS_TQDM else self.train_loader
        
        for step, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Warmup
            self._warmup_lr(self.state.epoch, step, len(self.train_loader))
            
            # Forward pass
            if self.config.mixed_precision and self.scaler is not None:
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    loss = loss / self.config.gradient_accumulation_steps
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.state.global_step += 1
            
            # Metrics
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Logging
            if HAS_TQDM and step % self.config.log_every == 0:
                loader.set_postfix({
                    'loss': total_loss / (step + 1),
                    'acc': 100. * correct / total,
                })
        
        return total_loss / len(self.train_loader), 100. * correct / total
    
    @torch.no_grad()
    def evaluate(self) -> Tuple[float, float]:
        """
        Evaluate on validation set.
        
        Returns:
            Tuple of (average_loss, accuracy).
        """
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in self.val_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        return total_loss / len(self.val_loader), 100. * correct / total
    
    def save_checkpoint(self, filename: str = 'checkpoint.pt'):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.state.epoch,
            'global_step': self.state.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_metric': self.state.best_metric,
            'history': self.state.history,
            'config': self.config,
        }
        
        path = Path(self.config.save_dir) / filename
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if checkpoint['scaler_state_dict'] and self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.state.epoch = checkpoint['epoch']
        self.state.global_step = checkpoint['global_step']
        self.state.best_metric = checkpoint['best_metric']
        self.state.history = checkpoint['history']
        
        print(f"Loaded checkpoint from {path} (epoch {self.state.epoch})")
    
    def train(self) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Returns:
            Training history dictionary.
        """
        print(f"Training on {self.device}")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.state.epoch, self.config.epochs):
            self.state.epoch = epoch
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Evaluate
            if (epoch + 1) % self.config.eval_every == 0:
                val_loss, val_acc = self.evaluate()
            else:
                val_loss, val_acc = 0.0, 0.0
            
            # Learning rate scheduling
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler is not None and epoch >= self.config.warmup_epochs:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    metric = val_acc if 'acc' in self.config.early_stopping_metric else val_loss
                    self.scheduler.step(metric)
                else:
                    self.scheduler.step()
            
            # Log metrics
            self.state.history['train_loss'].append(train_loss)
            self.state.history['train_acc'].append(train_acc)
            self.state.history['val_loss'].append(val_loss)
            self.state.history['val_acc'].append(val_acc)
            self.state.history['lr'].append(current_lr)
            
            print(f"Epoch {epoch+1}/{self.config.epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
                  f"LR: {current_lr:.6f}")
            
            # Checkpointing
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')
            
            # Best model
            metric = val_acc if 'acc' in self.config.early_stopping_metric else -val_loss
            if self.config.save_best and metric > self.state.best_metric:
                self.state.best_metric = metric
                self.save_checkpoint('best_model.pt')
                self.state.patience_counter = 0
            else:
                self.state.patience_counter += 1
            
            # Early stopping
            if self.state.patience_counter >= self.config.early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        return self.state.history


class SNNTrainer(StandardTrainer):
    """
    Trainer specialized for Spiking Neural Networks.
    
    Key differences from StandardTrainer:
    - Handles time-step based forward passes
    - Supports spike-based loss functions
    - Tracks spike statistics
    
    Args:
        model: SNN model (must have num_timesteps attribute).
        config: TrainingConfig instance.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        loss_type: SNN loss type ('rate', 'spike_count', 'cross_entropy').
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_type: str = 'cross_entropy',
    ):
        super().__init__(model, config, train_loader, val_loader)
        
        self.loss_type = loss_type
        
        # Override criterion for SNN-specific losses
        if loss_type == 'rate':
            self.criterion = self._rate_loss
        elif loss_type == 'spike_count':
            self.criterion = self._spike_count_loss
        # Otherwise use parent's CrossEntropyLoss
    
    def _rate_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Rate-based loss for SNNs.
        
        Uses MSE between output spike rates and one-hot targets.
        """
        # Outputs are spike rates: (B, num_classes)
        targets_onehot = F.one_hot(targets, num_classes=outputs.shape[-1]).float()
        return F.mse_loss(outputs, targets_onehot)
    
    def _spike_count_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Spike count loss for SNNs.
        
        Cross-entropy on cumulative spike counts.
        """
        return F.cross_entropy(outputs, targets)
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch (SNN-aware).
        
        Returns:
            Tuple of (average_loss, accuracy).
        """
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        loader = tqdm(self.train_loader, desc=f"Epoch {self.state.epoch}") if HAS_TQDM else self.train_loader
        
        for step, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            self._warmup_lr(self.state.epoch, step, len(self.train_loader))
            
            self.optimizer.zero_grad()
            
            # SNN forward pass (model handles time-steps internally)
            if self.config.mixed_precision and self.scaler is not None:
                with autocast():
                    outputs = self.model(inputs)  # Returns average spike rate
                    loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
            
            # Backward
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
            
            self.state.global_step += 1
            
            # Metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if HAS_TQDM and step % self.config.log_every == 0:
                loader.set_postfix({
                    'loss': total_loss / (step + 1),
                    'acc': 100. * correct / total,
                })
        
        return total_loss / len(self.train_loader), 100. * correct / total


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Optional[TrainingConfig] = None,
    is_snn: bool = False,
    **kwargs,
) -> Dict[str, List[float]]:
    """
    Convenience function to train a model.
    
    Args:
        model: Model to train.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        config: Training configuration (uses defaults if None).
        is_snn: Whether the model is an SNN.
        **kwargs: Override config parameters.
        
    Returns:
        Training history.
    """
    if config is None:
        config = TrainingConfig(**kwargs)
    else:
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    if is_snn:
        trainer = SNNTrainer(model, config, train_loader, val_loader)
    else:
        trainer = StandardTrainer(model, config, train_loader, val_loader)
    
    return trainer.train()
