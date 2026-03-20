"""
Training loop for sleep apnea classification.

Provides comprehensive training with:
- Mixed precision (AMP)
- Class imbalance handling
- Early stopping
- Checkpointing
- Comprehensive metrics
"""

import os
import json
import time
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)


@dataclass
class ApneaConfig:
    """Training configuration for apnea classification."""
    
    # Experiment
    experiment_name: str = 'apnea_experiment'
    output_dir: str = './sleep_apnea/outputs'
    seed: int = 42
    
    # Training
    epochs: int = 30
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-4
    optimizer: str = 'adamw'
    scheduler: str = 'cosine'
    warmup_epochs: int = 5
    batch_size: int = 64
    
    # Regularization
    mixed_precision: bool = True
    gradient_clip: float = 1.0
    label_smoothing: float = 0.1
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 10
    monitor_metric: str = 'val_f1_macro'
    monitor_mode: str = 'max'
    
    # Checkpointing
    save_best: bool = True
    save_every: int = 0  # 0 = disabled
    
    # Class imbalance
    use_class_weights: bool = True
    
    # Device
    device: str = 'auto'
    
    # Number of classes
    num_classes: int = 4


class ApneaTrainer:
    """
    Trainer for sleep apnea severity classification.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: ApneaConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Setup device
        if config.device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(config.device)
        
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        self.scheduler = self._create_scheduler()
        
        # Setup loss with class weights
        self.criterion = self._create_criterion()
        
        # Mixed precision
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # Early stopping
        self.best_metric = -float('inf') if config.monitor_mode == 'max' else float('inf')
        self.patience_counter = 0
        
        # Output directory
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'val_f1_macro': [],
        }
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
        if self.config.optimizer == 'adamw':
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == 'adam':
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _create_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler."""
        if self.config.scheduler == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs - self.config.warmup_epochs,
            )
        elif self.config.scheduler == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.1,
            )
        else:
            return None
    
    def _create_criterion(self) -> nn.Module:
        """Create loss function with class weights."""
        if self.config.use_class_weights:
            # Compute class weights from training data
            labels = []
            for _, y in self.train_loader:
                labels.extend(y.tolist())
            
            class_counts = np.bincount(labels, minlength=self.config.num_classes)
            class_weights = 1.0 / (class_counts + 1e-6)
            class_weights = class_weights / class_weights.mean()
            
            return nn.CrossEntropyLoss(
                weight=torch.FloatTensor(class_weights).to(self.device),
                label_smoothing=self.config.label_smoothing,
            )
        else:
            return nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        for batch_idx, (x, y) in enumerate(self.train_loader):
            x = x.to(self.device)
            y = y.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed precision
            if self.scaler is not None:
                with autocast('cuda'):
                    outputs = self.model(x)
                    loss = self.criterion(outputs, y)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.optimizer.step()
            
            total_loss += loss.item()
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
        
        # Compute metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        for x, y in self.val_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            
            outputs = self.model(x)
            loss = self.criterion(outputs, y)
            
            total_loss += loss.item()
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())
        
        # Compute metrics
        avg_loss = total_loss / len(self.val_loader)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1_macro': f1_score(all_labels, all_preds, average='macro'),
            'f1_weighted': f1_score(all_labels, all_preds, average='weighted'),
            'precision': precision_score(all_labels, all_preds, average='macro'),
            'recall': recall_score(all_labels, all_preds, average='macro'),
        }
        
        # AUC-ROC (multi-class)
        try:
            metrics['auc_roc'] = roc_auc_score(
                all_labels, all_probs,
                multi_class='ovr', average='macro'
            )
        except:
            metrics['auc_roc'] = 0.0
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(all_labels, all_preds).tolist()
        
        return metrics
    
    def fit(self) -> Dict[str, Any]:
        """Full training loop."""
        print(f"\n{'='*60}")
        print(f"  Training: {self.config.experiment_name}")
        print(f"  Device: {self.device}")
        print(f"  Epochs: {self.config.epochs}")
        print(f"  Output: {self.output_dir}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_f1_macro'].append(val_metrics['f1_macro'])
            
            # Print progress
            print(f"Epoch {epoch+1}/{self.config.epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.4f}")
            print(f"  Val Loss:   {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
            print(f"  Val F1:     {val_metrics['f1_macro']:.4f} | Val AUC: {val_metrics['auc_roc']:.4f}")
            
            # Scheduler step
            if self.scheduler is not None and epoch >= self.config.warmup_epochs:
                self.scheduler.step()
            
            # Check for improvement
            current_metric = val_metrics[self.config.monitor_metric]
            is_improved = (
                (self.config.monitor_mode == 'max' and current_metric > self.best_metric) or
                (self.config.monitor_mode == 'min' and current_metric < self.best_metric)
            )
            
            if is_improved:
                self.best_metric = current_metric
                self.patience_counter = 0
                
                # Save best checkpoint
                if self.config.save_best:
                    self._save_checkpoint(epoch, 'best')
                print(f"  ✨ New best: {self.config.monitor_metric} = {current_metric:.4f}")
            else:
                self.patience_counter += 1
            
            # Save periodic checkpoint
            if self.config.save_every > 0 and (epoch + 1) % self.config.save_every == 0:
                self._save_checkpoint(epoch, f'epoch_{epoch+1}')
            
            # Early stopping
            if self.config.early_stopping and self.patience_counter >= self.config.patience:
                print(f"\n  ⏹️  Early stopping at epoch {epoch+1}")
                break
        
        # Final test evaluation
        test_results = None
        if self.test_loader is not None:
            test_results = self.evaluate(self.test_loader)
        
        duration = time.time() - start_time
        
        # Save final results
        results = {
            'config': vars(self.config),
            'history': self.history,
            'best_metric': float(self.best_metric),
            'test_results': test_results,
            'duration_seconds': duration,
        }
        
        results_path = self.output_dir / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"  Training complete in {duration:.1f}s")
        print(f"  Best {self.config.monitor_metric}: {self.best_metric:.4f}")
        print(f"  Results saved to {results_path}")
        print(f"{'='*60}\n")
        
        return results
    
    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate on test set."""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        for x, y in test_loader:
            x = x.to(self.device)
            outputs = self.model(x)
            
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(y.numpy())
            all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        return {
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1_macro': f1_score(all_labels, all_preds, average='macro'),
            'f1_weighted': f1_score(all_labels, all_preds, average='weighted'),
            'precision': precision_score(all_labels, all_preds, average='macro'),
            'recall': recall_score(all_labels, all_preds, average='macro'),
            'auc_roc': roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro'),
            'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist(),
        }
    
    def _save_checkpoint(self, epoch: int, name: str):
        """Save model checkpoint."""
        checkpoint_path = self.output_dir / f'checkpoint_{name}.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'config': vars(self.config),
        }, checkpoint_path)
        print(f"  💾 Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_metric = checkpoint['best_metric']
        print(f"  📥 Loaded checkpoint: {checkpoint_path}")
