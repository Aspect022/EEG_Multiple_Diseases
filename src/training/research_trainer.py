"""
Research-Grade Training Utilities for ECG Classification.

Provides a 5-fold cross-validation training loop with:
- FLOPs / parameter count computation
- Comprehensive per-epoch metrics (Accuracy, F1, AUC-ROC, Precision, Recall)
- Enhanced checkpointing (best per fold, periodic, resume)
- Early stopping (configurable patience & metric)
- Full reproducibility seeding
- Paper-ready results aggregation (mean Â± std)
"""

import os
import gc
import csv
import json
import copy
import time
import random
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any, Callable
from dataclasses import dataclass, field, asdict

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
    classification_report,
)

from src.evaluation.metrics import compute_all_metrics

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Graceful W&B integration (no crash if offline or missing)
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# Graceful TensorBoard integration
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
@dataclass
class ResearchConfig:
    """Full configuration for research experiments."""

    # --- Experiment ---
    experiment_name: str = 'apnea_ecg_experiment'
    output_dir: str = './experiments/results'
    seed: int = 42

    # --- Training ---
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    optimizer: str = 'adamw'              # adam | adamw | sgd
    scheduler: str = 'cosine'            # cosine | step | plateau | none
    warmup_epochs: int = 3
    batch_size: int = 32
    num_workers: int = 4

    # --- Training tricks ---
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.0         # 0 for binary tasks usually

    # --- Cross-validation ---
    n_folds: int = 5

    # --- Checkpointing ---
    save_best: bool = True
    save_every: int = 0                  # 0 = disabled; N = every N epochs

    # --- Early stopping ---
    early_stopping: bool = True
    patience: int = 15
    monitor_metric: str = 'val_f1'       # val_loss | val_f1 | val_auc | val_acc
    monitor_mode: str = 'max'            # min | max

    # --- Device ---
    device: str = 'auto'                 # auto | cuda | cpu

    # --- Number of classes ---
    num_classes: int = 2

    # --- W&B ---
    wandb_project: str = 'eeg-sleep-staging'
    wandb_tags: List[str] = field(default_factory=list)
    wandb_config_extra: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Seeding
# --------------------------------------------------------------------------
def seed_everything(seed: int = 42):
    """Set seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True   # Keep True for speed (A100)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    num_classes: int = 2,
) -> Dict[str, float]:
    """
    Compute a comprehensive set of classification metrics.

    Returns dict with: accuracy, f1_macro, f1_weighted, precision, recall,
    specificity, auc_roc.
    """
    metrics: Dict[str, float] = {}

    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['f1_macro'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    metrics['f1_weighted'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    metrics['precision'] = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
    metrics['recall'] = float(recall_score(y_true, y_pred, average='macro', zero_division=0))

    # Specificity (for binary)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    if num_classes == 2 and cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['specificity'] = float(tn / (tn + fp + 1e-8))
        metrics['sensitivity'] = float(tp / (tp + fn + 1e-8))
    else:
        metrics['specificity'] = 0.0
        metrics['sensitivity'] = 0.0

    # AUC-ROC
    if y_prob is not None:
        try:
            if num_classes == 2:
                metrics['auc_roc'] = float(roc_auc_score(y_true, y_prob[:, 1]))
            else:
                metrics['auc_roc'] = float(roc_auc_score(
                    y_true, y_prob, multi_class='ovr', average='macro',
                ))
        except ValueError:
            metrics['auc_roc'] = 0.0
    else:
        metrics['auc_roc'] = 0.0

    metrics['confusion_matrix'] = cm.tolist()

    return metrics


# --------------------------------------------------------------------------
# FLOPs helper
# --------------------------------------------------------------------------
def compute_flops(model: nn.Module, input_shape: Tuple[int, ...], device: torch.device) -> Dict[str, Any]:
    """Compute FLOPs, MACs, and parameter counts."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    flops, macs = 0, 0

    # Try fvcore
    try:
        from fvcore.nn import FlopCountAnalysis
        dummy = torch.randn(input_shape, device=device)
        fa = FlopCountAnalysis(model, dummy)
        flops = int(fa.total())
        macs = flops // 2
    except Exception:
        pass

    # Fallback: thop
    if flops == 0:
        try:
            from thop import profile
            dummy = torch.randn(input_shape, device=device)
            macs_val, _ = profile(model, inputs=(dummy,), verbose=False)
            macs = int(macs_val)
            flops = macs * 2
        except Exception:
            pass

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'flops': flops,
        'macs': macs,
    }


def _fmt(n: int) -> str:
    """Human-readable large number format."""
    if n >= 1e9:
        return f"{n / 1e9:.2f}G"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.2f}K"
    return str(n)


# --------------------------------------------------------------------------
# Single-fold trainer
# --------------------------------------------------------------------------
class FoldTrainer:
    """
    Train a single model for one fold, tracking comprehensive metrics.
    """

    def __init__(
        self,
        model: nn.Module,
        config: ResearchConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        fold: int = 0,
        class_weights: Optional[torch.Tensor] = None,
    ):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.fold = fold

        # Device
        if config.device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(config.device)

        self.model = model.to(self.device)

        # Loss
        weight = class_weights.to(self.device) if class_weights is not None else None
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            label_smoothing=config.label_smoothing,
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'val_f1': [], 'val_auc': [],
            'val_precision': [], 'val_recall': [],
            'val_specificity': [], 'val_sensitivity': [],
            'lr': [],
        }

        # Early stopping state
        self.best_metric = -float('inf') if config.monitor_mode == 'max' else float('inf')
        self.patience_counter = 0
        self.best_model_state = None

        # Output
        self.fold_dir = Path(config.output_dir) / config.experiment_name / f'fold_{fold}'
        self.fold_dir.mkdir(parents=True, exist_ok=True)

    # -- Optimizer & Scheduler ------------------------------------------
    def _build_optimizer(self) -> torch.optim.Optimizer:
        c = self.config
        if c.optimizer == 'adamw':
            return torch.optim.AdamW(self.model.parameters(), lr=c.learning_rate, weight_decay=c.weight_decay)
        if c.optimizer == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=c.learning_rate, weight_decay=c.weight_decay)
        if c.optimizer == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=c.learning_rate, momentum=0.9, weight_decay=c.weight_decay)
        raise ValueError(f"Unknown optimizer: {c.optimizer}")

    def _build_scheduler(self):
        c = self.config
        if c.scheduler == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max(c.epochs - c.warmup_epochs, 1))
        if c.scheduler == 'step':
            return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)
        if c.scheduler == 'plateau':
            mode = 'max' if c.monitor_mode == 'max' else 'min'
            return torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode=mode, factor=0.5, patience=7)
        return None

    @staticmethod
    def _unwrap_outputs(outputs):
        """Allow models to return logits or (logits, aux...)."""
        if isinstance(outputs, (tuple, list)):
            return outputs[0]
        return outputs

    def _warmup_lr(self, epoch: int, step: int, steps_per_epoch: int):
        if epoch >= self.config.warmup_epochs:
            return
        total_warmup_steps = self.config.warmup_epochs * steps_per_epoch
        current = epoch * steps_per_epoch + step
        scale = min(1.0, current / max(total_warmup_steps, 1))
        for pg in self.optimizer.param_groups:
            pg['lr'] = self.config.learning_rate * scale

    # -- Train one epoch ------------------------------------------------
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        steps = len(self.train_loader)

        loader = tqdm(self.train_loader, desc=f"  [Fold {self.fold}] Epoch {epoch+1}", leave=False) if HAS_TQDM else self.train_loader

        for step, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self._warmup_lr(epoch, step, steps)

            # Forward
            if self.scaler is not None:
                with autocast('cuda'):
                    outputs = self._unwrap_outputs(self.model(inputs))
                    loss = self.criterion(outputs, targets)
                    
                    if hasattr(self.model, 'reg_loss'):
                        reg = self.model.reg_loss
                        if isinstance(reg, torch.Tensor) and reg.requires_grad:
                            loss = loss + reg
                            
                    loss = loss / self.config.gradient_accumulation_steps
                self.scaler.scale(loss).backward()
            else:
                outputs = self._unwrap_outputs(self.model(inputs))
                loss = self.criterion(outputs, targets)
                
                if hasattr(self.model, 'reg_loss'):
                    reg = self.model.reg_loss
                    if isinstance(reg, torch.Tensor) and reg.requires_grad:
                        loss = loss + reg
                        
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()

            # Step
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.config.gradient_accumulation_steps
            _, preds = outputs.max(1)
            total += targets.size(0)
            correct += preds.eq(targets).sum().item()

            if HAS_TQDM and step % 20 == 0:
                loader.set_postfix(loss=total_loss / (step + 1), acc=f"{100.0 * correct / total:.1f}%")

        avg_loss = total_loss / steps
        avg_acc = 100.0 * correct / total
        return avg_loss, avg_acc

    # -- Evaluate -------------------------------------------------------
    @torch.no_grad()
    def evaluate(self) -> Dict[str, Any]:
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []

        for inputs, targets in self.val_loader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            outputs = self._unwrap_outputs(self.model(inputs))
            loss = self.criterion(outputs, targets)
            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

        avg_loss = total_loss / max(len(self.val_loader), 1)
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)

        metrics = compute_all_metrics(y_true, y_pred, y_prob)
        metrics['val_loss'] = avg_loss

        return metrics

    # -- Checkpoint -----------------------------------------------------
    def save_checkpoint(self, filename: str, extra: Optional[Dict] = None):
        ckpt = {
            'fold': self.fold,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_metric': self.best_metric,
            'patience_counter': self.patience_counter,
            'history': self.history,
            'config': asdict(self.config),
        }
        if extra:
            ckpt.update(extra)
        torch.save(ckpt, self.fold_dir / filename)

    # -- Full training loop for one fold --------------------------------
    def fit(self) -> Dict[str, Any]:
        """
        Run full training loop for this fold.

        Returns:
            Dict with final metrics and history.
        """
        print(f"\n{'='*60}")
        print(f"  FOLD {self.fold}  |  Device: {self.device}  |  Epochs: {self.config.epochs}")
        print(f"{'='*60}")

        for epoch in range(self.config.epochs):
            t0 = time.time()

            # Train
            train_loss, train_acc = self.train_epoch(epoch)

            # Validate
            val_metrics = self.evaluate()
            val_loss = val_metrics['val_loss']
            val_acc = val_metrics['accuracy'] * 100.0
            val_f1 = val_metrics['f1_macro']
            val_auc = val_metrics['auc_roc']

            # Scheduler
            lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler is not None and epoch >= self.config.warmup_epochs:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get(self.config.monitor_metric.replace('val_', ''), val_loss))
                else:
                    self.scheduler.step()

            # Log
            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            self.history['val_auc'].append(val_auc)
            self.history['val_precision'].append(val_metrics.get('precision_macro', val_metrics.get('precision', 0.0)))
            self.history['val_recall'].append(val_metrics.get('recall_macro', val_metrics.get('recall', 0.0)))
            self.history['val_specificity'].append(val_metrics.get('specificity', 0.0))
            self.history['val_sensitivity'].append(val_metrics.get('sensitivity', 0.0))
            self.history['lr'].append(lr)

            elapsed = time.time() - t0
            print(f"  Epoch {epoch+1:3d}/{self.config.epochs} | "
                  f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.1f}% | "
                  f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.1f}%  F1: {val_f1:.4f}  AUC: {val_auc:.4f} | "
                  f"LR: {lr:.2e} | {elapsed:.1f}s")

            # W&B logging (graceful)
            if self._wandb_active:
                try:
                    wandb.log({
                        'epoch': epoch + 1,
                        'train/loss': train_loss,
                        'train/acc': train_acc,
                        'val/loss': val_loss,
                        'val/acc': val_acc,
                        'val/f1': val_f1,
                        'val/auc': val_auc,
                        'val/precision': val_metrics.get('precision_macro', 0),
                        'val/recall': val_metrics.get('recall_macro', 0),
                        'val/specificity': val_metrics.get('specificity', 0),
                        'lr': lr,
                        'epoch_time_s': elapsed,
                    })
                except Exception:
                    pass

            # TensorBoard logging (graceful)
            if self._tb_writer:
                try:
                    self._tb_writer.add_scalars('loss', {'train': train_loss, 'val': val_loss}, epoch + 1)
                    self._tb_writer.add_scalars('accuracy', {'train': train_acc, 'val': val_acc}, epoch + 1)
                    self._tb_writer.add_scalar('val/f1', val_f1, epoch + 1)
                    self._tb_writer.add_scalar('val/auc', val_auc, epoch + 1)
                    self._tb_writer.add_scalar('val/precision', val_metrics.get('precision_macro', 0), epoch + 1)
                    self._tb_writer.add_scalar('val/recall', val_metrics.get('recall_macro', 0), epoch + 1)
                    self._tb_writer.add_scalar('val/specificity', val_metrics.get('specificity', 0), epoch + 1)
                    self._tb_writer.add_scalar('val/sensitivity', val_metrics.get('sensitivity', 0), epoch + 1)
                    self._tb_writer.add_scalar('lr', lr, epoch + 1)
                    self._tb_writer.add_scalar('epoch_time_s', elapsed, epoch + 1)
                except Exception:
                    pass

            # Early stopping
            metric_value = self._get_monitor_value(val_metrics)
            improved = self._check_improvement(metric_value)

            if improved:
                self.best_metric = metric_value
                self.patience_counter = 0
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                if self.config.save_best:
                    self.save_checkpoint('best_model.pt', extra={'epoch': epoch})
                    print(f"    âœ“ Best model saved (val_{self.config.monitor_metric}: {metric_value:.4f})")
            else:
                self.patience_counter += 1

            # Periodic checkpoint
            if self.config.save_every > 0 and (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt', extra={'epoch': epoch})

            if self.config.early_stopping and self.patience_counter >= self.config.patience:
                print(f"  âœ— Early stopping at epoch {epoch+1} (patience={self.config.patience})")
                break

        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        # Final evaluation with best model
        final_metrics = self.evaluate()
        final_metrics['best_epoch'] = int(np.argmax(self.history['val_f1']) + 1) if self.history['val_f1'] else 0

        # Save history CSV
        self._save_history_csv()

        # Cleanup: close W&B and TensorBoard
        if self._wandb_active:
            try:
                # Log comprehensive final metrics
                wandb.log({
                    'final/accuracy': final_metrics.get('accuracy', 0),
                    'final/balanced_accuracy': final_metrics.get('balanced_accuracy', 0),
                    'final/f1_macro': final_metrics.get('f1_macro', 0),
                    'final/f1_weighted': final_metrics.get('f1_weighted', 0),
                    'final/precision_macro': final_metrics.get('precision_macro', 0),
                    'final/recall_macro': final_metrics.get('recall_macro', 0),
                    'final/specificity': final_metrics.get('specificity', 0),
                    'final/sensitivity': final_metrics.get('sensitivity', 0),
                    'final/auc_roc': final_metrics.get('auc_roc', 0),
                    'final/MCC': final_metrics.get('MCC', 0),
                    'final/cohens_kappa': final_metrics.get('cohens_kappa', 0),
                    'final/best_epoch': final_metrics.get('best_epoch', 0),
                })

                # Log confusion matrix if available
                cm = final_metrics.get('confusion_matrix')
                class_names = final_metrics.get('class_names', [])
                if cm is not None and class_names:
                    try:
                        wandb.log({
                            'confusion_matrix': wandb.plot.confusion_matrix(
                                probs=None,
                                y_true=None,
                                preds=None,
                                class_names=class_names,
                            )
                        })
                    except Exception:
                        pass

                # Summary metrics (appear in W&B run summary table)
                wandb.run.summary['best_accuracy'] = final_metrics.get('accuracy', 0)
                wandb.run.summary['best_f1'] = final_metrics.get('f1_macro', 0)
                wandb.run.summary['best_auc'] = final_metrics.get('auc_roc', 0)
                wandb.run.summary['best_epoch'] = final_metrics.get('best_epoch', 0)

                wandb.finish()
            except Exception:
                try:
                    wandb.finish()
                except Exception:
                    pass

        if self._tb_writer:
            try:
                # Add hparams summary
                self._tb_writer.add_hparams(
                    {'lr': self.config.learning_rate,
                     'batch_size': self.config.batch_size,
                     'epochs': self.config.epochs,
                     'optimizer': self.config.optimizer},
                    {'hparam/accuracy': final_metrics.get('accuracy', 0),
                     'hparam/f1': final_metrics.get('f1_macro', 0),
                     'hparam/auc': final_metrics.get('auc_roc', 0)},
                )
                self._tb_writer.close()
            except Exception:
                try:
                    self._tb_writer.close()
                except Exception:
                    pass

        return final_metrics

    def _get_monitor_value(self, val_metrics: Dict) -> float:
        key = self.config.monitor_metric
        if key == 'val_loss':
            return val_metrics['val_loss']
        if key == 'val_f1':
            return val_metrics['f1_macro']
        if key == 'val_auc':
            return val_metrics['auc_roc']
        if key == 'val_acc':
            return val_metrics['accuracy']
        return val_metrics.get(key, 0.0)

    def _check_improvement(self, value: float) -> bool:
        if self.config.monitor_mode == 'max':
            return value > self.best_metric
        return value < self.best_metric

    def _save_history_csv(self):
        csv_path = self.fold_dir / 'training_history.csv'
        keys = [k for k in self.history if k != 'epoch']
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch'] + keys)
            for i, ep in enumerate(self.history['epoch']):
                row = [ep] + [self.history[k][i] for k in keys]
                writer.writerow(row)


# --------------------------------------------------------------------------
# 5-Fold Cross-Validation Runner
# --------------------------------------------------------------------------
class CrossValidationRunner:
    """
    Run 5-fold cross-validation with a model factory.

    Args:
        model_factory: Callable that returns a fresh ``nn.Module`` each time.
        dataloader_factory: Callable ``(fold) -> (train_loader, val_loader)``.
        config: ``ResearchConfig`` instance.
        class_weights: Optional tensor of class weights for loss.
    """

    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        dataloader_factory: Callable[[int], Tuple[DataLoader, DataLoader]],
        config: ResearchConfig,
        class_weights: Optional[torch.Tensor] = None,
    ):
        self.model_factory = model_factory
        self.dataloader_factory = dataloader_factory
        self.config = config
        self.class_weights = class_weights

        self.results_dir = Path(config.output_dir) / config.experiment_name
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict[str, Any]:
        """
        Execute all folds and return aggregated results.

        Returns:
            Dict with per-fold results and aggregated mean Â± std.
        """
        seed_everything(self.config.seed)

        # Compute FLOPs once with a dummy model
        print("\n" + "=" * 70)
        print("  RESEARCH EXPERIMENT: " + self.config.experiment_name)
        print("=" * 70)

        dummy_model = self.model_factory()
        device = torch.device(
            'cuda' if (self.config.device == 'auto' and torch.cuda.is_available())
            else self.config.device if self.config.device != 'auto' else 'cpu'
        )
        dummy_model = dummy_model.to(device).eval()

        # Determine input shape from the first dataloader
        train_loader, _ = self.dataloader_factory(0)
        sample_input, _ = next(iter(train_loader))
        input_shape = tuple(sample_input[:1].shape)

        flops_info = compute_flops(dummy_model, input_shape, device)
        print(f"\n  Model:       {dummy_model.__class__.__name__}")
        print(f"  Parameters:  {_fmt(flops_info['total_params'])} total, {_fmt(flops_info['trainable_params'])} trainable")
        print(f"  FLOPs:       {_fmt(flops_info['flops'])}")
        print(f"  MACs:        {_fmt(flops_info['macs'])}")
        print(f"  Input shape: {input_shape}")
        print(f"  Device:      {device}")
        print(f"  Folds:       {self.config.n_folds}")

        del dummy_model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # Run folds
        fold_results = []
        for fold in range(self.config.n_folds):
            seed_everything(self.config.seed + fold)

            model = self.model_factory()
            train_loader, val_loader = self.dataloader_factory(fold)

            trainer = FoldTrainer(
                model=model,
                config=self.config,
                train_loader=train_loader,
                val_loader=val_loader,
                fold=fold,
                class_weights=self.class_weights,
            )

            fold_metrics = trainer.fit()
            fold_results.append(fold_metrics)

            # Free memory
            del model, trainer, train_loader, val_loader
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        # Aggregate
        aggregated = self._aggregate_results(fold_results)

        # Save summary
        summary = {
            'experiment': self.config.experiment_name,
            'config': asdict(self.config),
            'flops_info': flops_info,
            'input_shape': list(input_shape),
            'per_fold': fold_results,
            'aggregated': aggregated,
        }

        summary_path = self.results_dir / 'experiment_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        # Print summary
        self._print_summary(aggregated, flops_info)

        return summary

    def _aggregate_results(self, fold_results: List[Dict]) -> Dict[str, Dict[str, float]]:
        """Compute mean Â± std across folds for each metric."""
        metric_keys = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro',
                        'specificity', 'sensitivity', 'auc_roc']
        agg = {}
        for key in metric_keys:
            values = [fr.get(key, 0.0) for fr in fold_results]
            agg[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'values': values,
            }
        return agg

    def _print_summary(self, aggregated: Dict, flops_info: Dict):
        print("\n" + "=" * 70)
        print("  CROSS-VALIDATION RESULTS SUMMARY")
        print("=" * 70)
        print(f"  {'Metric':<20s} {'Mean':>10s} {'Â± Std':>10s}  {'Per-fold values'}")
        print("  " + "-" * 66)
        for key, vals in aggregated.items():
            per_fold = ", ".join(f"{v:.4f}" for v in vals['values'])
            print(f"  {key:<20s} {vals['mean']:>10.4f} {vals['std']:>10.4f}  [{per_fold}]")
        print("  " + "-" * 66)
        print(f"  FLOPs: {_fmt(flops_info['flops'])}  |  "
              f"Params: {_fmt(flops_info['total_params'])}  |  "
              f"MACs: {_fmt(flops_info['macs'])}")
        print("=" * 70)



