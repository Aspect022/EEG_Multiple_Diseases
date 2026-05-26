import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

from .metrics import compute_all_metrics
from ..utils.experiment_logger import log_experiment

class Trainer:
    def __init__(self, model: nn.Module, task: str, config):
        self.model = model
        self.task = task
        self.config = config
        
        # Set seeds
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
            
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def _apply_eeg_augmentation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply EEG-specific augmentation during training only."""
        # 1. Gaussian noise
        if random.random() < 0.5:
            x = x + torch.randn_like(x) * 0.05
        # 2. Random temporal shift (up to 10% of window)
        if random.random() < 0.5:
            shift = random.randint(0, x.shape[-1] // 10)
            x = torch.roll(x, shift, dims=-1)
        # 3. Channel dropout (zero out one random channel)
        if random.random() < 0.3 and x.shape[1] > 1:
            ch = random.randint(0, x.shape[1] - 1)
            x[:, ch, :] = 0.0
        # 4. Amplitude scaling
        if random.random() < 0.5:
            scale = random.uniform(0.8, 1.2)
            x = x * scale
        return x

    def fit(self, train_loader, val_loader, fold: int = 0) -> dict:
        """Run the complete training protocol."""
        # Calculate class weights for imbalanced targets
        train_labels = []
        for _, _, y in train_loader:
            train_labels.extend(y.numpy())
        train_labels = np.array(train_labels)
        
        unique_labels = np.unique(train_labels)
        class_weights = compute_class_weight('balanced', classes=unique_labels, y=train_labels)
        
        # Build weighted loss criterion
        weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
        criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=self.config.label_smoothing)
        
        # Setup Optimizer
        if self.config.optimizer.lower() == 'adamw':
            optimizer = AdamW(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        else:
            optimizer = Adam(self.model.parameters(), lr=self.config.lr)
            
        # Setup Scheduler
        if self.config.scheduler == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=self.config.epochs, eta_min=1e-6)
        else:
            scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
            
        scaler = GradScaler(enabled=self.config.amp)
        
        best_val_metric = -1.0
        best_metrics = {}
        early_stop_counter = 0
        
        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            train_loss = 0.0
            
            # Epoch loop
            for x_raw, x_scal, y in train_loader:
                # Augmentation (only on raw signal)
                x_raw = self._apply_eeg_augmentation(x_raw)
                
                # Move to device
                x_raw = x_raw.to(self.device)
                x_scal = x_scal.to(self.device)
                y = y.to(self.device)
                
                optimizer.zero_grad()
                
                # Mixed precision forward pass
                with autocast(enabled=self.config.amp):
                    # proposed model returns (logits, gate) or logits depending on return_gate
                    # baselines return logits directly
                    out = self.model(x_raw, x_scal)
                    if isinstance(out, tuple):
                        logits, _ = out
                    else:
                        logits = out
                    loss = criterion(logits, y)
                    
                scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                    
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item() * x_raw.size(0)
                
            train_loss /= len(train_loader.dataset)
            
            # Step scheduler
            if self.config.scheduler == 'cosine':
                scheduler.step()
                
            # Validation
            val_metrics = self.evaluate(val_loader)
            val_metric_val = val_metrics[self.config.checkpoint_metric]
            
            if self.config.scheduler == 'plateau':
                scheduler.step(val_metric_val)
                
            # Early stopping and checkpointing
            if val_metric_val > best_val_metric:
                best_val_metric = val_metric_val
                best_metrics = val_metrics
                best_metrics['epoch_best'] = epoch
                early_stop_counter = 0
                
                # Save best checkpoint
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': val_metrics,
                }, f"results/best_model_{self.task}_fold{fold}.pt")
            else:
                early_stop_counter += 1
                
            if early_stop_counter >= self.config.early_stop_patience:
                # Early stop triggered
                break
                
        # Log to canonical results csv
        log_data = {
            'run_id': f"run_{self.task}_{fold}_{int(np.random.randint(100000))}",
            'task': self.task,
            'model': self.model.__class__.__name__,
            'dataset': 'PhysioNet' if self.task in ['sleep_apnea', 'schizophrenia'] else 'OpenNeuro/MODMA',
            'fold': fold,
            'epoch_best': best_metrics.get('epoch_best', epoch),
            'accuracy': best_metrics.get('accuracy', 0.0),
            'balanced_accuracy': best_metrics.get('balanced_accuracy', 0.0),
            'macro_f1': best_metrics.get('macro_f1', 0.0),
            'weighted_f1': best_metrics.get('weighted_f1', 0.0),
            'auc_roc': best_metrics.get('auc_roc', 0.5),
            'seed': self.config.seed,
            'notes': f"Differentiable tau SNN stream + VQC layer. Device: {self.config.device}"
        }
        
        # Log router weights if the model is proposed QSpikeXAINet
        if hasattr(self.model, 'router'):
            # Collect gate weights during evaluation
            alpha_snn_list = []
            alpha_qnn_list = []
            self.model.eval()
            with torch.no_grad():
                for x_raw, x_scal, _ in val_loader:
                    x_raw = x_raw.to(self.device)
                    x_scal = x_scal.to(self.device)
                    # proposed model forward
                    _, gate = self.model(x_raw, x_scal, return_gate=True)
                    alpha_snn_list.extend(gate[:, 0].cpu().numpy())
                    alpha_qnn_list.extend(gate[:, 1].cpu().numpy())
            log_data['alpha_snn_mean'] = float(np.mean(alpha_snn_list))
            log_data['alpha_qnn_mean'] = float(np.mean(alpha_qnn_list))
            
        log_experiment(log_data, filepath=self.config.results_csv)
        
        return best_metrics

    def evaluate(self, dataloader) -> dict:
        """Evaluate model on a dataset."""
        self.model.eval()
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for x_raw, x_scal, y in dataloader:
                x_raw = x_raw.to(self.device)
                x_scal = x_scal.to(self.device)
                
                out = self.model(x_raw, x_scal)
                if isinstance(out, tuple):
                    logits, _ = out
                else:
                    logits = out
                    
                all_logits.append(logits.cpu().numpy())
                all_labels.extend(y.numpy())
                
        all_logits = np.concatenate(all_logits, axis=0)
        all_labels = np.array(all_labels)
        
        return compute_all_metrics(all_labels, all_logits)

from dataclasses import dataclass

@dataclass
class TrainingConfig:
    lr:              float = 1e-3
    weight_decay:    float = 1e-4
    optimizer:       str   = 'adamw'
    epochs:          int   = 80
    warmup_epochs:   int   = 5
    scheduler:       str   = 'cosine'
    label_smoothing: float = 0.1
    grad_clip:       float = 1.0
    dropout:         float = 0.3
    batch_size:      int   = 128
    n_folds:         int   = 5
    num_workers:     int   = 0
    device:          str   = 'cuda'
    seed:            int   = 42
    amp:             bool  = True
    early_stop_patience: int = 15
    checkpoint_metric: str = 'macro_f1'
    log_gate_weights: bool = True
    results_csv:     str   = 'results/canonical_results.csv'

