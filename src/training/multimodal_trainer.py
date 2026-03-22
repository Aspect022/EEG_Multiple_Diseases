"""
MultiModalFoldTrainer for models requiring paired raw-signal and scalogram data.

Uses a single dataloader that yields (raw_signal, scalogram, label) tuples so
multi-modal training stays sample-aligned.
"""

import time
import torch
from typing import Dict, Any, Tuple
from tqdm import tqdm
from torch.amp import autocast

from .research_trainer import FoldTrainer, ResearchConfig
from src.evaluation.metrics import compute_all_metrics

HAS_TQDM = True
try:
    import wandb
except ImportError:
    pass

class MultiModalFoldTrainer(FoldTrainer):
    """
    Trainer for Multi-Modal models (like Fusion-C) that require 
    both raw 1D signals and 2D scalograms.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        config: ResearchConfig,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        fold: int = 0,
        class_weights: torch.Tensor = None,
    ):
        super().__init__(
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            fold=fold,
            class_weights=class_weights,
        )

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        steps = len(self.train_loader)
        loader = self.train_loader
        
        if HAS_TQDM:
            loader = tqdm(loader, total=steps, desc=f"  [Fold {self.fold}] Epoch {epoch+1}", leave=False)

        for step, (inputs_1d, inputs_2d, targets) in enumerate(loader):
            inputs_1d = inputs_1d.to(self.device, non_blocking=True)
            inputs_2d = inputs_2d.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self._warmup_lr(epoch, step, steps)

            # Forward pass
            if self.scaler is not None:
                with autocast('cuda'):
                    outputs = self._unwrap_outputs(
                        self.model(raw_signal=inputs_1d, scalogram=inputs_2d)
                    )
                    loss = self.criterion(outputs, targets)
                    
                    if hasattr(self.model, 'reg_loss'):
                        reg = self.model.reg_loss
                        if isinstance(reg, torch.Tensor) and reg.requires_grad:
                            loss = loss + reg
                            
                    loss = loss / self.config.gradient_accumulation_steps
                self.scaler.scale(loss).backward()
            else:
                outputs = self._unwrap_outputs(
                    self.model(raw_signal=inputs_1d, scalogram=inputs_2d)
                )
                loss = self.criterion(outputs, targets)
                
                if hasattr(self.model, 'reg_loss'):
                    reg = self.model.reg_loss
                    if isinstance(reg, torch.Tensor) and reg.requires_grad:
                        loss = loss + reg
                        
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()

            # Optimizer Step
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

    @torch.no_grad()
    def evaluate(self) -> Dict[str, Any]:
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []

        steps = len(self.val_loader)

        for inputs_1d, inputs_2d, targets in self.val_loader:
            inputs_1d = inputs_1d.to(self.device, non_blocking=True)
            inputs_2d = inputs_2d.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            outputs = self._unwrap_outputs(
                self.model(raw_signal=inputs_1d, scalogram=inputs_2d)
            )
            loss = self.criterion(outputs, targets)
            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

        avg_loss = total_loss / max(steps, 1)
        import numpy as np
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)

        metrics = compute_all_metrics(y_true, y_pred, y_prob)
        metrics['val_loss'] = avg_loss

        return metrics
