import os
import gc
import json
import time
import copy
from pathlib import Path
from dataclasses import dataclass, asdict
import warnings

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

# Ignore some specific warnings to keep output output clean
warnings.filterwarnings('ignore')

from src.data.apnea_dataset import ApneaECGDataset, ScalogramTransform
try:
    from src.models.transformer.swin import SwinECGClassifier
except ImportError:
    SwinECGClassifier = None

try:
    from src.models.snn.spiking_resnet import SpikingResNet
except ImportError:
    SpikingResNet = None

try:
    from src.models.quantum.hybrid_cnn import HybridQuantumCNN
except ImportError:
    HybridQuantumCNN = None

from src.utils.metrics import compute_metrics, count_flops

@dataclass
class Config:
    # Run configuration
    run_snn: bool = True
    run_quantum: bool = True
    run_swin: bool = True
    
    # Common configurations
    experiment_name: str = 'unified_apnea_pipeline'
    seed: int = 42
    epochs: int = 15
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    optimizer: str = 'adamw'
    scheduler: str = 'cosine'
    warmup_epochs: int = 3
    batch_size: int = 16 # Adjust based on RTX 5050 6GB VRAM
    num_workers: int = 0  # To prevent issues on local Windows machines
    mixed_precision: bool = True  # Enable AMP to save VRAM and increase speed
    gradient_accumulation_steps: int = 4 # Emulate larger batch size
    max_grad_norm: float = 1.0
    n_folds: int = 1
    early_stopping: bool = True
    patience: int = 5
    num_classes: int = 2
    output_dir: str = 'experiments'
    data_dir: str = 'data/apnea-ecg-database-1.0.0'
    
    # Scalogram parameters
    img_size: int = 224
    num_scales: int = 64
    sampling_rate: int = 100
    
    # SNN specific
    snn_timesteps: int = 25
    snn_beta: float = 0.9
    
    # Quantum specific
    n_qubits: int = 4
    q_depth: int = 2


def seed_everything(seed=42):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_model(model_name: str, model: nn.Module, cfg: Config):
    print(f"\n{'='*60}")
    print(f"🚀 INITIALIZING TRAINING FOR: {model_name}")
    print(f"{'='*60}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Log FLOPs
    fi = count_flops(model, (1, 3, cfg.img_size, cfg.img_size), device)
    print(f"📊 {model_name} Complexity: {fi['total_params']:,} Params | {fi['flops']:,} FLOPs")
    
    transform = ScalogramTransform(output_size=(cfg.img_size, cfg.img_size), 
                                   num_scales=cfg.num_scales, sr=cfg.sampling_rate)
    
    # Use fold=0 for demonstration
    train_ds = ApneaECGDataset(cfg.data_dir, fold=0, n_folds=5, split='train', transform=transform)
    val_ds = ApneaECGDataset(cfg.data_dir, fold=0, n_folds=5, split='val', transform=transform)
    
    sampler = WeightedRandomSampler(train_ds.get_sample_weights(), len(train_ds))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = nn.CrossEntropyLoss(weight=train_ds.get_class_weights().to(device))
    scaler = GradScaler() if cfg.mixed_precision and device.type == 'cuda' else None

    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    best_f1 = -1
    patience_ctr = 0
    fold_dir = Path(cfg.output_dir) / cfg.experiment_name / model_name.lower()
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(cfg.epochs):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        
        for step, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            if scaler:
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss = loss / cfg.gradient_accumulation_steps
                scaler.scale(loss).backward()
                if (step + 1) % cfg.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss = loss / cfg.gradient_accumulation_steps
                loss.backward()
                if (step + 1) % cfg.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
            
            running_loss += loss.item() * cfg.gradient_accumulation_steps
            
        train_loss = running_loss / len(train_loader)
        
        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, preds = outputs.max(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(targets.numpy())
                
        metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
        
        scheduler.step()
        
        elapsed = time.time() - t0
        print(f"  Ep {epoch+1:2d}/{cfg.epochs} | TrL: {train_loss:.4f} | VaA: {metrics['accuracy']*100:.1f}% | F1: {metrics['f1_macro']:.4f} | {elapsed:.1f}s")
        
        if metrics['f1_macro'] > best_f1:
            best_f1 = metrics['f1_macro']
            patience_ctr = 0
            torch.save(model.state_dict(), fold_dir / 'best_model.pt')
        else:
            patience_ctr += 1
            if cfg.early_stopping and patience_ctr >= cfg.patience:
                print(f"  🛑 Early stopping triggered at epoch {epoch+1}")
                break
                
    print(f"✨ Finished Training {model_name}. Best F1: {best_f1:.4f}\n")
    
    # Clean up to save VRAM
    del model, train_loader, val_loader, optimizer, scheduler, criterion
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()

def main():
    cfg = Config()
    seed_everything(cfg.seed)
    
    print("\n" + "#"*60)
    print("🧠 UNIFIED ECG RESEARCH PIPELINE")
    print("🔌 Device target: Local execution (RTX 5050 optimized)")
    print("#"*60 + "\n")
    
    if cfg.run_snn:
        if SpikingResNet is None:
            print("⚠️ WARNING: snntorch missing or imports failed. Skipping SNN.")
        else:
            snn_model = SpikingResNet(num_classes=cfg.num_classes, beta=cfg.snn_beta, timesteps=cfg.snn_timesteps)
            train_model("Spiking_ResNet", snn_model, cfg)
            
    if cfg.run_quantum:
        if HybridQuantumCNN is None:
            print("⚠️ WARNING: PennyLane missing or imports failed. Skipping Quantum CNN.")
        else:
            quantum_model = HybridQuantumCNN(num_classes=cfg.num_classes, n_qubits=cfg.n_qubits, q_depth=cfg.q_depth)
            train_model("Hybrid_Quantum_CNN", quantum_model, cfg)
            
    if cfg.run_swin:
        if SwinECGClassifier is None:
            print("⚠️ WARNING: timm missing or imports failed. Skipping Swin Transformer.")
        else:
            swin_model = SwinECGClassifier(num_classes=cfg.num_classes)
            train_model("Swin_Transformer", swin_model, cfg)

    print("🎉 All scheduled pipelines completed successfully!")

if __name__ == "__main__":
    main()
