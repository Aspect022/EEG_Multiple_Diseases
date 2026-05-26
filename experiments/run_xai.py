import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from qspikexai.data.unified_loader import create_unified_dataloaders
from qspikexai.models.qspikexai_net import QSpikeXAINet
from qspikexai.xai.signal_xai import input_saliency, integrated_gradients
from qspikexai.xai.quantum_xai import quantum_gate_attribution, per_task_quantum_profile

def plot_heatmap(data, title, xlabel, ylabel, save_path):
    plt.figure(figsize=(10, 6))
    sns.heatmap(data, cmap='viridis', cbar=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Run Explainable AI (XAI) Module")
    parser.add_argument('--task', type=str, required=True, choices=['sleep_apnea', 'schizophrenia', 'mci', 'depression'])
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print(f"Loading model from checkpoint: {args.checkpoint}...")
    model = QSpikeXAINet(task=args.task, n_qubits=4, n_heads_vqc=4)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print("Loading data for explainability...")
    _, val_loader = create_unified_dataloaders(
        task=args.task,
        data_dir=args.data_dir,
        fold=0,
        n_folds=5,
        batch_size=4,
        seed=42
    )

    x_raw, x_scal, y = next(iter(val_loader))
    x_raw_sample = x_raw[0:1].to(device)
    x_scal_sample = x_scal[0:1].to(device)

    # Create output directory
    os.makedirs('results/xai', exist_ok=True)

    # 1. Compute Input Saliency
    print("Computing Input Saliency...")
    sal = input_saliency(model, x_raw_sample, target_class=y[0].item())
    # Save input saliency map
    np.save(f"results/xai/{args.task}_saliency.npy", sal)
    plot_heatmap(
        sal[0], 
        f"Input Saliency Map - {args.task.replace('_', ' ').title()}", 
        "Time Points", 
        "EEG Channels", 
        f"results/xai/{args.task}_saliency.png"
    )

    # 2. Compute Integrated Gradients
    print("Computing Integrated Gradients...")
    ig = integrated_gradients(model, x_raw_sample, steps=32)
    np.save(f"results/xai/{args.task}_integrated_gradients.npy", ig)
    plot_heatmap(
        ig[0], 
        f"Integrated Gradients Map - {args.task.replace('_', ' ').title()}", 
        "Time Points", 
        "EEG Channels", 
        f"results/xai/{args.task}_integrated_gradients.png"
    )

    # 3. Compute Quantum Gate Attribution
    print("Computing Quantum Gate Attribution...")
    quantum_attr = quantum_gate_attribution(model, x_raw_sample, x_scal_sample, target_class=y[0].item())
    theta_attr = quantum_attr['theta_attribution'] # (3, n_qubits, 3)
    np.save(f"results/xai/{args.task}_quantum_attribution.npy", theta_attr)

    # Plot average theta attribution over gates (3 layers x 4 qubits)
    avg_theta = theta_attr.mean(axis=-1) # (3, 4)
    plot_heatmap(
        avg_theta, 
        f"VQC Parameter Attribution - {args.task.replace('_', ' ').title()}", 
        "Qubits", 
        "VQC Layers", 
        f"results/xai/{args.task}_quantum_attribution.png"
    )

    print(f"XAI results successfully saved in results/xai/ for task {args.task}!")

if __name__ == '__main__':
    main()
