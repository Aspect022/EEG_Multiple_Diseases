#!/usr/bin/env python3
"""
Quick test: Check what the model actually outputs.
Run this on the Ubuntu server where the code is working.
"""

import sys
sys.path.insert(0, '/home/user04/Projects/Cardio/Cancer/EEG_Multiple_Diseases')

import torch
from src.models.snn.spiking_resnet import create_spiking_resnet

# Create model
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = create_spiking_resnet(
        model_name='resnet18',
        num_classes=5,
        neuron_type='lif',
        num_timesteps=8,
    ).to(device)

    model.eval()

    # Test with random input
    x = torch.randn(2, 3, 224, 224).to(device)

    with torch.no_grad():
        out = model(x)

    print("Output tensor:")
    print(out)
    print(f"\nOutput min: {out.min():.6f}")
    print(f"Output max: {out.max():.6f}")
    print(f"Output mean: {out.mean():.6f}")
    print(f"Output is all zeros: {(out == 0).all().item()}")
    print(f"Output contains NaN: {torch.isnan(out).any().item()}")
    print(f"Output contains -inf: {torch.isneginf(out).any().item()}")

    # Check predictions
    _, predicted = out.max(1)
    print(f"\nPredictions: {predicted}")
    print(f"Are all predictions 0? {(predicted == 0).all().item()}")

