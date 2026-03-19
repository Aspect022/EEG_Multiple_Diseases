#!/usr/bin/env python3
"""Debug: Check if SNN forward pass produces valid outputs."""

import sys
sys.path.insert(0, '/home/user04/Projects/Cardio/Cancer/EEG_Multiple_Diseases')

import torch
import numpy as np

# Import the model
from src.models.snn.spiking_resnet import create_spiking_resnet

# Create model
print("Creating SNN-ResNet18-LIF model...")
model = create_spiking_resnet(
    model_name='resnet18',
    num_classes=5,
    neuron_type='lif',
    num_timesteps=8,
)
model = model.cuda()
model.eval()

print(f"Model created. Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Create dummy input (simulating a batch of scalograms)
batch_size = 4
dummy_input = torch.rand(batch_size, 3, 224, 224).cuda()

print(f"\nInput shape: {dummy_input.shape}")
print(f"Input range: [{dummy_input.min():.4f}, {dummy_input.max():.4f}]")

# Run forward pass
print("\nRunning forward pass...")
with torch.no_grad():
    output = model(dummy_input)

print(f"\nOutput shape: {output.shape}")
print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
print(f"Output mean: {output.mean():.4f}")
print(f"Output std: {output.std():.4f}")

# Check for NaN/Inf
print(f"\nContains NaN: {torch.isnan(output).any().item()}")
print(f"Contains Inf: {torch.isinf(output).any().item()}")

# Check predictions
_, predicted = output.max(1)
print(f"\nPredictions: {predicted.cpu().numpy()}")
print(f"Unique predictions: {predicted.unique().cpu().numpy()}")

# Check if all predictions are the same
if len(predicted.unique()) == 1:
    print("\n⚠️  WARNING: All predictions are the same class!")
    print(f"   This explains why validation accuracy is stuck!")

# Check spike stats if available
if hasattr(model, 'spike_stats') and model.spike_stats:
    print(f"\nSpike stats:")
    print(f"  Average rate: {model.spike_stats.get('avg_rate', 'N/A')}")
    if 'rates' in model.spike_stats:
        print(f"  Rates per timestep: {model.spike_stats['rates']}")

# Now test with a different input to see if output changes
print("\n" + "="*60)
print("Testing with different input...")
dummy_input2 = torch.rand(batch_size, 3, 224, 224).cuda() * 0.1  # Lower amplitude

with torch.no_grad():
    output2 = model(dummy_input2)

print(f"Output2 range: [{output2.min():.4f}, {output2.max():.4f}]")
print(f"Output2 mean: {output2.mean():.4f}")

# Check if outputs are different
output_diff = (output - output2).abs().mean().item()
print(f"\nMean absolute difference between outputs: {output_diff:.6f}")

if output_diff < 0.001:
    print("⚠️  WARNING: Outputs are almost identical for different inputs!")
    print("   The model is not responding to input variations!")
else:
    print("✓ Model output varies with input (good)")
