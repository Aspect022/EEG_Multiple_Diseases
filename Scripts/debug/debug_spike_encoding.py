#!/usr/bin/env python3
"""Debug script to check SNN forward pass output."""

import torch
import numpy as np

# Create dummy input (simulating a scalogram batch)
batch_size = 2
dummy_input = torch.rand(batch_size, 3, 224, 224)  # (B, C, H, W)

print("Input stats:")
print(f"  Shape: {dummy_input.shape}")
print(f"  Min: {dummy_input.min():.4f}")
print(f"  Max: {dummy_input.max():.4f}")
print(f"  Mean: {dummy_input.mean():.4f}")

# Simulate the spike encoding from spiking_resnet.py
x = dummy_input
x_min = x.view(x.size(0), -1).min(dim=1, keepdim=True)[0]
x_max = x.view(x.size(0), -1).max(dim=1, keepdim=True)[0]
x_norm = (x - x_min.view(-1, 1, 1, 1)) / (x_max.view(-1, 1, 1, 1) - x_min.view(-1, 1, 1, 1) + 1e-8)

print("\nAfter normalization:")
print(f"  Min: {x_norm.min():.4f}")
print(f"  Max: {x_norm.max():.4f}")
print(f"  Mean: {x_norm.mean():.4f}")

# OLD encoding (0.3 scale)
x_spike_prob_old = x_norm * 0.3
print("\nOLD spike probability (scale=0.3):")
print(f"  Max prob: {x_spike_prob_old.max():.4f}")
print(f"  Mean prob: {x_spike_prob_old.mean():.4f}")

# Expected spikes per timestep
expected_spikes_old = x_spike_prob_old.mean().item() * 8  # 8 timesteps
print(f"  Expected spikes per pixel (8 timesteps): {expected_spikes_old:.4f}")

# NEW encoding (0.7 scale)
x_spike_prob_new = x_norm * 0.7
print("\nNEW spike probability (scale=0.7):")
print(f"  Max prob: {x_spike_prob_new.max():.4f}")
print(f"  Mean prob: {x_spike_prob_new.mean():.4f}")

expected_spikes_new = x_spike_prob_new.mean().item() * 8
print(f"  Expected spikes per pixel (8 timesteps): {expected_spikes_new:.4f}")

# Simulate spike generation
num_timesteps = 8
spike_record_old = []
spike_record_new = []

for t in range(num_timesteps):
    spike_mask_old = torch.rand_like(x_spike_prob_old) < x_spike_prob_old
    x_t_old = x_spike_prob_old * spike_mask_old.float()
    spike_record_old.append(x_t_old)
    
    spike_mask_new = torch.rand_like(x_spike_prob_new) < x_spike_prob_new
    x_t_new = x_spike_prob_new * spike_mask_new.float()
    spike_record_new.append(x_t_new)

# Sum across timesteps (what the model does)
output_old = torch.stack(spike_record_old, dim=0).sum(dim=0)
output_new = torch.stack(spike_record_new, dim=0).sum(dim=0)

print("\n=== OUTPUT COMPARISON (after summing timesteps) ===")
print(f"OLD (0.3 scale):")
print(f"  Shape: {output_old.shape}")
print(f"  Min: {output_old.min():.4f}")
print(f"  Max: {output_old.max():.4f}")
print(f"  Mean: {output_old.mean():.4f}")
print(f"  Non-zero elements: {(output_old > 0).sum().item()} / {output_old.numel()} ({(output_old > 0).float().mean()*100:.1f}%)")

print(f"\nNEW (0.7 scale):")
print(f"  Shape: {output_new.shape}")
print(f"  Min: {output_new.min():.4f}")
print(f"  Max: {output_new.max():.4f}")
print(f"  Mean: {output_new.mean():.4f}")
print(f"  Non-zero elements: {(output_new > 0).sum().item()} / {output_new.numel()} ({(output_new > 0).float().mean()*100:.1f}%)")

print("\n=== CONCLUSION ===")
print(f"Signal increase: {output_new.mean().item() / output_old.mean().item():.2f}x")
print("The NEW encoding provides significantly more signal to the network!")
