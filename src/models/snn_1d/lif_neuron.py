"""
Standard Leaky Integrate-and-Fire (LIF) neuron for EEG processing.

No binary quantization — full precision for best accuracy.
Uses surrogate gradient for backpropagation through discrete spikes.

Adapted from the ECG project's QuantLIFNeuron but without quantization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SurrogateGradient(torch.autograd.Function):
    """
    Surrogate gradient for spiking threshold.
    
    Uses fast sigmoid approximation for the backward pass:
    σ'(x) = 1 / (1 + k|x|)² where k controls sharpness.
    """
    scale = 25.0  # Sharpness of surrogate gradient
    
    @staticmethod
    def forward(ctx, membrane_potential, threshold):
        ctx.save_for_backward(membrane_potential, threshold)
        return (membrane_potential >= threshold).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        membrane_potential, threshold = ctx.saved_tensors
        grad_input = grad_output / (SurrogateGradient.scale * torch.abs(
            membrane_potential - threshold
        ) + 1.0) ** 2
        return grad_input, None


class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire neuron with learnable parameters.
    
    Implements the LIF dynamics:
        V(t+1) = τ * V(t) + I(t) - V_th * spike(t)
        spike(t) = Θ(V(t) - V_th)
    
    Includes spike regularization to encourage sparse but non-zero spiking.
    
    Args:
        threshold: Initial firing threshold (learnable)
        tau: Membrane time constant / leak factor (learnable)
        spike_reg: Spike regularization coefficient
    """
    
    def __init__(
        self,
        threshold: float = 1.0,
        tau: float = 0.75,
        spike_reg: float = 0.01,
    ):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(threshold))
        self.tau = nn.Parameter(torch.tensor(tau))
        self.spike_reg = spike_reg
    
    def forward(
        self, x: torch.Tensor, timesteps: int = 4
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process input through LIF dynamics over multiple timesteps.
        
        Args:
            x: Input current (batch, channels, length)
            timesteps: Number of simulation timesteps
            
        Returns:
            Tuple of (output_spikes, membrane_potential, reg_loss)
            - output_spikes: Accumulated spikes (batch, channels, length)
            - membrane_potential: Final membrane state
            - reg_loss: Spike regularization loss (scalar)
        """
        B, C, L = x.shape
        device = x.device
        
        # Clamp tau to valid range (0, 1)
        tau = torch.sigmoid(self.tau)
        
        # Initialize membrane potential
        membrane = torch.zeros(B, C, L, device=device)
        spike_sum = torch.zeros(B, C, L, device=device)
        
        # Divide input current across timesteps
        input_current = x / timesteps
        
        for t in range(timesteps):
            # Leak + integrate
            membrane = tau * membrane + input_current
            
            # Spike generation (with surrogate gradient)
            spikes = SurrogateGradient.apply(membrane, self.threshold)
            spike_sum = spike_sum + spikes
            
            # Reset after spike (soft reset)
            membrane = membrane - spikes * self.threshold
        
        # Spike regularization: encourage ~50% firing rate per neuron
        firing_rate = spike_sum.mean() / timesteps
        target_rate = 0.5
        reg_loss = self.spike_reg * (firing_rate - target_rate) ** 2
        
        return spike_sum, membrane, reg_loss


class LIFLayer(nn.Module):
    """
    Convenience layer: BatchNorm + LIF neuron.
    
    Drop-in replacement for ReLU in any Conv1d pipeline.
    
    Args:
        channels: Number of channels for BatchNorm
        threshold: LIF threshold
        tau: LIF leak factor
    """
    
    def __init__(self, channels: int, threshold: float = 1.0, tau: float = 0.75):
        super().__init__()
        self.bn = nn.BatchNorm1d(channels)
        self.lif = LIFNeuron(threshold=threshold, tau=tau)
    
    def forward(
        self, x: torch.Tensor, timesteps: int = 4
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, channels, length)
        Returns:
            Tuple of (output, reg_loss)
        """
        x = self.bn(x)
        spikes, _, reg_loss = self.lif(x, timesteps=timesteps)
        return spikes, reg_loss
