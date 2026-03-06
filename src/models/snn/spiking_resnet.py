"""
Spiking Neural Network models based on ResNet architecture.

Supports two neuron types:
- LIF (Leaky Integrate-and-Fire): Standard linear membrane dynamics.
- QIF (Quadratic Integrate-and-Fire): Nonlinear quadratic membrane dynamics.

References:
    - LIF: snntorch.Leaky with surrogate gradient (fast_sigmoid).
    - QIF: tau * dv/dt = v^2 + mu(t), spike at v >= v_th, reset to v_reset.
      (Izhikevich, 2007; Latham et al., 2000)
"""

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from typing import Tuple, Optional


# ==========================================================================
# Quadratic Integrate-and-Fire Neuron
# ==========================================================================

class QuadraticIF(nn.Module):
    """
    Quadratic Integrate-and-Fire (QIF) neuron model.

    Membrane dynamics (discrete):
        v[t+1] = v[t] + (dt/tau) * (v[t]^2 + mu[t])
        if v[t+1] >= v_th: spike, reset v -> v_reset

    Compatible with snntorch API: call returns spike tensor (single output).

    Args:
        threshold: Spiking threshold. Default: 1.0.
        v_reset: Reset voltage after spike. Default: -1.0.
        tau: Membrane time constant. Default: 1.0.
        dt: Integration timestep. Default: 0.1.
        spike_grad: Surrogate gradient function.
    """

    def __init__(
        self,
        threshold: float = 1.0,
        v_reset: float = -1.0,
        tau: float = 1.0,
        dt: float = 0.1,
        spike_grad=None,
        **kwargs,
    ):
        super().__init__()
        self.threshold = threshold
        self.v_reset = v_reset
        self.tau = tau
        self.dt = dt

        if spike_grad is None:
            spike_grad = surrogate.fast_sigmoid(slope=25)
        self.spike_grad = spike_grad

        self.mem = None

    def init_leaky(self):
        """Reset membrane potential (snntorch-compatible name)."""
        self.mem = None

    def reset_mem(self):
        """Reset membrane potential."""
        self.mem = None

    def _init_mem(self, input_: torch.Tensor) -> torch.Tensor:
        return torch.full_like(input_, self.v_reset)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """
        Single timestep forward. Returns spike tensor only.

        Args:
            input_: Input current (*).
        Returns:
            Spike tensor (*).
        """
        if self.mem is None:
            self.mem = self._init_mem(input_)

        # QIF dynamics: v += (dt/tau) * (v^2 + input)
        mem_clamped = torch.clamp(self.mem, min=-10.0, max=10.0)
        new_mem = mem_clamped + (self.dt / self.tau) * (mem_clamped ** 2 + input_)

        # Spike detection with surrogate gradient
        spike = self.spike_grad(new_mem - self.threshold)

        # Reset where spike occurred
        new_mem = new_mem * (1.0 - spike.detach()) + self.v_reset * spike.detach()
        self.mem = new_mem

        return spike


# ==========================================================================
# Neuron Factory
# ==========================================================================

def create_neuron(neuron_type: str = 'lif', beta: float = 0.9, spike_grad=None, **kwargs):
    """
    Factory to create a spiking neuron.

    Args:
        neuron_type: 'lif' or 'qif'.
        beta: Decay rate for LIF.
        spike_grad: Surrogate gradient function.
    Returns:
        Spiking neuron module. Call with (input) -> spike.
    """
    if spike_grad is None:
        spike_grad = surrogate.fast_sigmoid(slope=25)

    if neuron_type == 'lif':
        return snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
    elif neuron_type == 'qif':
        return QuadraticIF(spike_grad=spike_grad, **kwargs)
    else:
        raise ValueError(f"Unknown neuron_type: {neuron_type}. Use 'lif' or 'qif'.")


# ==========================================================================
# Spiking Conv2d
# ==========================================================================

class SpikingConv2d(nn.Module):
    """Conv2d + BN + spiking neuron."""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1,
                 neuron_type='lif', beta=0.9):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.neuron = create_neuron(neuron_type=neuron_type, beta=beta)

    def forward(self, x):
        return self.neuron(self.bn(self.conv(x)))


# ==========================================================================
# Spiking Basic Block
# ==========================================================================

class SpikingBasicBlock(nn.Module):
    """Spiking residual block."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, neuron_type='lif', beta=0.9):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.neuron1 = create_neuron(neuron_type=neuron_type, beta=beta)

        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.neuron2 = create_neuron(neuron_type=neuron_type, beta=beta)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.neuron1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        out = self.neuron2(out)
        return out


# ==========================================================================
# Spiking ResNet
# ==========================================================================

class SpikingResNet(nn.Module):
    """
    Spiking ResNet for scalogram-based ECG classification.

    Args:
        block: Residual block class.
        num_blocks: Blocks per stage.
        num_classes: Output classes. Default: 5.
        in_channels: Input channels. Default: 3.
        num_timesteps: Temporal steps. Default: 25.
        neuron_type: 'lif' or 'qif'. Default: 'lif'.
        beta: LIF decay rate. Default: 0.9.
    """

    def __init__(self, block, num_blocks, num_classes=5, in_channels=3,
                 num_timesteps=25, neuron_type='lif', beta=0.9):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.neuron_type = neuron_type
        self.beta = beta
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.lif1 = create_neuron(neuron_type=neuron_type, beta=beta)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.lif_out = create_neuron(neuron_type=neuron_type, beta=beta)

        self._initialize_weights()

    def _make_layer(self, block, planes, num_blocks, stride):
        layers = []
        layers.append(block(self.in_planes, planes, stride=stride,
                            neuron_type=self.neuron_type, beta=self.beta))
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes,
                                neuron_type=self.neuron_type, beta=self.beta))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _reset_states(self):
        """Reset all spiking neurons."""
        for module in self.modules():
            if isinstance(module, snn.Leaky):
                module.init_leaky()
            elif isinstance(module, QuadraticIF):
                module.reset_mem()

    def forward_timestep(self, x):
        out = self.lif1(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.lif_out(self.fc(out))
        return out

    def forward(self, x):
        self._reset_states()
        spike_record = []
        for t in range(self.num_timesteps):
            spk = self.forward_timestep(x)
            spike_record.append(spk)
        return torch.stack(spike_record, dim=0).mean(dim=0)

    @classmethod
    def resnet18(cls, **kwargs):
        return cls(SpikingBasicBlock, [2, 2, 2, 2], **kwargs)

    @classmethod
    def resnet34(cls, **kwargs):
        return cls(SpikingBasicBlock, [3, 4, 6, 3], **kwargs)


# ==========================================================================
# Spiking ResNet 1D
# ==========================================================================

class SpikingResNet1D(nn.Module):
    """Spiking ResNet for 1D ECG signals."""

    def __init__(self, num_classes=5, in_channels=12, num_timesteps=25,
                 neuron_type='lif', beta=0.9):
        super().__init__()
        self.num_timesteps = num_timesteps

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 64, 15, stride=2, padding=7),
            nn.BatchNorm1d(64),
        )
        self.lif_enc = create_neuron(neuron_type=neuron_type, beta=beta)

        self.block1 = self._make_block(64, 128, neuron_type, beta)
        self.block2 = self._make_block(128, 256, neuron_type, beta)
        self.block3 = self._make_block(256, 512, neuron_type, beta)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)
        self.lif_out = create_neuron(neuron_type=neuron_type, beta=beta)

    def _make_block(self, in_ch, out_ch, neuron_type, beta):
        return nn.ModuleDict({
            'conv1': nn.Conv1d(in_ch, out_ch, 7, stride=2, padding=3),
            'bn1': nn.BatchNorm1d(out_ch),
            'lif1': create_neuron(neuron_type=neuron_type, beta=beta),
            'conv2': nn.Conv1d(out_ch, out_ch, 7, stride=1, padding=3),
            'bn2': nn.BatchNorm1d(out_ch),
            'lif2': create_neuron(neuron_type=neuron_type, beta=beta),
            'downsample': nn.Conv1d(in_ch, out_ch, 1, stride=2),
        })

    def _forward_block(self, x, block):
        identity = block['downsample'](x)
        out = block['lif1'](block['bn1'](block['conv1'](x)))
        out = block['bn2'](block['conv2'](out))
        out = out + identity
        out = block['lif2'](out)
        return out

    def _reset_states(self):
        for module in self.modules():
            if isinstance(module, snn.Leaky):
                module.init_leaky()
            elif isinstance(module, QuadraticIF):
                module.reset_mem()

    def forward(self, x):
        self._reset_states()
        spike_record = []
        for t in range(self.num_timesteps):
            out = self.lif_enc(self.encoder(x))
            out = self._forward_block(out, self.block1)
            out = self._forward_block(out, self.block2)
            out = self._forward_block(out, self.block3)
            out = self.lif_out(self.fc(self.global_pool(out).squeeze(-1)))
            spike_record.append(out)
        return torch.stack(spike_record, dim=0).mean(dim=0)


# ==========================================================================
# Factory
# ==========================================================================

def create_spiking_resnet(model_name='resnet18', num_classes=5, num_timesteps=25,
                          neuron_type='lif', pretrained=False, **kwargs):
    """Factory for Spiking ResNet models."""
    if model_name == 'resnet18':
        return SpikingResNet.resnet18(num_classes=num_classes,
                                      num_timesteps=num_timesteps,
                                      neuron_type=neuron_type, **kwargs)
    elif model_name == 'resnet34':
        return SpikingResNet.resnet34(num_classes=num_classes,
                                      num_timesteps=num_timesteps,
                                      neuron_type=neuron_type, **kwargs)
    elif model_name == '1d':
        return SpikingResNet1D(num_classes=num_classes,
                               num_timesteps=num_timesteps,
                               neuron_type=neuron_type, **kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")
