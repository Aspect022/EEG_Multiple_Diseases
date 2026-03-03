"""
Spiking Neural Network ResNet for ECG Classification.

This module implements a Spiking ResNet using snntorch with Leaky Integrate-and-Fire
(LIF) neurons. The architecture converts 2D inputs (scalograms) through spiking
convolutions with temporal dynamics.

Reference: 
- snntorch: https://snntorch.readthedocs.io/
- Spiking ResNets: https://arxiv.org/abs/2108.05340
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import snntorch as snn
from snntorch import surrogate


class SpikingConv2d(nn.Module):
    """
    Spiking convolutional layer with LIF neuron.
    
    Combines a standard Conv2d with batch normalization and a LIF neuron
    that produces spike outputs.
    
    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        stride: Convolution stride. Default: 1.
        padding: Convolution padding. Default: 0.
        beta: LIF neuron decay rate. Default: 0.9.
        spike_grad: Surrogate gradient function. Default: fast_sigmoid.
        threshold: Spike threshold. Default: 1.0.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        beta: float = 0.9,
        spike_grad: Optional[callable] = None,
        threshold: float = 1.0,
    ):
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        
        if spike_grad is None:
            spike_grad = surrogate.fast_sigmoid(slope=25)
        
        self.lif = snn.Leaky(
            beta=beta,
            spike_grad=spike_grad,
            threshold=threshold,
            init_hidden=True,
        )
        
    def forward(self, x: torch.Tensor, mem: Optional[torch.Tensor] = None):
        """
        Forward pass through spiking conv layer.
        
        Args:
            x: Input tensor of shape (B, C, H, W).
            mem: Optional membrane potential from previous timestep.
            
        Returns:
            Tuple of (spikes, membrane_potential).
        """
        x = self.conv(x)
        x = self.bn(x)
        spk, mem = self.lif(x, mem)
        return spk, mem


class SpikingBasicBlock(nn.Module):
    """
    Spiking version of ResNet BasicBlock.
    
    Contains two spiking conv layers with a residual connection.
    The residual is added to the membrane potential before spiking.
    
    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Stride for first convolution. Default: 1.
        downsample: Optional downsampling layer for residual. Default: None.
        beta: LIF neuron decay rate. Default: 0.9.
    """
    
    expansion = 1
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        beta: float = 0.9,
    ):
        super().__init__()
        
        spike_grad = surrogate.fast_sigmoid(slope=25)
        
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        
        self.downsample = downsample
        
    def forward(
        self,
        x: torch.Tensor,
        mem1: Optional[torch.Tensor] = None,
        mem2: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through spiking basic block.
        
        Args:
            x: Input spikes of shape (B, C, H, W).
            mem1: Membrane potential for first LIF.
            mem2: Membrane potential for second LIF.
            
        Returns:
            Tuple of (output_spikes, mem1, mem2).
        """
        identity = x
        
        # First conv + LIF
        out = self.conv1(x)
        out = self.bn1(out)
        spk1, mem1 = self.lif1(out, mem1)
        
        # Second conv (no spike yet, add residual first)
        out = self.conv2(spk1)
        out = self.bn2(out)
        
        # Downsample residual if needed
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Add residual to membrane potential before spiking
        out = out + identity
        spk2, mem2 = self.lif2(out, mem2)
        
        return spk2, mem1, mem2


class SpikingResNet(nn.Module):
    """
    Spiking ResNet for image classification.
    
    Implements a ResNet architecture using spiking neurons (LIF) for 
    temporal processing of static images. The input is presented across
    multiple timesteps, and the output is the average spike rate of
    the output neurons.
    
    Args:
        block: Block type (SpikingBasicBlock).
        layers: List of number of blocks per stage.
        num_classes: Number of output classes. Default: 5.
        in_channels: Number of input channels. Default: 3.
        num_timesteps: Number of simulation timesteps. Default: 25.
        beta: LIF neuron decay rate. Default: 0.9.
        
    Example:
        >>> model = SpikingResNet.resnet18(num_classes=5, num_timesteps=25)
        >>> x = torch.randn(4, 3, 224, 224)
        >>> output = model(x)  # (4, 5)
    """
    
    def __init__(
        self,
        block,
        layers: List[int],
        num_classes: int = 5,
        in_channels: int = 3,
        num_timesteps: int = 25,
        beta: float = 0.9,
    ):
        super().__init__()
        
        self.num_timesteps = num_timesteps
        self.beta = beta
        self.in_planes = 64
        
        spike_grad = surrogate.fast_sigmoid(slope=25)
        
        # Initial convolution
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7,
            stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual stages
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.lif_out = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        
        # Initialize weights
        self._initialize_weights()
        
    def _make_layer(
        self,
        block,
        planes: int,
        num_blocks: int,
        stride: int = 1,
    ) -> nn.ModuleList:
        """Create a stage of residual blocks."""
        downsample = None
        
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_planes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )
        
        layers = nn.ModuleList()
        layers.append(block(
            self.in_planes, planes, stride, downsample, beta=self.beta
        ))
        
        self.in_planes = planes * block.expansion
        
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, beta=self.beta))
        
        return layers
    
    def _initialize_weights(self):
        """Initialize network weights."""
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
        """Reset all membrane potentials to None for new sequence."""
        for module in self.modules():
            if isinstance(module, snn.Leaky):
                module.reset_mem()
    
    def forward_timestep(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a single timestep.
        
        Args:
            x: Input tensor of shape (B, C, H, W).
            
        Returns:
            Output spikes of shape (B, num_classes).
        """
        # Initial conv
        out = self.conv1(x)
        out = self.bn1(out)
        spk, _ = self.lif1(out)
        out = self.maxpool(spk)
        
        # Residual stages
        for block in self.layer1:
            out, _, _ = block(out)
        for block in self.layer2:
            out, _, _ = block(out)
        for block in self.layer3:
            out, _, _ = block(out)
        for block in self.layer4:
            out, _, _ = block(out)
        
        # Classifier
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        spk_out, _ = self.lif_out(out)
        
        return spk_out
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with temporal simulation.
        
        The input is presented for num_timesteps iterations,
        and the output is the average spike rate.
        
        Args:
            x: Input tensor of shape (B, C, H, W).
            
        Returns:
            Output logits of shape (B, num_classes) as spike rates.
        """
        self._reset_states()
        
        # Accumulate spikes over timesteps
        spike_record = []
        
        for t in range(self.num_timesteps):
            spk = self.forward_timestep(x)
            spike_record.append(spk)
        
        # Stack and compute average spike rate
        spike_record = torch.stack(spike_record, dim=0)  # (T, B, classes)
        output = spike_record.mean(dim=0)  # (B, classes)
        
        return output
    
    @classmethod
    def resnet18(cls, **kwargs) -> 'SpikingResNet':
        """Create Spiking ResNet-18."""
        return cls(SpikingBasicBlock, [2, 2, 2, 2], **kwargs)
    
    @classmethod
    def resnet34(cls, **kwargs) -> 'SpikingResNet':
        """Create Spiking ResNet-34."""
        return cls(SpikingBasicBlock, [3, 4, 6, 3], **kwargs)


class SpikingResNet1D(nn.Module):
    """
    Spiking ResNet for 1D ECG signals (without scalogram transform).
    
    Processes raw 12-lead ECG signals directly using 1D convolutions
    with spiking neurons.
    
    Args:
        num_classes: Number of output classes. Default: 5.
        in_channels: Number of ECG leads. Default: 12.
        num_timesteps: Number of simulation timesteps. Default: 25.
        beta: LIF neuron decay rate. Default: 0.9.
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        in_channels: int = 12,
        num_timesteps: int = 25,
        beta: float = 0.9,
    ):
        super().__init__()
        
        self.num_timesteps = num_timesteps
        spike_grad = surrogate.fast_sigmoid(slope=25)
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 64, 15, stride=2, padding=7),
            nn.BatchNorm1d(64),
        )
        self.lif_enc = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        
        # Residual blocks (simplified)
        self.block1 = self._make_block(64, 128, beta, spike_grad)
        self.block2 = self._make_block(128, 256, beta, spike_grad)
        self.block3 = self._make_block(256, 512, beta, spike_grad)
        
        # Classifier
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)
        self.lif_out = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        
    def _make_block(self, in_ch, out_ch, beta, spike_grad):
        return nn.ModuleDict({
            'conv1': nn.Conv1d(in_ch, out_ch, 7, stride=2, padding=3),
            'bn1': nn.BatchNorm1d(out_ch),
            'lif1': snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            'conv2': nn.Conv1d(out_ch, out_ch, 7, stride=1, padding=3),
            'bn2': nn.BatchNorm1d(out_ch),
            'lif2': snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            'downsample': nn.Conv1d(in_ch, out_ch, 1, stride=2),
        })
    
    def _forward_block(self, x, block):
        identity = block['downsample'](x)
        
        out = block['conv1'](x)
        out = block['bn1'](out)
        spk, _ = block['lif1'](out)
        
        out = block['conv2'](spk)
        out = block['bn2'](out)
        out = out + identity
        spk, _ = block['lif2'](out)
        
        return spk
    
    def _reset_states(self):
        for module in self.modules():
            if isinstance(module, snn.Leaky):
                module.reset_mem()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for 1D ECG signals.
        
        Args:
            x: Input tensor of shape (B, leads, seq_len).
            
        Returns:
            Output logits of shape (B, num_classes).
        """
        self._reset_states()
        
        spike_record = []
        
        for t in range(self.num_timesteps):
            # Encoder
            out = self.encoder(x)
            spk, _ = self.lif_enc(out)
            
            # Blocks
            spk = self._forward_block(spk, self.block1)
            spk = self._forward_block(spk, self.block2)
            spk = self._forward_block(spk, self.block3)
            
            # Classifier
            out = self.global_pool(spk).squeeze(-1)
            out = self.fc(out)
            spk_out, _ = self.lif_out(out)
            
            spike_record.append(spk_out)
        
        output = torch.stack(spike_record, dim=0).mean(dim=0)
        return output


def create_spiking_resnet(
    model_name: str = 'resnet18',
    num_classes: int = 5,
    num_timesteps: int = 25,
    pretrained: bool = False,
    **kwargs,
) -> SpikingResNet:
    """
    Factory function to create Spiking ResNet models.
    
    Args:
        model_name: Model variant ('resnet18', 'resnet34', '1d').
        num_classes: Number of output classes.
        num_timesteps: Number of simulation timesteps.
        pretrained: Ignored (no pretrained weights available).
        **kwargs: Additional arguments passed to model constructor.
        
    Returns:
        Spiking ResNet model.
    """
    if model_name == 'resnet18':
        return SpikingResNet.resnet18(
            num_classes=num_classes,
            num_timesteps=num_timesteps,
            **kwargs
        )
    elif model_name == 'resnet34':
        return SpikingResNet.resnet34(
            num_classes=num_classes,
            num_timesteps=num_timesteps,
            **kwargs
        )
    elif model_name == '1d':
        return SpikingResNet1D(
            num_classes=num_classes,
            num_timesteps=num_timesteps,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
