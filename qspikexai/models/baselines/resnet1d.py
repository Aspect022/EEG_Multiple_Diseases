import torch
import torch.nn as nn
import torch.nn.functional as F
from ..qspikexai_net import TASK_CHANNELS, TASK_SEQ_LEN

class ResNetBasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet1D(nn.Module):
    """
    1D ResNet18 adapted for 1D EEG classification.
    """

    def __init__(self, task: str, base_filters: int = 64):
        super().__init__()
        self.task = task
        self.in_channels = TASK_CHANNELS[task]
        self.seq_len = TASK_SEQ_LEN[task]
        
        self.in_planes = base_filters

        self.conv1 = nn.Conv1d(self.in_channels, base_filters, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(base_filters)
        
        self.layer1 = self._make_layer(base_filters, 2, stride=1)
        self.layer2 = self._make_layer(base_filters * 2, 2, stride=2)
        self.layer3 = self._make_layer(base_filters * 4, 2, stride=2)
        self.layer4 = self._make_layer(base_filters * 8, 2, stride=2)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(base_filters * 8 * ResNetBasicBlock1D.expansion, 4 if task == 'sleep_apnea' else 2)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(ResNetBasicBlock1D(self.in_planes, planes, s))
            self.in_planes = planes * ResNetBasicBlock1D.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, T)
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        h = self.pool(h).squeeze(-1)
        return self.classifier(h)
