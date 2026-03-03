"""
Baseline CNN models for ECG classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Simple CNN baseline for ECG classification
    Architecture: 3 conv blocks + 2 FC layers
    """
    
    def __init__(self, num_classes=4, in_channels=3):
        super(SimpleCNN, self).__init__()
        
        self.num_classes = num_classes
        
        # Conv Block 1
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Conv Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Conv Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Calculate flattened size
        # After 3 pooling layers: 224 -> 112 -> 56 -> 28
        self.flat_features = 128 * 28 * 28
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flat_features, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Conv Block 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # Conv Block 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Conv Block 3
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        
        return x
    
    def get_model_size(self):
        """Calculate model size in MB"""
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024**2
        return size_mb


class DeepCNN(nn.Module):
    """
    Deeper CNN with more layers
    Architecture: 6 conv blocks + 3 FC layers
    """
    
    def __init__(self, num_classes=4, in_channels=3):
        super(DeepCNN, self).__init__()
        
        self.num_classes = num_classes
        
        # Conv Block 1
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Conv Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Conv Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Conv Block 4
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Conv Block 5
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        
        # Conv Block 6
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Calculate flattened size: 224 -> 112 -> 56 -> 28
        self.flat_features = 256 * 28 * 28
        
        # FC layers
        self.fc1 = nn.Linear(self.flat_features, 1024)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Block 1-2
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(F.relu(self.bn2(self.conv2(x))))
        
        # Block 3-4
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool2(F.relu(self.bn4(self.conv4(x))))
        
        # Block 5-6
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool3(F.relu(self.bn6(self.conv6(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x
    
    def get_model_size(self):
        """Calculate model size in MB"""
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024**2
        return size_mb


def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def test_models():
    """Test model architectures"""
    print("=" * 70)
    print("🧪 Testing CNN Models")
    print("=" * 70)
    
    # Test input
    batch_size = 4
    channels = 3
    height, width = 224, 224
    num_classes = 4
    
    x = torch.randn(batch_size, channels, height, width)
    
    # Test SimpleCNN
    print("\n📊 SimpleCNN:")
    model1 = SimpleCNN(num_classes=num_classes)
    output1 = model1(x)
    total_params, trainable_params = count_parameters(model1)
    
    print(f"   Input shape:  {x.shape}")
    print(f"   Output shape: {output1.shape}")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: {model1.get_model_size():.2f} MB")
    
    # Test DeepCNN
    print("\n📊 DeepCNN:")
    model2 = DeepCNN(num_classes=num_classes)
    output2 = model2(x)
    total_params, trainable_params = count_parameters(model2)
    
    print(f"   Input shape:  {x.shape}")
    print(f"   Output shape: {output2.shape}")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: {model2.get_model_size():.2f} MB")
    
    print("\n✅ Model tests complete!")


if __name__ == "__main__":
    test_models()