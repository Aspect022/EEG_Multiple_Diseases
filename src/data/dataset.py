"""
PyTorch Dataset classes for ECG image classification
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from pathlib import Path
import numpy as np


class ECGImageDataset(Dataset):
    """
    PyTorch Dataset for ECG images
    """
    
    def __init__(self, data_path, split='train', transform=None, img_size=(224, 224)):
        """
        Args:
            data_path: Path to processed data directory
            split: 'train', 'val', or 'test'
            transform: Optional torchvision transforms
            img_size: Image size (height, width)
        """
        self.data_path = Path(data_path)
        self.split = split
        self.img_size = img_size
        
        # Load split CSV
        csv_path = self.data_path / f"{split}_split.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Split file not found: {csv_path}")
        
        self.data_df = pd.read_csv(csv_path)
        
        # Class mapping
        self.classes = ["MI", "History_MI", "Abnormal_Heartbeat", "Normal"]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        # Update paths to point to processed folder structure
        self.data_df['processed_path'] = self.data_df.apply(
            lambda row: self.data_path / split / row['class_name'] / row['filename'],
            axis=1
        )
        
        # Set transform
        if transform is None:
            self.transform = self.get_default_transform()
        else:
            self.transform = transform
    
    def get_default_transform(self):
        """Get default transforms based on split"""
        if self.split == 'train':
            # Training: data augmentation
            return transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            # Val/Test: no augmentation
            return transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx):
        """Get a single item"""
        row = self.data_df.iloc[idx]
        
        # Load image
        img_path = row['processed_path']
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = row['class_idx']
        
        return image, label
    
    def get_class_weights(self):
        """Calculate class weights for imbalanced dataset"""
        class_counts = self.data_df['class_idx'].value_counts().sort_index()
        total = len(self.data_df)
        weights = total / (len(class_counts) * class_counts.values)
        return torch.FloatTensor(weights)


def get_dataloaders(data_path, batch_size=32, num_workers=4, img_size=(224, 224)):
    """
    Create train, validation, and test dataloaders
    
    Args:
        data_path: Path to processed data directory
        batch_size: Batch size
        num_workers: Number of workers for data loading (set to 0 for CPU)
        img_size: Image size
        
    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    
    # Create datasets
    train_dataset = ECGImageDataset(
        data_path=data_path,
        split='train',
        img_size=img_size
    )
    
    val_dataset = ECGImageDataset(
        data_path=data_path,
        split='val',
        img_size=img_size
    )
    
    test_dataset = ECGImageDataset(
        data_path=data_path,
        split='test',
        img_size=img_size
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False  # CPU only
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    print(f"✓ DataLoaders created:")
    print(f"   Train: {len(train_dataset)} images, {len(train_loader)} batches")
    print(f"   Val:   {len(val_dataset)} images, {len(val_loader)} batches")
    print(f"   Test:  {len(test_dataset)} images, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader, train_dataset.classes


def main():
    """Test the dataset and dataloader"""
    print("=" * 70)
    print("🧪 Testing Dataset and DataLoader")
    print("=" * 70)
    
    data_path = Path("data/processed/kaggle_ecg")
    
    # Create dataloaders
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        data_path=data_path,
        batch_size=16,
        num_workers=0,  # Use 0 for CPU
        img_size=(224, 224)
    )
    
    # Test a batch
    print("\n📦 Testing a batch:")
    images, labels = next(iter(train_loader))
    print(f"   Image batch shape: {images.shape}")
    print(f"   Label batch shape: {labels.shape}")
    print(f"   Label values: {labels}")
    print(f"   Classes: {classes}")
    
    # Print class distribution
    print("\n📊 Class Distribution in Train Set:")
    train_dataset = train_loader.dataset
    class_counts = train_dataset.data_df['class_name'].value_counts()
    for class_name, count in class_counts.items():
        print(f"   {class_name:20s}: {count:3d}")
    
    # Get class weights
    weights = train_dataset.get_class_weights()
    print(f"\n⚖️  Class Weights: {weights}")
    
    print("\n✅ Dataset test complete!")


if __name__ == "__main__":
    main()