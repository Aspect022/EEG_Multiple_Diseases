"""
Data preprocessing utilities for ECG image classification
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from collections import Counter
import shutil
from tqdm import tqdm


class ECGDataPreprocessor:
    """
    Preprocess ECG images for training
    """
    
    def __init__(self, raw_data_path, processed_data_path, img_size=(224, 224)):
        """
        Args:
            raw_data_path: Path to raw kaggle_ecg folder
            processed_data_path: Path to save processed data
            img_size: Target image size (height, width)
        """
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        self.img_size = img_size
        
        self.classes = ["MI", "History_MI", "Abnormal_Heartbeat", "Normal"]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
    def analyze_dataset(self):
        """Analyze the raw dataset"""
        print("=" * 70)
        print("📊 Dataset Analysis")
        print("=" * 70)
        
        class_distribution = {}
        all_sizes = []
        
        for class_name in self.classes:
            class_path = self.raw_data_path / class_name
            
            if not class_path.exists():
                print(f"⚠️  Warning: {class_path} not found!")
                continue
            
            # Get all images
            images = list(class_path.glob("*.png")) + list(class_path.glob("*.jpg"))
            class_distribution[class_name] = len(images)
            
            # Sample some images to get sizes
            for img_path in images[:20]:
                try:
                    img = Image.open(img_path)
                    all_sizes.append(img.size)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
        
        # Print distribution
        print("\n📈 Class Distribution:")
        total_images = sum(class_distribution.values())
        for class_name, count in class_distribution.items():
            percentage = (count / total_images) * 100
            print(f"   {class_name:20s}: {count:4d} images ({percentage:5.2f}%)")
        
        print(f"\n   Total: {total_images} images")
        
        # Print size analysis
        size_counter = Counter(all_sizes)
        print(f"\n🖼️  Image Sizes Found (from sample):")
        for size, count in size_counter.most_common(5):
            print(f"   {size[0]}x{size[1]}: {count} images")
        
        return class_distribution
    
    def create_metadata_csv(self):
        """Create a CSV file with all image paths and labels"""
        print("\n📝 Creating metadata CSV...")
        
        data_list = []
        
        for class_name in self.classes:
            class_path = self.raw_data_path / class_name
            
            if not class_path.exists():
                continue
            
            images = list(class_path.glob("*.png")) + list(class_path.glob("*.jpg"))
            
            for img_path in images:
                data_list.append({
                    'image_path': str(img_path),
                    'class_name': class_name,
                    'class_idx': self.class_to_idx[class_name],
                    'filename': img_path.name
                })
        
        df = pd.DataFrame(data_list)
        
        # Save to processed data path
        csv_path = self.processed_data_path / "metadata.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        
        print(f"✓ Metadata saved to {csv_path}")
        print(f"   Total entries: {len(df)}")
        
        return df
    
    def split_dataset(self, test_size=0.15, val_size=0.15, random_state=42):
        """
        Split dataset into train/val/test
        
        Args:
            test_size: Proportion for test set
            val_size: Proportion for validation set (from remaining after test)
            random_state: Random seed for reproducibility
        """
        print("\n✂️  Splitting dataset...")
        
        # Load or create metadata
        metadata_path = self.processed_data_path / "metadata.csv"
        if metadata_path.exists():
            df = pd.read_csv(metadata_path)
        else:
            df = self.create_metadata_csv()
        
        # Stratified split
        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            stratify=df['class_name'],
            random_state=random_state
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)  # Adjust val size
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size_adjusted,
            stratify=train_val_df['class_name'],
            random_state=random_state
        )
        
        # Save splits
        train_df.to_csv(self.processed_data_path / "train_split.csv", index=False)
        val_df.to_csv(self.processed_data_path / "val_split.csv", index=False)
        test_df.to_csv(self.processed_data_path / "test_split.csv", index=False)
        
        # Print split info
        print(f"\n   Train: {len(train_df)} images ({len(train_df)/len(df)*100:.1f}%)")
        print(f"   Val:   {len(val_df)} images ({len(val_df)/len(df)*100:.1f}%)")
        print(f"   Test:  {len(test_df)} images ({len(test_df)/len(df)*100:.1f}%)")
        
        # Print class distribution for each split
        for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            print(f"\n   {split_name} distribution:")
            for class_name in self.classes:
                count = len(split_df[split_df['class_name'] == class_name])
                print(f"      {class_name:20s}: {count:3d}")
        
        return train_df, val_df, test_df
    
    def preprocess_and_copy(self, resize=True, normalize=False):
        """
        Preprocess images and organize into train/val/test folders
        
        Args:
            resize: Whether to resize images
            normalize: Whether to normalize (not recommended for saving as images)
        """
        print("\n🔄 Preprocessing and organizing images...")
        
        # Load splits
        splits = {
            'train': pd.read_csv(self.processed_data_path / "train_split.csv"),
            'val': pd.read_csv(self.processed_data_path / "val_split.csv"),
            'test': pd.read_csv(self.processed_data_path / "test_split.csv")
        }
        
        for split_name, split_df in splits.items():
            print(f"\n📁 Processing {split_name} split...")
            
            for class_name in self.classes:
                # Create class directory
                class_dir = self.processed_data_path / split_name / class_name
                class_dir.mkdir(parents=True, exist_ok=True)
                
                # Get images for this class
                class_images = split_df[split_df['class_name'] == class_name]
                
                for _, row in tqdm(class_images.iterrows(), 
                                  total=len(class_images),
                                  desc=f"   {class_name}"):
                    src_path = Path(row['image_path'])
                    dst_path = class_dir / row['filename']
                    
                    if resize:
                        # Load, resize, and save
                        try:
                            img = Image.open(src_path)
                            
                            # Convert to RGB if needed
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                            
                            # Resize
                            img = img.resize(self.img_size, Image.Resampling.LANCZOS)
                            
                            # Save
                            img.save(dst_path)
                        except Exception as e:
                            print(f"\n⚠️  Error processing {src_path}: {e}")
                    else:
                        # Just copy
                        shutil.copy2(src_path, dst_path)
        
        print("\n✅ Preprocessing complete!")
        print(f"   Processed images saved to: {self.processed_data_path}")
    
    def get_dataset_statistics(self):
        """Calculate dataset statistics for normalization"""
        print("\n📊 Calculating dataset statistics...")
        
        train_path = self.processed_data_path / "train"
        
        if not train_path.exists():
            print("⚠️  Train folder not found. Run preprocessing first.")
            return None
        
        # Sample images to calculate mean and std
        all_images = []
        for class_name in self.classes:
            class_path = train_path / class_name
            images = list(class_path.glob("*.png")) + list(class_path.glob("*.jpg"))
            all_images.extend(images[:50])  # Sample 50 per class
        
        pixel_values = []
        
        for img_path in tqdm(all_images, desc="   Analyzing"):
            img = Image.open(img_path)
            img_array = np.array(img) / 255.0  # Normalize to [0, 1]
            pixel_values.append(img_array)
        
        pixel_values = np.concatenate([pv.reshape(-1, 3) for pv in pixel_values])
        
        mean = pixel_values.mean(axis=0)
        std = pixel_values.std(axis=0)
        
        print(f"\n   Mean: [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
        print(f"   Std:  [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")
        
        return {'mean': mean.tolist(), 'std': std.tolist()}


def main():
    """Main preprocessing pipeline"""
    
    print("=" * 70)
    print("🚀 ECG Data Preprocessing Pipeline")
    print("=" * 70)
    
    # Define paths
    raw_path = Path("data/raw/kaggle_ecg")
    processed_path = Path("data/processed/kaggle_ecg")
    
    # Initialize preprocessor
    preprocessor = ECGDataPreprocessor(
        raw_data_path=raw_path,
        processed_data_path=processed_path,
        img_size=(224, 224)  # Standard size for CNNs
    )
    
    # Step 1: Analyze dataset
    preprocessor.analyze_dataset()
    
    # Step 2: Create metadata
    preprocessor.create_metadata_csv()
    
    # Step 3: Split dataset
    preprocessor.split_dataset(test_size=0.15, val_size=0.15, random_state=42)
    
    # Step 4: Preprocess and organize
    preprocessor.preprocess_and_copy(resize=True)
    
    # Step 5: Calculate statistics
    stats = preprocessor.get_dataset_statistics()
    
    # Save statistics
    if stats:
        import yaml
        stats_path = processed_path / "dataset_statistics.yaml"
        with open(stats_path, 'w') as f:
            yaml.dump(stats, f)
        print(f"\n✓ Statistics saved to {stats_path}")
    
    print("\n" + "=" * 70)
    print("✅ Preprocessing pipeline complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()