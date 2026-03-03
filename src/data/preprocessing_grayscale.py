"""
Grayscale preprocessing to match the reference paper
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import shutil


class GrayscaleECGPreprocessor:
    """
    Preprocess ECG images to match the reference paper:
    - 64x64 size
    - Grayscale
    - Normalize to [-1, 1]
    """
    
    def __init__(self, raw_data_path, processed_data_path, img_size=(64, 64)):
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        self.img_size = img_size
        
        self.classes = ["MI", "History_MI", "Abnormal_Heartbeat", "Normal"]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
    def split_dataset(self, test_size=0.15, val_size=0.15, random_state=42):
        """Split dataset into train/val/test"""
        print("\n✂️  Splitting dataset...")
        
        # Create metadata
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
        
        # Stratified split
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, stratify=df['class_name'], random_state=random_state
        )
        
        val_size_adjusted = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_size_adjusted, 
            stratify=train_val_df['class_name'], random_state=random_state
        )
        
        # Save splits
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(self.processed_data_path / "train_split.csv", index=False)
        val_df.to_csv(self.processed_data_path / "val_split.csv", index=False)
        test_df.to_csv(self.processed_data_path / "test_split.csv", index=False)
        
        print(f"\n   Train: {len(train_df)} images")
        print(f"   Val:   {len(val_df)} images")
        print(f"   Test:  {len(test_df)} images")
        
        return train_df, val_df, test_df
    
    def preprocess_and_save(self):
        """
        Preprocess images to match reference paper:
        - Convert to grayscale
        - Resize to 64x64
        - Normalize to [-1, 1] range
        """
        print("\n🔄 Preprocessing images (grayscale, 64x64)...")
        
        splits = {
            'train': pd.read_csv(self.processed_data_path / "train_split.csv"),
            'val': pd.read_csv(self.processed_data_path / "val_split.csv"),
            'test': pd.read_csv(self.processed_data_path / "test_split.csv")
        }
        
        for split_name, split_df in splits.items():
            print(f"\n📁 Processing {split_name} split...")
            
            for class_name in self.classes:
                class_dir = self.processed_data_path / split_name / class_name
                class_dir.mkdir(parents=True, exist_ok=True)
                
                class_images = split_df[split_df['class_name'] == class_name]
                
                for _, row in tqdm(class_images.iterrows(), 
                                  total=len(class_images),
                                  desc=f"   {class_name}"):
                    src_path = Path(row['image_path'])
                    dst_path = class_dir / row['filename']
                    
                    try:
                        # Load image
                        img = Image.open(src_path)
                        
                        # Convert to grayscale
                        img = img.convert('L')  # 'L' mode = grayscale
                        
                        # Resize to 64x64
                        img = img.resize(self.img_size, Image.Resampling.LANCZOS)
                        
                        # Save
                        img.save(dst_path)
                        
                    except Exception as e:
                        print(f"\n⚠️  Error processing {src_path}: {e}")
        
        print("\n✅ Preprocessing complete!")


def main():
    """Main preprocessing pipeline matching reference paper"""
    
    print("=" * 70)
    print("🚀 ECG Grayscale Preprocessing (Reference Paper Format)")
    print("=" * 70)
    
    raw_path = Path("data/raw/kaggle_ecg")
    processed_path = Path("data/processed/kaggle_ecg_grayscale")
    
    preprocessor = GrayscaleECGPreprocessor(
        raw_data_path=raw_path,
        processed_data_path=processed_path,
        img_size=(64, 64)
    )
    
    # Split dataset
    preprocessor.split_dataset(test_size=0.15, val_size=0.15, random_state=42)
    
    # Preprocess and save
    preprocessor.preprocess_and_save()
    
    print("\n" + "=" * 70)
    print("✅ Preprocessing complete!")
    print(f"   Output: {processed_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()