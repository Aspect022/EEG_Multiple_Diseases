import os
import shutil
from pathlib import Path
from tqdm import tqdm

def reorganize_kaggle_dataset():
    """
    Reorganize the Kaggle ECG dataset into a cleaner structure
    """
    
    # Define source directories (your current structure)
    source_mapping = {
        "MI": "ECG Images of Myocardial Infarction Patients (240x12=2880)",
        "History_MI": "ECG Images of Patient that have History of MI (172x12=2064)",
        "Abnormal_Heartbeat": "ECG Images of Patient that have abnormal heartbeat (233x12=2796)",
        "Normal": "Normal Person ECG Images (284x12=3408)"
    }
    
    base_source = Path("Datasets")
    base_dest = Path("data/raw/kaggle_ecg")
    
    print("Reorganizing Kaggle ECG dataset...")
    print("=" * 50)
    
    for dest_name, source_name in source_mapping.items():
        source_dir = base_source / source_name
        dest_dir = base_dest / dest_name
        
        if not source_dir.exists():
            print(f"⚠️  Warning: {source_dir} not found, skipping...")
            continue
        
        # Create destination directory
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_files = list(source_dir.glob("*.png")) + \
                     list(source_dir.glob("*.jpg")) + \
                     list(source_dir.glob("*.jpeg"))
        
        print(f"\n📁 Processing {dest_name}:")
        print(f"   Source: {source_dir}")
        print(f"   Destination: {dest_dir}")
        print(f"   Files found: {len(image_files)}")
        
        # Copy files with progress bar
        for img_file in tqdm(image_files, desc=f"   Copying {dest_name}"):
            dest_file = dest_dir / img_file.name
            if not dest_file.exists():
                shutil.copy2(img_file, dest_file)
        
        print(f"   ✓ Completed!")
    
    print("\n" + "=" * 50)
    print("✓ Dataset reorganization complete!")
    
    # Print summary
    print("\n📊 Dataset Summary:")
    for class_name in source_mapping.keys():
        class_dir = base_dest / class_name
        if class_dir.exists():
            num_images = len(list(class_dir.glob("*.png")) + 
                           list(class_dir.glob("*.jpg")) + 
                           list(class_dir.glob("*.jpeg")))
            print(f"   {class_name}: {num_images} images")

def move_ptbxl_dataset():
    """
    Move PTB-XL dataset to proper location
    """
    source = Path("physionet.org/files/ptb-xl/1.0.3")
    dest = Path("data/raw/ptb_xl")
    
    if source.exists():
        print("\nMoving PTB-XL dataset...")
        dest.parent.mkdir(parents=True, exist_ok=True)
        
        if not dest.exists():
            shutil.move(str(source), str(dest))
            print("✓ PTB-XL dataset moved successfully!")
        else:
            print("⚠️  PTB-XL destination already exists, skipping...")
    else:
        print("⚠️  PTB-XL source not found (it may still be downloading)")

if __name__ == "__main__":
    print("🚀 Starting dataset reorganization...\n")
    
    # Reorganize Kaggle dataset
    reorganize_kaggle_dataset()
    
    # Move PTB-XL (if download complete)
    move_ptbxl_dataset()
    
    print("\n✅ All done! Your data is now organized.")