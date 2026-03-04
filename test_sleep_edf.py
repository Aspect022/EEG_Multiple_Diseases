import sys
import os

# Add src to path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from data.sleep_edf_dataset import SleepEDFDataset, create_sleep_edf_dataloaders
from data.transforms import create_scalogram_transform

def test():
    data_dir = "data/sleep-edf"
    
    psg_exists = len(list(os.scandir(data_dir))) > 0 if os.path.exists(data_dir) else False
    if not psg_exists:
        print(f"Data directory {data_dir} is empty.")
        return
        
    print("Testing SleepEDFDataset...")
    transform = create_scalogram_transform(output_size=(224, 224), sampling_rate=100, apply_filter=False)
    
    try:
        ds = SleepEDFDataset(data_dir, split='train', transform=transform)
        print(f"Dataset size: {len(ds)}")
        if len(ds) > 0:
            x, y = ds[0]
            print(f"Signal shape: {x.shape}")
            print(f"Label: {y}")
            
            # Print class distribution
            print(f"Class distribution: {ds.get_class_distribution()}")
        else:
            print("Dataset is empty. Was the EDF properly loaded?")
    except Exception as e:
        print(f"Error loading dataset: {e}")

if __name__ == "__main__":
    test()
