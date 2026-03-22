import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch

from sleep_edf_pipeline import resolve_models, validate_model_forward
from src.data.sleep_edf_dataset import (
    SleepEDFDataset,
    create_sleep_edf_multimodal_dataloaders,
    verify_sleep_edf_dataset,
)
from src.data.transforms import create_scalogram_transform


def main():
    data_dir = "data/sleep-edf"
    if not verify_sleep_edf_dataset(data_dir):
        print(f"Sleep-EDF dataset missing at {data_dir}")
        return

    print("=== Sleep-EDF Dataset Test ===")
    ds = SleepEDFDataset(data_dir=data_dir, split="train")
    print(f"Train epochs: {len(ds)}")
    print(f"Class distribution: {ds.get_class_distribution()}")
    if len(ds) > 0:
        x, y = ds[0]
        print(f"Raw sample shape: {tuple(x.shape)} | label={y}")
        assert tuple(x.shape) == (6, 3000)

    print("\n=== Sleep-EDF Multimodal Loader Test ===")
    transform = create_scalogram_transform(output_size=(224, 224), sampling_rate=100)
    train_loader, val_loader, _ = create_sleep_edf_multimodal_dataloaders(
        data_dir=data_dir,
        batch_size=2,
        scalogram_transform=transform,
        max_records=1,
    )
    raw_signal, scalogram, labels = next(iter(train_loader))
    print(f"Raw batch: {tuple(raw_signal.shape)}")
    print(f"Scalogram batch: {tuple(scalogram.shape)}")
    print(f"Labels: {tuple(labels.shape)}")
    assert tuple(raw_signal.shape[1:]) == (6, 3000)
    assert tuple(scalogram.shape[1:]) == (3, 224, 224)

    print("\n=== Sleep-EDF Model Validation Smoke Test ===")
    for model_key in ["tcanet", "snn_1d_lif", "convnext", "fusion_c"]:
        status, message = validate_model_forward(model_key, resolve_models(model_key)[model_key])
        print(f"{model_key}: {status} - {message}")
        if status == "FAIL":
            raise AssertionError(f"{model_key} failed validation: {message}")

    print("\nAll Sleep-EDF smoke tests passed.")


if __name__ == "__main__":
    main()
