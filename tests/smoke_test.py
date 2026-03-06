"""Smoke test for all model variants and metrics."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_metrics():
    print("=== Testing Metrics ===")
    from src.evaluation.metrics import compute_all_metrics, format_metrics_table
    import numpy as np

    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    y_pred = np.array([0, 1, 2, 0, 2, 1, 0, 1, 0, 1])
    y_prob = np.random.rand(10, 3)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)

    metrics = compute_all_metrics(y_true, y_pred, y_prob, ['NORM', 'MI', 'STTC'])

    excluded = ('per_class_metrics', 'confusion_matrix', 'class_names')
    n_keys = len([k for k in metrics if k not in excluded])
    print(f"  Total metric keys: {n_keys}")
    assert n_keys >= 25, f"Expected >= 25 metrics, got {n_keys}"

    for key in ['accuracy', 'balanced_accuracy', 'MCC', 'cohens_kappa',
                'FPR', 'FNR', 'FDR', 'FOR', 'informedness_J', 'markedness_MK',
                'DOR', 'LR_plus', 'LR_minus', 'PPV', 'NPV', 'hamming_loss',
                'threat_score_CSI', 'prevalence_threshold', 'auc_roc']:
        assert key in metrics, f"Missing metric: {key}"

    assert 'per_class_metrics' in metrics
    assert len(metrics['per_class_metrics']) == 3

    print(f"  accuracy: {metrics['accuracy']:.4f}")
    print(f"  MCC: {metrics['MCC']:.4f}")
    print(f"  Cohen's Kappa: {metrics['cohens_kappa']:.4f}")
    print("  METRICS TEST PASSED\n")


def test_snn_imports():
    print("=== Testing SNN Imports ===")
    from src.models.snn.spiking_resnet import SpikingResNet, QuadraticIF, create_spiking_resnet
    from src.models.snn.spiking_vit import SpikingVisionTransformer, create_spiking_vit
    print("  All SNN imports OK\n")


def test_snn_forward():
    print("=== Testing SNN Forward Passes ===")
    import torch
    from src.models.snn.spiking_resnet import create_spiking_resnet
    from src.models.snn.spiking_vit import create_spiking_vit

    x = torch.randn(2, 3, 224, 224)

    # ResNet LIF
    model = create_spiking_resnet('resnet18', num_classes=5, neuron_type='lif', num_timesteps=2)
    out = model(x)
    assert out.shape == (2, 5), f"Expected (2, 5), got {out.shape}"
    print(f"  SNN-ResNet18-LIF: {out.shape} OK")

    # ResNet QIF
    model = create_spiking_resnet('resnet18', num_classes=5, neuron_type='qif', num_timesteps=2)
    out = model(x)
    assert out.shape == (2, 5), f"Expected (2, 5), got {out.shape}"
    print(f"  SNN-ResNet18-QIF: {out.shape} OK")

    # ViT LIF
    model = create_spiking_vit(num_classes=5, neuron_type='lif', variant='tiny', num_timesteps=2)
    out = model(x)
    assert out.shape == (2, 5), f"Expected (2, 5), got {out.shape}"
    print(f"  SNN-ViT-LIF:     {out.shape} OK")

    # ViT QIF
    model = create_spiking_vit(num_classes=5, neuron_type='qif', variant='tiny', num_timesteps=2)
    out = model(x)
    assert out.shape == (2, 5), f"Expected (2, 5), got {out.shape}"
    print(f"  SNN-ViT-QIF:     {out.shape} OK")

    print("  SNN FORWARD TESTS PASSED\n")


def test_quantum_imports():
    print("=== Testing Quantum Imports ===")
    from src.models.quantum.hybrid_cnn import (
        EfficientHybridCNN, create_hybrid_quantum_cnn,
        VALID_ROTATIONS, VALID_ENTANGLEMENTS, get_quantum_device,
    )
    assert len(VALID_ROTATIONS) == 7
    assert len(VALID_ENTANGLEMENTS) == 5
    print(f"  Rotations: {VALID_ROTATIONS}")
    print(f"  Entanglements: {VALID_ENTANGLEMENTS}")
    print("  Quantum imports OK\n")


def test_quantum_forward():
    print("=== Testing Quantum Forward Pass ===")
    import torch
    from src.models.quantum.hybrid_cnn import create_hybrid_quantum_cnn

    x = torch.randn(2, 3, 224, 224)

    # Test one combo: ring + RY
    model = create_hybrid_quantum_cnn(
        model_name='efficient',
        num_classes=5,
        entanglement_type='ring',
        rotation_type='RY',
    )
    out = model(x)
    assert out.shape == (2, 5), f"Expected (2, 5), got {out.shape}"
    print(f"  Quantum-ring-RY: {out.shape} OK")
    print("  QUANTUM FORWARD TEST PASSED\n")


def test_swin_import():
    print("=== Testing Swin Import ===")
    from src.models.transformer.swin import SwinECGClassifier, create_swin_classifier
    print("  Swin imports OK\n")


def test_pipeline_import():
    print("=== Testing Pipeline Import ===")
    from pipeline import EXPERIMENT_DEFS, create_model, verify_dataset
    n_exp = len(EXPERIMENT_DEFS)
    print(f"  Total experiments defined: {n_exp}")
    assert n_exp == 19, f"Expected 19 experiments, got {n_exp}"
    print("  Pipeline import OK\n")


if __name__ == '__main__':
    print("=" * 60)
    print("  SMOKE TEST SUITE")
    print("=" * 60)
    print()

    test_metrics()
    test_snn_imports()
    test_snn_forward()
    test_quantum_imports()
    test_quantum_forward()
    test_swin_import()
    test_pipeline_import()

    print("=" * 60)
    print("  ALL SMOKE TESTS PASSED!")
    print("=" * 60)
