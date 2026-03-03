"""
Test script to verify all required packages are installed correctly
"""

import sys

def test_import(package_name, import_name=None):
    """Test if a package can be imported"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✓ {package_name:30s} - OK")
        return True
    except ImportError as e:
        print(f"✗ {package_name:30s} - FAILED: {str(e)}")
        return False

print("=" * 70)
print("Testing Package Installations")
print("=" * 70)

# Core packages
core_packages = [
    ("PyTorch", "torch"),
    ("TorchVision", "torchvision"),
    ("NumPy", "numpy"),
    ("Pandas", "pandas"),
    ("Matplotlib", "matplotlib"),
    ("Seaborn", "seaborn"),
    ("Pillow", "PIL"),
    ("OpenCV", "cv2"),
    ("tqdm", "tqdm"),
]

print("\n📦 Core Packages:")
core_results = [test_import(name, imp) for name, imp in core_packages]

# ML packages
ml_packages = [
    ("scikit-learn", "sklearn"),
    ("SciPy", "scipy"),
    ("Timm", "timm"),
    ("Einops", "einops"),
]

print("\n🤖 Machine Learning Packages:")
ml_results = [test_import(name, imp) for name, imp in ml_packages]

# SNN packages
snn_packages = [
    ("snnTorch", "snntorch"),
]

print("\n⚡ Spiking Neural Network Packages:")
snn_results = [test_import(name, imp) for name, imp in snn_packages]

# Quantum packages
quantum_packages = [
    ("PennyLane", "pennylane"),
]

print("\n🔬 Quantum Computing Packages:")
quantum_results = [test_import(name, imp) for name, imp in quantum_packages]

# Medical packages
medical_packages = [
    ("WFDB", "wfdb"),
    ("NeuroKit2", "neurokit2"),
]

print("\n🏥 Medical Signal Processing Packages:")
medical_results = [test_import(name, imp) for name, imp in medical_packages]

# Utility packages
utility_packages = [
    ("TensorBoard", "tensorboard"),
    ("YAML", "yaml"),
    ("Jupyter", "jupyter"),
]

print("\n🛠️  Utility Packages:")
utility_results = [test_import(name, imp) for name, imp in utility_packages]

# Summary
print("\n" + "=" * 70)
total_packages = len(core_packages) + len(ml_packages) + len(snn_packages) + \
                 len(quantum_packages) + len(medical_packages) + len(utility_packages)
total_success = sum(core_results) + sum(ml_results) + sum(snn_results) + \
                sum(quantum_results) + sum(medical_results) + sum(utility_results)

print(f"📊 Summary: {total_success}/{total_packages} packages installed successfully")

if total_success == total_packages:
    print("✅ All packages installed correctly! You're ready to go! 🚀")
else:
    print(f"⚠️  {total_packages - total_success} package(s) missing. Please install them.")
    print("\nMissing packages can be installed with:")
    print("pip install <package_name>")

print("=" * 70)

# Additional system info
print("\n🖥️  System Information:")
print(f"Python version: {sys.version}")

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
except:
    pass