#!/bin/bash

echo "======================================"
echo "ECG Heart Disease Classification Setup"
echo "======================================"

# Create virtual environment
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_CHANNEL="${TORCH_CHANNEL:-cu121}"
INSTALL_OPTIONAL_QUANTUM="${INSTALL_OPTIONAL_QUANTUM:-0}"

echo "Creating virtual environment..."
$PYTHON_BIN -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch
if [ "$TORCH_CHANNEL" = "cpu" ]; then
    echo "Installing PyTorch (CPU version)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
else
    echo "Installing PyTorch ($TORCH_CHANNEL)..."
    pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/$TORCH_CHANNEL"
fi

# Install other requirements
echo "Installing other dependencies..."
pip install -r requirements.txt

if [ "$INSTALL_OPTIONAL_QUANTUM" = "1" ]; then
    echo "Installing optional PennyLane backends..."
    pip install -r requirements_optional_quantum.txt
fi

echo ""
echo "✓ Environment setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To deactivate, run:"
echo "  deactivate"
