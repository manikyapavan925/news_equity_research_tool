#!/bin/bash
set -e  # Exit on error

# Ensure we're using Python 3.9
PYTHON_VERSION=$(python3.9 --version)
echo "Using Python version: $PYTHON_VERSION"

# Upgrade pip
echo "Upgrading pip..."
python3.9 -m pip install --upgrade pip

# Install base requirements
echo "Installing base requirements..."
python3.9 -m pip install --upgrade setuptools wheel

# Install PyTorch CPU version first
echo "Installing PyTorch CPU..."
python3.9 -m pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
echo "Installing other dependencies..."
python3.9 -m pip install -r requirements.txt

# Verify installations
echo "Verifying critical installations..."
python3.9 -m pip freeze | grep -E "torch|streamlit|langchain|faiss-cpu|transformers"

echo "Build script completed successfully"
