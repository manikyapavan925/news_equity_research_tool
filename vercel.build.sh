#!/bin/bash
set -e  # Exit on error

# Verify Python version
if ! command -v python3.9 &> /dev/null; then
    echo "Python 3.9 is required but not installed."
    exit 1
fi

echo "Python version:"
python3.9 --version

# Upgrade pip and install base requirements
python3.9 -m pip install --upgrade pip setuptools wheel

# Install PyTorch CPU first
python3.9 -m pip install torch==2.0.0+cpu torchvision==0.15.1+cpu --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
python3.9 -m pip install -r requirements.txt --no-cache-dir

echo "Installation completed"
