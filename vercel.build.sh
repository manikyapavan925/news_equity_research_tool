#!/bin/bash
set -e  # Exit on error

echo "Python version:"
python3 --version

echo "Upgrading pip..."
python3 -m pip install --upgrade pip

echo "Installing PyTorch CPU version..."
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu

echo "Installing other dependencies..."
python3 -m pip install -r requirements.txt --no-cache-dir

echo "Verifying installations..."
python3 -m pip freeze

echo "Build script completed successfully"
