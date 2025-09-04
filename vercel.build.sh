#!/bin/bash
set -e  # Exit on error

echo "Python version:"
python --version

echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Installing dependencies..."
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt --no-deps

echo "Installing PyTorch..."
python -m pip install torch==1.13.1+cpu torchvision==0.14.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu

echo "Verifying installations..."
python -m pip list

echo "Build script completed successfully"
