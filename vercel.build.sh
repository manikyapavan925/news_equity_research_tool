#!/bin/bash
set -e  # Exit on error

echo "Python version:"
python --version

echo "Pip version:"
pip --version

echo "Installing dependencies..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

echo "Verifying installations..."
pip list

echo "Build script completed successfully"
