#!/bin/bash
set -e  # Exit on error

# Print Python and pip versions
echo "Python version:"
python3 --version
python3 -m pip --version

# Ensure pip is up to date
echo "Upgrading pip..."
python3 -m pip install --upgrade pip

# Install dependencies with specific Python version
echo "Installing dependencies..."
python3 -m pip install --no-cache-dir -r requirements.txt

# Verify critical installations
echo "Verifying installations..."
python3 -m pip list | grep -E "streamlit|langchain|faiss-cpu|transformers|torch"

echo "Build script completed successfully"
