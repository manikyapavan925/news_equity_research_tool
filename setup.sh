#!/bin/bash

# News Equity Research Tool - Local Setup Script
echo "🚀 Setting up News Equity Research Tool..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Set environment variables
export PYTORCH_NO_CUDA=1
export TORCH_DEVICE=cpu

echo "✅ Setup complete!"
echo ""
echo "🎯 To run the application:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Run the app: streamlit run src/app.py"
echo ""
echo "🌐 The app will open in your browser at http://localhost:8501"
