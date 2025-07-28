#!/bin/bash

# COATI Environment Setup
# This script creates a dedicated conda environment for COATI

echo "=== COATI Environment Setup ==="

# Environment name
ENV_NAME="coati"

echo "Creating conda environment: $ENV_NAME"

# Create the conda environment with Python 3.10 (good compatibility)
conda create -n $ENV_NAME python=3.10 -y

echo "Activating environment..."
source activate $ENV_NAME

echo "Installing core dependencies..."

# Install PyTorch first (CPU version for compatibility)
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# Install RDKit via conda (more reliable than pip)
conda install -c conda-forge rdkit -y

# Install other scientific packages
conda install -c conda-forge pandas numpy matplotlib scipy scikit-learn -y

# Install additional dependencies
conda install -c conda-forge jupyter seaborn altair boto3 tqdm -y

echo "Installing COATI package..."
# Go to the main directory and install COATI
cd ..
pip install -e .

echo "Installing additional reaction analysis dependencies..."
pip install requests

echo ""
echo "=== Environment Setup Complete! ==="
echo ""
echo "To activate the environment:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To deactivate:"
echo "  conda deactivate"
echo ""
echo "To remove the environment (if needed):"
echo "  conda env remove -n $ENV_NAME"
echo ""
echo "Environment name: $ENV_NAME"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "RDKit version: $(python -c 'import rdkit; print(rdkit.__version__)')"
echo ""
echo "Ready to run COATI analysis scripts!" 