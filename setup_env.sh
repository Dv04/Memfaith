#!/bin/bash
set -e

echo "Setting up MemFaith environment..."
if conda info --envs | grep -q memfaith; then
    echo "Environment memfaith already exists. Updating..."
    conda env update -f environment.yml --prune
else
    echo "Creating new conda environment..."
    conda env create -f environment.yml
fi

echo "Installing SpaCy english model..."
# Explicitly use the conda env's python
conda run -n memfaith python -m spacy download en_core_web_sm

echo "MemFaith environment setup complete!"
echo "Activation: conda activate memfaith"
