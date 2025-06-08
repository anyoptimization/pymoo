#!/bin/bash

# Build the HTML documentation in the pymoo-doc conda environment

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Building documentation in conda environment: pymoo-doc"

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pymoo-doc

# Change to docs directory
cd "$SCRIPT_DIR"

# Build HTML documentation
echo "Running: make html"
make html

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "Documentation build completed successfully!"
    echo "Output available in: $SCRIPT_DIR/build/html/index.html"
else
    echo "Documentation build failed!"
    exit 1
fi