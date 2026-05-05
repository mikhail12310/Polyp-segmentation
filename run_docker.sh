#!/bin/bash

# Exit on error
set -e

echo "Building IllumiSeg Docker image..."
docker build -t illumiseg-project .

echo ""
echo "==========================================================="
echo "Docker image built successfully! 🐳"
echo "==========================================================="
echo ""
echo "To reproduce the final evaluation results, running:"
docker run --rm illumiseg-project python evaluate.py

# Note: If you want to run training instead, you can run:
# docker run --rm illumiseg-project python train.py
