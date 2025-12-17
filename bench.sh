#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "========================================="
echo "Loading environment variables from .env"
echo "========================================="
source .env

# Verify CUDA_HOME is set and is a directory
if [ -z "$CUDA_HOME" ] || [ ! -d "$CUDA_HOME" ]; then
    echo "Error: CUDA_HOME is not set or is not a valid directory. Please check your .env file."
    exit 1
fi

# Verify CUDA is found
if ! command -v nvcc &> /dev/null
then
    echo "nvcc could not be found. Please check your .env file."
    exit 1
fi
nvcc --version

echo "========================================="
echo "Setting up results directory"
echo "========================================="
mkdir -p results

echo "========================================="
echo "Compiling Custom CUDA Kernel"
echo "========================================="
(cd kernel && uv run python setup.py install)

echo "========================================="
echo "Running Baseline Benchmark"
echo "========================================="
"$CUDA_HOME/bin/nsys" profile -o results/baseline_profile.nsys-rep uv run python base/inference_vit_pytorch.py

echo "========================================="
echo "Running Custom Kernel Benchmark"
echo "========================================="
"$CUDA_HOME/bin/nsys" profile -o results/custom_kernel_profile.nsys-rep uv run python kernel/vit_with_custom_cuda.py

echo "========================================="
echo "Benchmarking Complete."
echo "Reports generated in results/ directory:"
echo "- baseline_profile.nsys-rep"
echo "- custom_kernel_profile.nsys-rep"
echo "========================================="
