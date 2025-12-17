# Accelerating Vision Transformer Inference with Custom CUDA Kernels

This is a repository for NTU Parallel Programming Final Project. We aim to accelerate the Self-Attention mechanism in [Vision Transformer (ViT)](https://github.com/lucidrains/vit-pytorch) by implementing different custom CUDA kernels.

## Project Overview

We implemented and compared multiple CUDA kernel versions for the Self-Attention mechanism:

- **baseline**: PyTorch native implementation (uses highly optimized CUTLASS GEMM + cuDNN softmax)
- **v0**: Naive CUDA kernel (single-threaded per output element)
- **v1**: Optimized with shared memory tiling
- **v2**: Further optimized with register blocking
- **v3**: Best performing custom kernel with additional optimizations

## Scalability Benchmark Results

We benchmarked different kernel versions across various image sizes (token counts) to analyze scalability:

| Version | 224px (197 tokens) | 384px (577 tokens) | 512px (1025 tokens) |
|---------|-------------------|-------------------|---------------------|
| baseline | 38.11 µs | 670.62 µs | 2.12 ms |
| v0 | 48.01 ms | 410.74 ms | 1.30 s |
| v1 | 4.68 ms | 39.83 ms | 129.61 ms |
| v2 | 1.92 ms | 15.69 ms | 52.09 ms |
| v3 | 1.42 ms | 11.47 ms | 36.68 ms |

*Table: Average attention kernel time per call (32 batch × 10 iterations × 12 layers × 2 calls/layer = 240 calls)*

### Scalability Analysis

All custom kernels show O(N²) scaling as expected for self-attention:
- Token count ratio: 224→384 = 2.93x, 224→512 = 5.20x
- Theoretical O(N²) growth: 384 = 8.58x, 512 = 27.07x
- Measured growth factors are consistent with O(N²) complexity

### Performance Visualization

#### With all versions
![Scalability Plot with v0](results/scalability/scalability_plot_with_v0.png)

#### Without v0
![Scalability Plot without v0](results/scalability/scalability_plot_no_v0.png)

## Project Structure

```
.
├── base/                           # Baseline implementation
│   ├── inference_vit_pytorch.py    # ViT inference using vit-pytorch
│   ├── train_vit_pytorch.py        # Training script for ViT on CIFAR-10
│   └── vit_cifar10.pth             # Pretrained weights
├── custom/                         
│   ├── v0                          
│   ├── v1                          
│   ├── v2                          
│   └── v3                          
├── data/                           # Dataset directory (CIFAR-10)
├── results/                        # Nsight profiling results
├── vit-pytorch/                    # vit-pytorch library (submodule)
├── bench.sh                        # Benchmark script
└── README.md
```

## Requirements

- Python 3.11+
- PyTorch 2.0+ with CUDA support
- CUDA Toolkit 12.x
- NVIDIA GPU with compute capability 7.0+

## Setup

TBD.

## Usage

TBD.

### Run Benchmark
```bash
bash bench.sh
```
This script will:
1. Compile the custom CUDA kernel
2. Run baseline inference with vit-pytorch (using PyTorch softmax)
3. Run optimized inference with custom CUDA.
4. Generate Nsight Systems profiling reports in `results/`

### Run Individual Scripts

**Baseline inference:**
```bash
python base/inference_vit_pytorch.py
```

**Custom CUDA kernel inference:**

## References

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [vit-pytorch](https://github.com/lucidrains/vit-pytorch)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

## License

This project is for educational purposes (NTU Parallel Programming Course).
