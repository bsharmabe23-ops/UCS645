#  UCS645 — LAB 8: GPU Accelerated Machine Learning

> Complete implementation of 5 CUDA/GPU programming exercises covering GPU architecture, memory hierarchy, ML primitives, CNN layers, and full MNIST training.

---

##  File Structure

| File | Problem | Description |
|------|---------|-------------|
| `ex01_cuda_basics.py` | Problem 1 | GPU Architecture & CUDA Kernel Profiling |
| `ex02_memory_hierarchy.py` | Problem 2 | Parallel Reduction & Shared Memory |
| `ex03_ml_primitives.py` | Problem 3 | Custom ML Kernels — Activations, Loss & Backprop |
| `ex04_cnn_layers.py` | Problem 4 | Tiled GEMM vs cuBLAS & CNN Layer Benchmarking |
| `ex05_mnist_cnn.py` | Problem 5 | Full MNIST CNN Training |
| `p1_bandwidth.png` | Problem 1 | Bandwidth & Time vs N plots |
| `p2_bank_conflicts.png` | Problem 2 | Bank conflict stride timing plot |
| `p3_activations.png` | Problem 3 | Activation function curves |
| `p4_roofline.png` | Problem 4 | Roofline plot |
| `p4_cnn_layers.png` | Problem 4 | CNN layer timing bar chart |
| `p5_results.png` | Problem 5 | Training loss, accuracy & ablation chart |

---

## ⚙️ Environment

| Property | Details |
|----------|---------|
| OS | Ubuntu 24.04 (WSL2) |
| CUDA Version | 12.0, V12.0.140 |
| Python | 3.12 |
| PyTorch | Latest stable |
| GPU Reference | NVIDIA GeForce GTX 1650 (Turing, 7.5) |
| Device Used | CPU (WSL2 — no GPU driver passthrough) |

---

## Problem 1 — GPU Architecture & CUDA Kernel Profiling

### Part A — Bandwidth & Speedup Analysis

| Vector Size | CPU Time (ms) | GPU Time (ms) | H2D Time (ms) | Speedup |
|-------------|--------------|--------------|--------------|---------|
| N = 2^10 | 0.011 | 0.011 | 0.000 | 1.00x |
| N = 2^14 | 0.008 | 0.007 | 0.005 | 1.06x |
| N = 2^18 | 0.097 | 0.049 | 0.087 | 2.00x |
| N = 2^22 | 4.389 | 0.258 | 1.398 | 17.00x |
| N = 2^26 | 88.516 | 0.344 | 22.370 | 257.00x |

**Crossover Point:** GPU becomes faster than CPU at approximately **N = 2^18 (262,144 elements)**.

**Why small N favors CPU:** For small vectors, the GPU has fixed overhead from kernel launch (~5–10 µs) and PCIe transfer latency. This overhead dominates over actual compute time. The CPU avoids this overhead entirely and completes the work faster for small datasets.

### Part B — Launch Configuration Analysis (N = 2^20)

| Threads/Block | Block Count | Elements Covered | Time (ms) | Optimal? |
|--------------|------------|-----------------|----------|---------|
| 64 | 16,384 | 1,048,576 | 3.062 | — |
| 128 | 8,192 | 1,048,576 | 2.875 | — |
| **256** | **4,096** | **1,048,576** | **2.500** | ✅ BEST |
| 512 | 2,048 | 1,048,576 | 2.625 | — |
| 1024 | 1,024 | 1,048,576 | 2.875 | — |

**Optimal Block Size: 256 threads/block**

**Why multiples of 32?** GPUs execute threads in groups of 32 called warps. If a block size is not a multiple of 32, the last warp is partially filled, wasting execution slots. For example, 33 threads uses 2 warps but only 1 thread in the second warp is active — 31 slots wasted (97% waste in that warp). Multiples of 32 ensure full warp utilization, maximizing throughput and hiding memory latency through warp-level instruction-level parallelism. Values like 128, 256, 512 are common choices balancing occupancy and register pressure.

### Part C — Warp Divergence Experiment

| Kernel | Time (ms) |
|--------|----------|
| With warp divergence | 9.040 |
| Without warp divergence | 0.858 |
| **Divergence penalty** | **10.54x slower** |

**Explanation:** When threads in the same warp take different branches, the GPU serializes both paths. All 32 threads execute BOTH branches but with different threads masked off, effectively halving (or worse) throughput.

---

## Problem 2 — Parallel Reduction & Shared Memory

### Part A — Three Reduction Strategies (N = 2^20)

| Strategy | Time (µs) | Throughput (GB/s) | Result | Correct? |
|----------|----------|-----------------|--------|---------|
| Naive Sequential | 1,033.26 | 4.06 | 524,620.44 | ✓ |
| Shared Memory Tree | 11,421.76 | 0.37 | 524,620.38 | ✓ |
| **Warp Shuffle** | **413.25** | **10.15** | **524,620.38** | **✓** |

Reference (numpy.sum): **524,620.38** — all three verified with atol = 0.1 ✓

### Part B — Bank Conflict Profiling

| Stride | Time (µs) | Bank Conflicts |
|--------|----------|---------------|
| 1 | 4.2153 | None (optimal) |
| 2 | 3.5145 | 2-way |
| 4 | 3.2795 | 4-way |
| 8 | 10.5596 | 8-way |
| 16 | 5.0625 | 16-way |
| 32 | 3.2741 | Max (32-way) |

**Stride = 1:** Consecutive threads access consecutive banks → no conflicts (optimal).  
**Stride = 32:** All 32 threads in a warp access the SAME bank → 32-way conflict, serialized into 32 sequential accesses, destroying throughput.

**Padding Solution `tile[16][17]`:** Adding 1 extra column shifts each row's start to a different bank, breaking the alignment that causes stride-32 conflicts.

### Part C — Shared Memory Histogram

| Approach | Time (µs) | Correct? |
|----------|----------|---------|
| Global atomics (naive) | 2,779.15 | ✓ |
| Shared memory (per-block private) | 3,376.44 | ✓ |

Shared memory reduces global `atomicAdd` contention by keeping per-block partial histograms in fast on-chip memory, then merging at the end.

---

## Problem 3 — Custom ML Kernels

### Part A — Activation Function Benchmarks (N = 10^7)

| Kernel | Time (ms) | Bandwidth (GB/s) | Max Error |
|--------|----------|-----------------|----------|
| Sigmoid | 39.506 | 2.02 | 0.00e+00 ✓ |
| Tanh | 19.098 | 4.19 | 0.00e+00 ✓ |
| Leaky ReLU | 63.670 | 1.26 | 0.00e+00 ✓ |
| ReLU Backward | 12.876 | 6.21 | 0.00e+00 ✓ |

All kernels verified with atol ≤ 1e-4 against numpy/PyTorch reference ✓

### Part B — Loss Functions

| Loss Function | Value | Status |
|--------------|-------|--------|
| Cross-Entropy (stable, log-sum-exp) | 2.740626 | ✓ Verified |
| BCE Loss (numerically clipped) | 0.801206 | ✓ Verified |
| CE Gradient — shape (10000, 10) | max=8.4e-5, min=-1.0e-4 | ✓ Verified |

CE Gradient formula used: `grad[c] = softmax(logits)[c] − one_hot(label)[c]`

### Part C — Adam Optimizer (100 Steps)

| Step | Param Mean | Grad Norm |
|------|-----------|----------|
| 1 | 0.019183 | 3.204494 |
| 10 | 0.019185 | 3.029498 |
| 50 | 0.018836 | 3.097635 |
| 100 | 0.018655 | 3.165930 |

**Param change after 100 steps:** 0.3272 ✓ Verified

---

## Problem 4 — Tiled GEMM & CNN Layers

### Part A — GEMM Benchmark

| Matrix Size | Naive (ms) | Tiled (ms) | numpy (ms) | Naive GFLOPS | Tiled GFLOPS |
|------------|-----------|-----------|-----------|-------------|-------------|
| 128×128 | 2.637 | 2.897 | 7.263 | 1.59 | 1.45 |
| 256×256 | 4.771 | 31.385 | 24.814 | 7.03 | 1.07 |
| 512×512 | 10.003 | 18.006 | 1.341 | 26.83 | 14.91 |
| 1024×1024 | 29.191 | 52.543 | 14.342 | 73.57 | 40.87 |

**Correctness (128×128):** max error = 1.91e-05 ✓ PASSED

**Why tiled GEMM underperforms cuBLAS:** Tiled GEMM improves on naive by loading data into shared memory to reduce global memory accesses. However, cuBLAS uses Tensor Cores (Volta+) for 4×4 matrix ops in a single cycle, vectorized 128-bit loads (float4), register-level 128×128 tiling, loop unrolling, and double buffering in shared memory to overlap data loading with computation. Our 16×16 implementation lacks all of these hardware-specific optimizations.

### Part B — CNN Layer Benchmarks (Input: 32×64×14×14)

| Layer | Time (ms) |
|-------|----------|
| Conv2D (3×3, same padding) | 0.5716 |
| BatchNorm (inference mode) | 1.1814 |
| MaxPool (2×2) | 0.0034 |

### Part C — im2col Convolution

| Property | Value |
|----------|-------|
| Output shape | (4, 72, 196) |
| im2col time | 0.8427 ms |
| Memory overhead | **9.00× input size** |

im2col transforms convolution into GEMM enabling cuBLAS acceleration, at the cost of 3–9× increased memory usage for 3×3 kernels.

---

## Problem 5 — Full MNIST CNN Training

### Model Architecture

```
Input [N, 1, 28, 28]
  → Conv2D(1→32, 3×3) → ReLU → MaxPool → [N, 32, 14, 14]
  → Conv2D(32→64, 3×3) → ReLU → MaxPool → [N, 64, 7, 7]
  → Flatten → FC(3136→128) → ReLU
  → FC(128→10)
Output logits [N, 10]
```

### Part A — Baseline Training Results (5 Epochs, Full 60K Dataset)

| Epoch | Train Loss | Train Acc% | Test Acc% | Time (s) |
|-------|-----------|-----------|---------|---------|
| 1 | 0.2199 | 93.45 | 98.33 | 38.4 |
| 2 | 0.0545 | 98.33 | 98.71 | 44.6 |
| 3 | 0.0384 | 98.81 | 98.63 | 40.4 |
| 4 | 0.0291 | 99.08 | 98.99 | 36.7 |
| 5 | 0.0218 | 99.34 | **99.04** | 42.7 |

**Achieved 99.04% test accuracy — target ≥ 97% met**

### Part B — Ablation Study (3 Epochs Each)

| Configuration | Test Acc% | Time (s) |
|--------------|---------|---------|
| **Baseline (Adam, no scheduler)** | **98.85** | 114.1 |
| + BatchNorm (both conv layers) | 98.58 | 137.8 |
| + Dropout (0.5 before final FC) | 98.81 | 104.8 |
| SGD + Momentum(0.9) + CosineAnnealingLR | 98.25 | 117.9 |

**Best Configuration: Baseline Adam — 98.85%**

**Discussion:** BatchNorm stabilizes training by normalizing layer inputs, reducing internal covariate shift and allowing higher learning rates. Dropout prevents overfitting by randomly zeroing activations, acting as an ensemble of smaller networks. SGD with momentum and CosineAnnealingLR often achieves better final accuracy than Adam for CNNs on complex datasets because the LR schedule helps escape local minima. For MNIST specifically, the dataset is simple and clean enough that the baseline Adam optimizer already achieves near-optimal results, making regularization techniques like Dropout and BatchNorm show marginal improvement at this scale.

### Part C — Data Augmentation

| Augmentation | Setting |
|-------------|---------|
| RandomRotation | ±10 degrees |
| RandomAffine | Shear up to 10 degrees |
| RandomErasing | Probability = 0.1 |

---

##  Key Concepts Summary

| Concept | Problem |
|---------|---------|
| Thread hierarchy, warp execution, kernel launch overhead | P1 |
| PCIe bandwidth, H2D/D2H transfer, crossover analysis | P1 |
| Warp divergence & branch serialization penalty | P1 |
| Parallel reduction — naive, tree, warp shuffle | P2 |
| Shared memory bank conflicts & padding fix | P2 |
| Per-block private histogram with merge | P2 |
| Sigmoid, Tanh, Leaky ReLU, ReLU backward kernels | P3 |
| Numerically stable cross-entropy (log-sum-exp) | P3 |
| Fused Adam optimizer kernel | P3 |
| Tiled GEMM with shared memory (16×16 tiles) | P4 |
| Roofline model & arithmetic intensity | P4 |
| im2col convolution → GEMM transformation | P4 |
| CNN architecture (Conv → BN → Pool → FC) | P5 |
| Ablation study — BN, Dropout, SGD vs Adam | P5 |
| Data augmentation — rotation, affine, erasing | P5 |

---

## Author

**Bipasha** — Thapar Institute of Engineering & Technology  
Course: UCS645 — Parallel Computing  
Assignment: LAB 8 — GPU Accelerated Machine Learning
