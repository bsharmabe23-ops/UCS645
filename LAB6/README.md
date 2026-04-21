# 🖥️ UCS645 — LAB 6: Introduction to CUDA

> CUDA Programming Assignment covering Device Query, Array Sum, and Matrix Addition using parallel GPU computing.

---

## 📁 File Structure

| File | Description |
|------|-------------|
| `q1.cu` | Part A — Device Query (GPU properties) |
| `q2.cu` | Part B — Array Sum using CUDA kernel |
| `q3.cu` | Part C — Matrix Addition using CUDA kernel |
| `README.md` | Assignment documentation |

---

## ⚙️ Environment

| Property | Details |
|----------|---------|
| OS | Ubuntu 24.04 (WSL2) |
| CUDA Version | 12.0, V12.0.140 |
| Compiler | NVCC (NVIDIA CUDA Compiler) |
| GPU Reference | NVIDIA GeForce GTX 1650 |
| Compute Capability | 7.5 (Turing Architecture) |

---

## Part A — Device Query (`q1.cu`)

Queries GPU hardware properties using `cudaGetDeviceProperties()`.

### How to Compile & Run
```bash
nvcc q1.cu -o q1
./q1
```

### Output
```
=== GPU 0 (Reference: NVIDIA GeForce GTX 1650) ===
Compute capability:       7.5
Total global memory:      4.00 GB
Shared mem per block:     49152 bytes
Constant memory:          65536 bytes
Warp size:                32
Max threads per block:    1024
Max block dim:            [1024, 1024, 64]
Max grid dim:             [2147483647, 65535, 65535]
Multiprocessors:          16
Double precision support: Yes
```

### Report Answers

| Question | Answer |
|----------|--------|
| Q1. Architecture & Compute Capability | Turing Architecture, Compute Capability **7.5** |
| Q2. Maximum Block Dimensions | **[1024, 1024, 64]** |
| Q3. Max threads (grid=65535, block=512) | 65535 × 512 = **33,553,920 threads** |
| Q4. Why not always launch max threads? | If data is smaller than max, extra threads are idle and waste registers/shared memory |
| Q5. What limits max thread launch? | Global memory, registers per thread, shared memory per block, SM occupancy |
| Q6. Shared Memory | Fast on-chip memory per block — **49152 bytes (48 KB)** |
| Q7. Global Memory | Main GPU DRAM accessible by all threads — **4.00 GB** |
| Q8. Constant Memory | Read-only cached memory for constants — **65536 bytes (64 KB)** |
| Q9. Warp Size | 32 threads execute simultaneously as one unit — Warp size = **32** |
| Q10. Double Precision Support | **Yes** (Compute Capability 7.5 ≥ 2.0) |

---

## Part B — Array Sum (`q2.cu`)

Computes the sum of 1024 single-precision floating point numbers using **parallel reduction** with shared memory.

### How to Compile & Run
```bash
nvcc q2.cu -o q2
./q2
```

### Output
```
Sum = 1024.00 (expected 1024.00)
```

### CUDA Execution Steps

| Step | Description |
|------|-------------|
| 1 | Allocate device memory using `cudaMalloc` |
| 2 | Copy host array to device using `cudaMemcpy` |
| 3 | Set block size = 256, grid size = 4 |
| 4 | Launch `sumKernel` with shared memory parallel reduction |
| 5 | Copy result back to host using `cudaMemcpy` |
| 6 | Free device memory using `cudaFree` |

### Kernel Design

- Each thread loads one element into **shared memory**
- Parallel reduction halves active threads each step using a stride loop
- Thread 0 of each block writes the block's partial sum to global memory
- Final sum is computed on the host across all block results

---

## Part C — Matrix Addition (`q3.cu`)

Performs addition of two **1024 × 1024** integer matrices using a 2D CUDA kernel.

### How to Compile & Run
```bash
nvcc q3.cu -o q3
./q3
```

### Output
```
A[0][0]=0,    B[0][0]=0,    C[0][0]=0
A[1][0]=1024, B[1][0]=2048, C[1][0]=3072
Matrix addition completed for 1024x1024 matrices.
```

### Grid & Block Configuration

| Parameter | Value |
|-----------|-------|
| Block size | 16 × 16 = 256 threads |
| Grid size | 64 × 64 = 4096 blocks |
| Total threads | 1,048,576 |

### Report Answers

| Question | Answer |
|----------|--------|
| Q1. Floating point operations | 1 addition per element → **1,048,576 operations** |
| Q2. Global memory reads | 2 reads per thread (A and B) → **2,097,152 reads** |
| Q3. Global memory writes | 1 write per thread (C) → **1,048,576 writes** |

---

## 🔑 Key CUDA Concepts Used

| Concept | Used In |
|---------|---------|
| `cudaMalloc` / `cudaFree` | Part B, Part C |
| `cudaMemcpy` (Host ↔ Device) | Part B, Part C |
| `__shared__` memory | Part B |
| `__syncthreads()` | Part B |
| Parallel reduction | Part B |
| 2D grid and block dimensions (`dim3`) | Part C |
| `cudaGetDeviceProperties` | Part A |

---

## 👩‍💻 Author

**Bipasha** — Thapar Institute of Engineering & Technology  
Course: UCS645 — Parallel Computing  
