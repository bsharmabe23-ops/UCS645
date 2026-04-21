#  Assignment 7: CUDA Part II

**Course:** Parallel Computing  
**Student:** Bipasha Sharma  
**Institute:** Thapar Institute of Engineering & Technology  

---

##  Project Structure

CUDA_Assignment7/
├── problem1.cu     
├── problem2.cu     
├── problem3.cu     
└── README.md  

---

##  Environment

- **Compiler:** NVCC (CUDA Toolkit)  
- **Operating System:** Ubuntu 24.04 (WSL2)  
- **Language:** CUDA C++  

---

##  Problem 1: Sum of First N Integers (N = 1024)

###  Description
A single CUDA kernel uses two threads performing different tasks:

- **Thread 0:** Computes sum iteratively (1 → N)  
- **Thread 1:** Computes sum using formula → `N × (N + 1) / 2`  

###  Key Concepts
- `threadIdx.x` for task division  
- `cudaMalloc` for GPU memory allocation  
- `cudaMemcpy` for CPU ↔ GPU transfer  
- Branching within kernel based on thread ID  

###  Output
```
=== Problem 1: Sum of First 1024 Integers ===
Iterative Sum (Thread 0): 524800
Formula Sum   (Thread 1): 524800
Both results match: YES (correct)
```

---

##  Problem 2: Merge Sort — Pipelined vs CUDA Parallel (N = 1000)

###  Description

- **Part A:** CPU recursive merge sort (pipelined)  
- **Part B:** CUDA parallel merge sort (bottom-up approach)  
- **Part C:** Performance comparison  

###  Key Concepts
- Pipelining → recursive divide-and-conquer  
- CUDA → parallel merging across threads  
- GPU overhead affects small inputs  
- CUDA performs better for large datasets (100,000+)  

###  Output
```
=== Problem 2: Merge Sort (N=1000) ===
Original first 5: 6166 2740 8881 3241 1012

Part A - CPU Pipelined Merge Sort:
Sorted first 5:  18 26 46 51 58
Time: 0.1110 ms

Part B - CUDA Parallel Merge Sort:
Sorted first 5:  18 26 46 51 58
Time: 0.0810 ms

========= Part C: Performance Comparison =========
Part A - CPU Recursive (Pipelined): 0.1110 ms
Part B - CUDA Parallel Merge Sort:  0.0810 ms
Result: CUDA is faster by 1.37x

Note: For small N, CPU may outperform GPU due to launch overhead.
```

---

##  Problem 3: Vector Addition + Profiling (N = 1024)

###  Description

This problem includes multiple CUDA concepts:

1. Static device arrays (no `cudaMalloc`)  
2. Kernel timing using CUDA Events  
3. Theoretical memory bandwidth calculation  
4. Measured bandwidth analysis  

###  Key Concepts

- `__device__` arrays declared globally  
- `cudaMemcpyToSymbol` for data transfer  
- `cudaEvent_t` for precise kernel timing  

### 📐 Bandwidth Formulas

**Theoretical Bandwidth**
```
BW = 2 × memClockRate × (busWidth / 8) / 1e6   (GB/s)
```

**Measured Bandwidth**
```
BW = (Bytes Read + Bytes Written) / time / 1e9
```

###  Output
```
=== Problem 3: Vector Addition (N=1024) ===

1.1 Static device arrays used (cudaMemcpyToSymbol)
1.2 Kernel execution time: 0.000647 ms

Sample results:
  A[1]=1   + B[1]=2   = C[1]=3
  A[10]=10 + B[10]=20 = C[10]=30
  A[99]=99 + B[99]=198 = C[99]=297

1.3 Mem Clock     : 7000000 kHz
    Bus Width     : 128 bits
    Theoretical BW: 224.00 GB/s

1.4 Bytes Read    : 8192 bytes
    Bytes Written : 4096 bytes
    Kernel Time   : 0.000647 ms
    Measured BW   : 18.9923 GB/s

--- Comparison ---
Theoretical BW : 224.00 GB/s
Measured BW    : 18.9923 GB/s
Efficiency     : 8.48%
```

###  Note
Measured bandwidth is lower due to:
- Memory latency  
- Cache effects  
- Small input size overhead  

---

##  Compilation & Execution

```bash
nvcc problem1.cu -o problem1
nvcc problem2.cu -o problem2
nvcc -O2 problem3.cu -o problem3

./problem1
./problem2
./problem3
```

---

##  Conclusion

- CUDA enables massive parallelism but introduces overhead  
- CPU can outperform GPU for smaller inputs  
- GPU shows clear advantage for large-scale computations  
- Profiling helps understand real vs theoretical performance  

---

 This assignment demonstrates practical GPU programming concepts including parallel execution, memory management, and performance analysis using CUDA.
