Assignment 7: CUDA Part II
Course: Parallel Computing
Student: Bipasha
Institute: Thapar Institute of Engineering & Technology

Problem 1: Sum of First N Integers (N=1024)
What it does
Two threads perform different tasks inside the same CUDA kernel:

Thread 0 computes sum iteratively (loop from 1 to N)
Thread 1 computes sum using direct formula: N*(N+1)/2

Output
=== Problem 1: Sum of First 1024 Integers ===
Iterative Sum (Thread 0): 524800
Formula Sum   (Thread 1): 524800
Both results match: YES (correct)
Key Concepts

threadIdx.x used to assign different tasks to different threads
cudaMalloc allocates GPU memory
cudaMemcpy transfers data between CPU and GPU
Both threads run in same kernel but branch on their thread ID


Problem 2: Merge Sort — Pipelined vs CUDA Parallel (N=1000)
What it does

Part A: CPU recursive merge sort (pipelined)
Part B: CUDA parallel merge sort (bottom-up, each thread merges one segment)
Part C: Performance comparison of both

Output
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
Note: For N=1000, CPU often wins due to GPU launch overhead.
      CUDA shows advantage at N=100,000+
Key Concepts

Pipelining = recursive divide-and-conquer (sequential stages)
CUDA parallelism = all merge operations at same width run simultaneously
For small N, GPU launch overhead can negate parallelism benefit
CUDA wins significantly at large N (100,000+)


Problem 3: Vector Addition + Profiling (N=1024)
Sub-parts covered
Sub-partDescription1.1Static global device arrays — no cudaMalloc needed1.2Kernel timing using CUDA Events1.3Theoretical memory bandwidth from device properties1.4Measured memory bandwidth from kernel timing
Output
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
Efficiency     : 8.4787%
Key Concepts

1.1: Static arrays declared with __device__ float d_A[N] at compile time. Must use cudaMemcpyToSymbol to copy data — never pass as kernel argument directly as it causes invalid memory access
1.2: cudaEvent_t gives accurate GPU kernel timing in milliseconds
1.3: Formula: Theoretical BW = 2 x memClockRate(kHz) x (busWidth/8) / 1e6 in GB/s. Multiplied by 2 because DDR memory is double-pumped
1.4: Formula: Measured BW = (RBytes + WBytes) / time_seconds / 1e9. Kernel reads A and B, writes C = 3 arrays total
Why measured is much lower than theoretical: Real hardware has memory latency, cache misses, and small data overhead. Theoretical is the ideal maximum never fully achieved


File Structure
CUDA_Assignment7/
├── problem1.cu     
├── problem2.cu     
├── problem3.cu     
└── README.md
Environment

Compiler: NVCC (CUDA Toolkit)
OS: Ubuntu 24.04 (WSL2)
Language: CUDA C++

Compilation
bashnvcc problem1.cu -o problem1
nvcc problem2.cu -o problem2
nvcc -O2 problem3.cu -o problem3
