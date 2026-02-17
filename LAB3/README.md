# Parallel Correlation Matrix Computation

## 1. Introduction

This project implements a parallel correlation matrix computation using C++ and OpenMP.  
Given a matrix of size **ny × nx**, where each row represents an input vector, the objective is to compute the Pearson correlation coefficient between every pair of rows.

The computed values are stored in the lower triangular part of the result matrix.

---

## 2. Problem Statement

Given a matrix of size **ny × nx**:

- Each row represents one input vector.
- For all pairs satisfying:

  0 ≤ j ≤ i < ny

  Compute:

  correlation(row_i, row_j)

- Store the result at:

  result[i + j * ny]

All arithmetic operations are performed using double precision to ensure numerical accuracy.

---

## 3. Implementation Details

The implementation is written in C++ and compiled using GCC with OpenMP support.

### 3.1 Correlation Formula

The Pearson correlation coefficient between two vectors A and B is defined as:

\[
corr(A,B) = \frac{\sum (A_i - \mu_A)(B_i - \mu_B)}{\sqrt{\sum (A_i - \mu_A)^2} \cdot \sqrt{\sum (B_i - \mu_B)^2}}
\]

Where:
- \( \mu_A \), \( \mu_B \) are the means of the respective vectors.

### 3.2 Versions Implemented

| Version      | Description                                      |
|-------------|--------------------------------------------------|
| Sequential  | Single-threaded baseline implementation          |
| OpenMP      | Multi-threaded parallel implementation           |
| Optimized   | OpenMP with -O3 and architecture optimizations   |

Parallelization is achieved using:

```
#pragma omp parallel for schedule(static)
```

Compiler optimizations include:

```
-O3 -march=native -fopenmp
```

---

## 4. Compilation

To compile the program:

```bash
make
```

This generates the executable:

```
correlate_program
```

---

## 5. Execution

To run the program:

```bash
OMP_NUM_THREADS=<threads> ./correlate_program <ny> <nx>
```

Example:

```bash
OMP_NUM_THREADS=4 ./correlate_program 1000 1000
```

---

## 6. Performance Evaluation

Matrix size tested: **1000 × 1000**

### 6.1 Measured Execution Times

| Threads | Execution Time (seconds) |
|----------|--------------------------|
| 1        | 0.517286                 |
| 2        | 0.379678                 |
| 4        | 0.239682                 |
| 8        | 0.202947                 |

---

## 7. Speedup and Efficiency Analysis

Speedup is defined as:

\[
Speedup(p) = \frac{T(1)}{T(p)}
\]

Where:
- \( T(1) \) = Execution time with 1 thread
- \( T(p) \) = Execution time with p threads

Using baseline time:

\[
T(1) = 0.517286 \text{ seconds}
\]

### 7.1 Calculated Speedup and Efficiency

| Threads | Time (s)  | Speedup | Efficiency |
|----------|----------|----------|------------|
| 1        | 0.517286 | 1.00     | 100%       |
| 2        | 0.379678 | 1.36     | 68%        |
| 4        | 0.239682 | 2.16     | 54%        |
| 8        | 0.202947 | 2.55     | 32%        |

Efficiency is calculated as:

\[
Efficiency = \frac{Speedup}{Number\ of\ Threads}
\]

---

## 8. Discussion

The results demonstrate that:

- Execution time decreases as the number of threads increases.
- Speedup improves with parallel execution.
- Efficiency decreases at higher thread counts due to:
  - Memory bandwidth limitations
  - Thread scheduling overhead
  - Cache contention
  - Parallel overhead for moderate matrix sizes

The workload size (1000 × 1000) provides noticeable parallel benefit but does not achieve ideal linear scaling.

---

## 9. Conclusion

The OpenMP-based parallel implementation significantly improves performance compared to the sequential version. However, scalability is limited by hardware constraints and memory access patterns.

Further performance improvements could be achieved using:
- Cache blocking
- Better memory layout optimization
- Explicit SIMD vectorization
- Larger workload sizes

