# UCS645 – LAB1 (OpenMP Experiments)

Student: Bipasha Sharma  
Roll NO: 102303388

---

## EXPERIMENTAL QUESTIONS & RESULTS

---

## Q1. DAXPY Loop

**Operation:**  
X[i] = a × X[i] + Y[i], where X and Y are vectors of size 2^16.

### Output

OMP_NUM_THREADS = 1  
Time taken: 0.000262 seconds  

OMP_NUM_THREADS = 2  
Time taken: 0.000298 seconds  

OMP_NUM_THREADS = 4  
Time taken: 0.003944 seconds  

OMP_NUM_THREADS = 8  
Time taken: 0.000964 seconds  

### Observation

Execution time initially remains low for smaller thread counts.
As the number of threads increases beyond the available CPU cores,
performance fluctuates due to thread management overhead and context switching.
Maximum efficiency is observed for lower to moderate thread counts.

---

## Q2. Matrix Multiplication (1D vs 2D Threading)

**Matrix Size:** 500 × 500  

### Output – 1D Threading

OMP_NUM_THREADS = 1  
Time taken (1D): 0.715027 seconds  

OMP_NUM_THREADS = 2  
Time taken (1D): 0.708881 seconds  

OMP_NUM_THREADS = 4  
Time taken (1D): 0.294908 seconds  

OMP_NUM_THREADS = 8  
Time taken (1D): 0.358535 seconds  

---

### Output – 2D Threading

OMP_NUM_THREADS = 1  
Time taken (2D): 0.505215 seconds  

OMP_NUM_THREADS = 2  
Time taken (2D): 0.410227 seconds  

OMP_NUM_THREADS = 4  
Time taken (2D): 0.302412 seconds  

OMP_NUM_THREADS = 8  
Time taken (2D): 0.370897 seconds  

### Observation

2D threading generally provides better load balancing compared to 1D threading.
Performance improves as the number of threads increases up to an optimal point.
Beyond this, overhead due to increased synchronization reduces the benefit
of additional threads.

---

## Q3. Calculation of π Using Numerical Integration

### Output

OMP_NUM_THREADS = 1  
Calculated PI = 3.141593  
Time taken: 0.381978 seconds  

OMP_NUM_THREADS = 2  
Calculated PI = 3.141593  
Time taken: 0.197506 seconds  

OMP_NUM_THREADS = 4  
Calculated PI = 3.141593  
Time taken: 0.146089 seconds  

OMP_NUM_THREADS = 8  
Calculated PI = 3.141593  
Time taken: 0.156900 seconds  

### Observation

The calculated value of π remains accurate across all thread counts.
Execution time decreases as threads increase up to an optimal limit.
Beyond this point, additional threads introduce overhead, slightly
increasing execution time.

---

## Conclusion

OpenMP significantly improves performance for computationally intensive tasks
by utilizing parallel execution. However, performance gain is limited by the
number of physical CPU cores. Increasing the number of threads beyond this
limit results in overhead and reduced efficiency.

**End of LAB1 – Experimental Section**
