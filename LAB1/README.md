# UCS645 – LAB 1 (OpenMP)

## Aim
To understand and implement parallel programming using OpenMP in C.

## Programs Implemented
1. hello_omp.c – Basic OpenMP hello world program
2. q1_daxpy.c – DAXPY vector operation using OpenMP
3. q2_matrix_1d.c – Matrix multiplication using 1D OpenMP parallelism
4. q2_matrix_2d.c – Matrix multiplication using 2D OpenMP parallelism
5. q3_pi.c – Parallel calculation of PI using reduction

## Observations
- Execution time decreases as the number of threads increases.
- 2D matrix parallelization performs better than 1D due to improved load distribution.
- Reduction clause avoids race conditions in PI calculation.
- Maximum speedup is achieved near the number of CPU cores.

## Platform
- Ubuntu (WSL)
- GCC with OpenMP
- Visual Studio Code
