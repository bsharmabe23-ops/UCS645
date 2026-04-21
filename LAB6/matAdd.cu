#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define ROWS 1024
#define COLS 1024

__global__ void matAddKernel(int *A, int *B, int *C, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        C[row * cols + col] = A[row * cols + col] + B[row * cols + col];
    }
}

int main() {
    int size = ROWS * COLS * sizeof(int);
    int *h_A, *h_B, *h_C;
    int *d_A, *d_B, *d_C;

    h_A = (int*)malloc(size);
    h_B = (int*)malloc(size);
    h_C = (int*)malloc(size);

    for (int i = 0; i < ROWS * COLS; i++) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((COLS + 15) / 16, (ROWS + 15) / 16);

    matAddKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, ROWS, COLS);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("A[0][0]=%d, B[0][0]=%d, C[0][0]=%d\n", h_A[0], h_B[0], h_C[0]);
    printf("A[1][0]=%d, B[1][0]=%d, C[1][0]=%d\n", h_A[1024], h_B[1024], h_C[1024]);
    printf("Matrix addition completed for %dx%d matrices.\n", ROWS, COLS);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}
