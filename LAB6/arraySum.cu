#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024

__global__ void sumKernel(float *d_in, float *d_out, int n) {
    __shared__ float sharedData[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sharedData[tid] = (idx < n) ? d_in[idx] : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            sharedData[tid] += sharedData[tid + stride];
        __syncthreads();
    }

    if (tid == 0)
        d_out[blockIdx.x] = sharedData[0];
}

int main() {
    int size = N * sizeof(float);
    float h_in[N], h_out[4];
    float *d_in, *d_out;

    for (int i = 0; i < N; i++) h_in[i] = 1.0f;

    cudaMalloc(&d_in,  size);
    cudaMalloc(&d_out, 4 * sizeof(float));

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize  = (N + blockSize - 1) / blockSize;

    sumKernel<<<gridSize, blockSize>>>(d_in, d_out, N);

    cudaMemcpy(h_out, d_out, gridSize * sizeof(float), cudaMemcpyDeviceToHost);

    float total = 0;
    for (int i = 0; i < gridSize; i++) total += h_out[i];
    printf("Sum = %.2f (expected %.2f)\n", total, (float)N);

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
