#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024

__global__ void sumKernel(int *input, long long *output) {
    int tid = threadIdx.x;
    if (tid == 0) {
        long long sum = 0;
        for (int i = 0; i < N; i++) sum += input[i];
        output[0] = sum;
    }
    else if (tid == 1) {
        output[1] = (long long)N * (N + 1) / 2;
    }
}

int main() {
    printf("=== Problem 1: Sum of First %d Integers ===\n\n", N);

    int h_input[N];
    for (int i = 0; i < N; i++) h_input[i] = i + 1;

    // Check GPU
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount > 0) {
        // ---- GPU PATH ----
        int *d_input;
        long long *d_output;
        long long h_output[2] = {0, 0};

        cudaMalloc(&d_input,  N * sizeof(int));
        cudaMalloc(&d_output, 2 * sizeof(long long));
        cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemset(d_output, 0, 2 * sizeof(long long));

        sumKernel<<<1, 2>>>(d_input, d_output);
        cudaDeviceSynchronize();

        cudaMemcpy(h_output, d_output, 2 * sizeof(long long), cudaMemcpyDeviceToHost);

        printf("Running on GPU:\n");
        printf("Iterative Sum (Thread 0): %lld\n", h_output[0]);
        printf("Formula Sum   (Thread 1): %lld\n", h_output[1]);

        cudaFree(d_input);
        cudaFree(d_output);
    } else {
        // ---- CPU FALLBACK ----
        printf("Note: No GPU found, running equivalent CPU code.\n");
        printf("(CUDA kernel logic is identical — same algorithm)\n\n");

        // Simulating Thread 0 — iterative
        long long sum_iter = 0;
        for (int i = 0; i < N; i++) sum_iter += h_input[i];

        // Simulating Thread 1 — formula
        long long sum_formula = (long long)N * (N + 1) / 2;

        printf("Iterative Sum (Thread 0): %lld\n", sum_iter);
        printf("Formula Sum   (Thread 1): %lld\n", sum_formula);
        printf("\nBoth results match: %s\n",
               sum_iter == sum_formula ? "YES (correct)" : "NO (error)");
    }

    return 0;
}
