#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>

#define N 1024

// 1.1: Statically defined global device arrays
__device__ float d_A[N];
__device__ float d_B[N];
__device__ float d_C[N];

__global__ void vectorAdd() {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        d_C[i] = d_A[i] + d_B[i];
}

int main() {
    printf("=== Problem 3: Vector Addition (N=%d) ===\n\n", N);

    float h_A[N], h_B[N], h_C[N];
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)i;
        h_B[i] = (float)(i * 2);
    }

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    int threadsPerBlock = 256;
    int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;

    float ms = 0.0f;

    if (deviceCount > 0) {
        // ---- REAL GPU PATH ----
        // 1.1: Copy using cudaMemcpyToSymbol
        cudaMemcpyToSymbol(d_A, h_A, N * sizeof(float));
        cudaMemcpyToSymbol(d_B, h_B, N * sizeof(float));

        // 1.2: Time the kernel
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&ms, start, stop);
        cudaMemcpyFromSymbol(h_C, d_C, N * sizeof(float));

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        printf("1.1 Static device arrays used (cudaMemcpyToSymbol)\n");
        printf("1.2 Kernel execution time: %.6f ms\n\n", ms);

    } else {
        // ---- CPU FALLBACK ----
        printf("Note: No GPU detected. Running CPU equivalent.\n");
        printf("1.1 Static device arrays defined (compiled successfully)\n");

        struct timespec t1, t2;
        clock_gettime(CLOCK_MONOTONIC, &t1);
        for (int i = 0; i < N; i++) h_C[i] = h_A[i] + h_B[i];
        clock_gettime(CLOCK_MONOTONIC, &t2);

        ms = (t2.tv_sec - t1.tv_sec) * 1000.0f +
             (t2.tv_nsec - t1.tv_nsec) / 1e6f;

        printf("1.2 Kernel execution time (CPU sim): %.6f ms\n\n", ms);
    }

    // Verify
    printf("Sample results:\n");
    printf("  A[1]=%.0f  + B[1]=%.0f  = C[1]=%.0f\n",  h_A[1],  h_B[1],  h_C[1]);
    printf("  A[10]=%.0f + B[10]=%.0f = C[10]=%.0f\n", h_A[10], h_B[10], h_C[10]);
    printf("  A[99]=%.0f + B[99]=%.0f = C[99]=%.0f\n\n",h_A[99], h_B[99], h_C[99]);

    // 1.3: Theoretical Bandwidth (works even without GPU — returns 0 if no device)
    cudaDeviceProp prop;
    double theoreticalBW = 0;

    if (deviceCount > 0) {
        cudaGetDeviceProperties(&prop, 0);
        theoreticalBW = 2.0
                      * prop.memoryClockRate
                      * (prop.memoryBusWidth / 8.0)
                      / 1e6;
        printf("1.3 Device Name   : %s\n", prop.name);
        printf("    Mem Clock     : %d kHz\n",  prop.memoryClockRate);
        printf("    Bus Width     : %d bits\n", prop.memoryBusWidth);
    } else {
        // Typical values for a mid-range NVIDIA GPU (for demonstration)
        int memClockRate  = 7000000; // kHz (7 GHz typical)
        int memBusWidth   = 128;     // bits (typical for laptop GPU)
        theoreticalBW = 2.0 * memClockRate * (memBusWidth / 8.0) / 1e6;
        printf("1.3 No GPU found — using typical demo values:\n");
        printf("    Mem Clock     : %d kHz (example)\n", memClockRate);
        printf("    Bus Width     : %d bits (example)\n", memBusWidth);
    }

    printf("    Theoretical BW: %.2f GB/s\n\n", theoreticalBW);

    // 1.4: Measured Bandwidth
    long long rBytes  = 2LL * N * sizeof(float);
    long long wBytes  = 1LL * N * sizeof(float);
    double t_seconds  = ms / 1000.0;
    double measuredBW = (t_seconds > 0)
                      ? (rBytes + wBytes) / t_seconds / 1e9
                      : 0.0;

    printf("1.4 Bytes Read    : %lld bytes\n", rBytes);
    printf("    Bytes Written : %lld bytes\n", wBytes);
    printf("    Kernel Time   : %.6f ms\n",    ms);
    printf("    Measured BW   : %.4f GB/s\n\n", measuredBW);

    printf("--- Comparison ---\n");
    printf("Theoretical BW : %.2f GB/s\n", theoreticalBW);
    printf("Measured BW    : %.4f GB/s\n", measuredBW);
    if (theoreticalBW > 0)
        printf("Efficiency     : %.4f%%\n", (measuredBW/theoreticalBW)*100.0);
    printf("\nMeasured < Theoretical because of memory latency,\n");
    printf("cache effects, and small data size overhead.\n");
    printf("=========================================\n");

    return 0;
}
