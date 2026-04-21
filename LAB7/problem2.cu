#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define N 1000

// ---- CPU Merge Sort ----
void merge_cpu(int *arr, int l, int m, int r) {
    int n1 = m - l + 1, n2 = r - m;
    int *L = (int*)malloc(n1 * sizeof(int));
    int *R = (int*)malloc(n2 * sizeof(int));
    for (int i = 0; i < n1; i++) L[i] = arr[l + i];
    for (int i = 0; i < n2; i++) R[i] = arr[m + 1 + i];
    int i = 0, j = 0, k = l;
    while (i < n1 && j < n2)
        arr[k++] = (L[i] <= R[j]) ? L[i++] : R[j++];
    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];
    free(L); free(R);
}

void mergeSort_cpu(int *arr, int l, int r) {
    if (l < r) {
        int m = (l + r) / 2;
        mergeSort_cpu(arr, l, m);
        mergeSort_cpu(arr, m + 1, r);
        merge_cpu(arr, l, m, r);
    }
}

// ---- CUDA Kernel (compiled but runs on GPU if available) ----
__global__ void mergeKernel(int *arr, int *temp, int width, int n) {
    int tid   = blockIdx.x * blockDim.x + threadIdx.x;
    int left  = tid * 2 * width;
    if (left >= n) return;
    int mid   = min(left + width, n);
    int right = min(left + 2 * width, n);
    int i = left, j = mid, k = left;
    while (i < mid && j < right)
        temp[k++] = (arr[i] <= arr[j]) ? arr[i++] : arr[j++];
    while (i < mid)   temp[k++] = arr[i++];
    while (j < right) temp[k++] = arr[j++];
}

// ---- CPU simulation of parallel merge (iterative bottom-up) ----
void mergeSort_parallel_sim(int *arr, int n) {
    int *temp = (int*)malloc(n * sizeof(int));
    for (int width = 1; width < n; width *= 2) {
        for (int left = 0; left < n; left += 2 * width) {
            int mid   = left + width < n ? left + width : n;
            int right = left + 2*width < n ? left + 2*width : n;
            int i = left, j = mid, k = left;
            while (i < mid && j < right)
                temp[k++] = (arr[i] <= arr[j]) ? arr[i++] : arr[j++];
            while (i < mid)   temp[k++] = arr[i++];
            while (j < right) temp[k++] = arr[j++];
        }
        for (int i = 0; i < n; i++) arr[i] = temp[i];
    }
    free(temp);
}

int main() {
    printf("=== Problem 2: Merge Sort (N=%d) ===\n\n", N);

    int arr_a[N], arr_b[N];
    srand(42);
    for (int i = 0; i < N; i++)
        arr_a[i] = arr_b[i] = rand() % 10000;

    printf("Original first 5: %d %d %d %d %d\n",
           arr_a[0],arr_a[1],arr_a[2],arr_a[3],arr_a[4]);

    // ---- PART A: CPU Recursive Merge Sort (Pipelined) ----
    clock_t t1 = clock();
    mergeSort_cpu(arr_a, 0, N - 1);
    clock_t t2 = clock();
    double cpu_ms = 1000.0 * (t2 - t1) / CLOCKS_PER_SEC;

    printf("\nPart A - CPU Pipelined Merge Sort:\n");
    printf("Sorted first 5:  %d %d %d %d %d\n",
           arr_a[0],arr_a[1],arr_a[2],arr_a[3],arr_a[4]);
    printf("Time: %.4f ms\n", cpu_ms);

    // ---- PART B: CUDA or CPU simulation ----
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    float gpu_ms = 0;

    if (deviceCount > 0) {
        // Real CUDA
        int *d_arr, *d_temp;
        cudaMalloc(&d_arr,  N * sizeof(int));
        cudaMalloc(&d_temp, N * sizeof(int));

        // reset arr_b
        srand(42);
        for (int i = 0; i < N; i++) arr_b[i] = rand() % 10000;
        cudaMemcpy(d_arr, arr_b, N * sizeof(int), cudaMemcpyHostToDevice);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        for (int width = 1; width < N; width *= 2) {
            int numMerges = (N + 2*width - 1) / (2*width);
            int threads = 256;
            int blocks  = (numMerges + threads - 1) / threads;
            mergeKernel<<<blocks, threads>>>(d_arr, d_temp, width, N);
            cudaDeviceSynchronize();
            int *tmp = d_arr; d_arr = d_temp; d_temp = tmp;
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpu_ms, start, stop);
        cudaMemcpy(arr_b, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(d_arr); cudaFree(d_temp);

        printf("\nPart B - CUDA Parallel Merge Sort (GPU):\n");
    } else {
        // CPU simulation of CUDA bottom-up parallel merge
        srand(42);
        for (int i = 0; i < N; i++) arr_b[i] = rand() % 10000;

        clock_t t3 = clock();
        mergeSort_parallel_sim(arr_b, N);
        clock_t t4 = clock();
        gpu_ms = 1000.0 * (t4 - t3) / CLOCKS_PER_SEC;

        printf("\nPart B - CUDA Parallel Merge Sort (CPU simulation, no GPU):\n");
    }

    printf("Sorted first 5:  %d %d %d %d %d\n",
           arr_b[0],arr_b[1],arr_b[2],arr_b[3],arr_b[4]);
    printf("Time: %.4f ms\n", gpu_ms);

    // ---- PART C: Comparison ----
    printf("\n========= Part C: Performance Comparison =========\n");
    printf("Part A - CPU Recursive (Pipelined): %.4f ms\n", cpu_ms);
    printf("Part B - CUDA Parallel Merge Sort:  %.4f ms\n", gpu_ms);
    if (gpu_ms < cpu_ms)
        printf("Result: CUDA is faster by %.2fx\n", cpu_ms/gpu_ms);
    else
        printf("Result: CPU is faster (%.2fx)\n", gpu_ms/cpu_ms);
    printf("Note: For N=%d, CPU often wins due to GPU launch overhead.\n", N);
    printf("      CUDA shows advantage at N=100,000+\n");
    printf("==================================================\n");

    return 0;
}
