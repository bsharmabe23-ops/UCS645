#include <stdio.h>
#include <omp.h>

#define N 65536

int main() {
    double X[N], Y[N];
    double a = 2.0;
    int i;

    for (i = 0; i < N; i++) {
        X[i] = 1.0;
        Y[i] = 2.0;
    }

    double start = omp_get_wtime();

    #pragma omp parallel for
    for (i = 0; i < N; i++) {
        X[i] = a * X[i] + Y[i];
    }

    double end = omp_get_wtime();

    printf("Time taken: %f seconds\n", end - start);
    return 0;
}
