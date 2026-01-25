#include <stdio.h>
#include <omp.h>

int main() {
    long num_steps = 100000000;
    double step = 1.0 / num_steps;
    double sum = 0.0;
    int i;

    double start = omp_get_wtime();

    #pragma omp parallel for reduction(+:sum)
    for (i = 0; i < num_steps; i++) {
        double x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }

    double pi = step * sum;
    double end = omp_get_wtime();

    printf("Calculated PI = %f\n", pi);
    printf("Time taken: %f seconds\n", end - start);

    return 0;
}
