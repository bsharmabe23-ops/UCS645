// Q1: Molecular Dynamics – Lennard-Jones Potential
#include <stdio.h>
#include <math.h>
#include <omp.h>

#define N 2000
#define EPS 1.0
#define SIGMA 1.0

int main() {
    double x[N], y[N], z[N];
    double potential = 0.0;

    for (int i = 0; i < N; i++) {
        x[i] = y[i] = z[i] = i * 0.01;
    }

    double start = omp_get_wtime();

    #pragma omp parallel for reduction(+:potential) schedule(dynamic)
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            double dx = x[i] - x[j];
            double dy = y[i] - y[j];
            double dz = z[i] - z[j];
            double r2 = dx*dx + dy*dy + dz*dz;
            double inv_r6 = pow(SIGMA*SIGMA / r2, 3);
            potential += 4 * EPS * inv_r6 * (inv_r6 - 1);
        }
    }

    double end = omp_get_wtime();

    printf("Q1 Lennard-Jones\n");
    printf("Potential = %f\n", potential);
    printf("Time = %f seconds\n", end - start);
    return 0;
}
