// Q3: Scientific Computing – Heat Diffusion Simulation
#include <stdio.h>
#include <omp.h>

#define N 200
#define STEPS 500

int main() {
    double grid[N][N] = {0};
    double next_grid[N][N] = {0};

    // Initial heat source
    grid[N/2][N/2] = 100.0;

    double start = omp_get_wtime();

    for (int t = 0; t < STEPS; t++) {

        #pragma omp parallel for
        for (int i = 1; i < N-1; i++) {
            for (int j = 1; j < N-1; j++) {
                next_grid[i][j] = 0.25 * (
                    grid[i-1][j] + grid[i+1][j] +
                    grid[i][j-1] + grid[i][j+1]
                );
            }
        }

        #pragma omp parallel for
        for (int i = 1; i < N-1; i++)
            for (int j = 1; j < N-1; j++)
                grid[i][j] = next_grid[i][j];
    }

    double end = omp_get_wtime();

    printf("Q3 Heat Diffusion Simulation\n");
    printf("Time = %f seconds\n", end - start);

    return 0;
}
