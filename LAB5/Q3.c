#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1000000

int main(int argc, char** argv) {
    int rank, size;
    double local_sum = 0, total_sum;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int chunk = N / size;

    double start = MPI_Wtime();

    for(int i=0; i<chunk; i++) {
        local_sum += 1.0 * 2.0; // A[i]=1, B[i]=2
    }

    MPI_Reduce(&local_sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double end = MPI_Wtime();

    if(rank==0) {
        printf("Dot Product = %f\n", total_sum);
        printf("Time = %f\n", end-start);
    }

    MPI_Finalize();
}