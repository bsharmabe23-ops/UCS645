#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1000000

int main(int argc, char** argv) {
    int rank, size;
    double *data;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    data = (double*)malloc(N * sizeof(double));

    double start, end;

    // Part A: Manual Send
    if(rank == 0) {
        start = MPI_Wtime();
        for(int i=1; i<size; i++) {
            MPI_Send(data, N, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(data, N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    end = MPI_Wtime();
    if(rank==0) printf("Manual Broadcast Time = %f\n", end-start);

    // Part B: MPI_Bcast
    start = MPI_Wtime();
    MPI_Bcast(data, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    end = MPI_Wtime();

    if(rank==0) printf("MPI_Bcast Time = %f\n", end-start);

    MPI_Finalize();
    return 0;
}