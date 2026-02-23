#include <mpi.h>
#include <stdio.h>

#define N 100

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int arr[N];
    int local_size = N / size;
    int local_arr[local_size];

    if(rank == 0) {
        for(int i = 0; i < N; i++)
            arr[i] = i + 1;
    }

    MPI_Scatter(arr, local_size, MPI_INT,
                local_arr, local_size, MPI_INT,
                0, MPI_COMM_WORLD);

    int local_sum = 0;
    for(int i = 0; i < local_size; i++)
        local_sum += local_arr[i];

    int global_sum;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT,
               MPI_SUM, 0, MPI_COMM_WORLD);

    if(rank == 0) {
        printf("Global Sum = %d\n", global_sum);
        printf("Average = %.2f\n", global_sum / (float)N);
    }

    MPI_Finalize();
    return 0;
}