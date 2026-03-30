#include <mpi.h>
#include <stdio.h>

int isPerfect(int n) {
    int sum = 0;

    for(int i = 1; i <= n/2; i++) {
        if(n % i == 0) {
            sum += i;
        }
    }

    return sum == n;
}

int main(int argc, char** argv) {
    int rank, size;
    int max = 1000;  // you can change limit

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Divide work among processes
    for(int i = 2 + rank; i <= max; i += size) {
        if(isPerfect(i)) {
            printf("Process %d: %d is Perfect\n", rank, i);
        }
    }

    MPI_Finalize();
    return 0;
}