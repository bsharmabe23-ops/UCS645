#include <mpi.h>
#include <stdio.h>
#include <math.h>

int isPrime(int n) {
    if(n < 2) return 0;
    for(int i=2; i<=sqrt(n); i++)
        if(n % i == 0) return 0;
    return 1;
}

int main(int argc, char** argv) {
    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int num = 10;

    if(rank == 0) {
        for(int i=2; i<=num; i++) {
            printf("%d is %s\n", i, isPrime(i) ? "Prime" : "Not Prime");
        }
    }

    MPI_Finalize();
}