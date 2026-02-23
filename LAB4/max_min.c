#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    srand(time(NULL) + rank);

    int nums[10];
    int local_max = 0;
    int local_min = 1000;

    for(int i = 0; i < 10; i++) {
        nums[i] = rand() % 1000;

        if(nums[i] > local_max)
            local_max = nums[i];

        if(nums[i] < local_min)
            local_min = nums[i];
    }

    struct {
        int value;
        int rank;
    } in, out;

    in.value = local_max;
    in.rank = rank;

    MPI_Reduce(&in, &out, 1, MPI_2INT,
               MPI_MAXLOC, 0, MPI_COMM_WORLD);

    if(rank == 0)
        printf("Global Maximum: %d (Process %d)\n",
               out.value, out.rank);

    in.value = local_min;
    in.rank = rank;

    MPI_Reduce(&in, &out, 1, MPI_2INT,
               MPI_MINLOC, 0, MPI_COMM_WORLD);

    if(rank == 0)
        printf("Global Minimum: %d (Process %d)\n",
               out.value, out.rank);

    MPI_Finalize();
    return 0;
}