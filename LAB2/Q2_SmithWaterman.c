// Q2: Bioinformatics – Smith-Waterman DNA Sequence Alignment
#include <stdio.h>
#include <string.h>
#include <omp.h>

#define MAX 100

int max3(int a, int b, int c) {
    if (a >= b && a >= c) return a;
    if (b >= a && b >= c) return b;
    return c;
}

int main() {
    char X[] = "ACACACTA";
    char Y[] = "AGCACACA";

    int H[MAX][MAX] = {0};
    int m = strlen(X);
    int n = strlen(Y);
    int max_score = 0;

    double start = omp_get_wtime();

    // Wavefront (diagonal) parallelization
    for (int d = 1; d <= m + n - 1; d++) {
        #pragma omp parallel for
        for (int i = 1; i <= m; i++) {
            int j = d - i;
            if (j >= 1 && j <= n) {
                int match = (X[i-1] == Y[j-1]) ? 2 : -1;
                H[i][j] = max3(
                    0,
                    H[i-1][j-1] + match,
                    H[i-1][j] - 1
                );
                if (H[i][j] > max_score)
                    max_score = H[i][j];
            }
        }
    }

    double end = omp_get_wtime();

    printf("Q2 Smith-Waterman Alignment\n");
    printf("Maximum Alignment Score = %d\n", max_score);
    printf("Time = %f seconds\n", end - start);

    return 0;
}
