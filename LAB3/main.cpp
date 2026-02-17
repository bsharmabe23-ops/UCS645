#include <iostream>
#include <cstdlib>
#include <chrono>

using namespace std;
using namespace std::chrono;

void correlate(int ny, int nx, const float* data, float* result);

int main(int argc, char* argv[])
{
    if (argc < 3) {
        cout << "Usage: ./correlate_program ny nx\n";
        return 1;
    }

    int ny = atoi(argv[1]);
    int nx = atoi(argv[2]);

    float* data = new float[ny * nx];
    float* result = new float[ny * ny];

    srand(0);

    for (int i = 0; i < ny * nx; i++) {
        data[i] = (float)rand() / RAND_MAX;
    }

    auto start = high_resolution_clock::now();

    correlate(ny, nx, data, result);

    auto end = high_resolution_clock::now();
    duration<double> elapsed = end - start;

    cout << "Execution time: " << elapsed.count() << " seconds\n";

    delete[] data;
    delete[] result;

    return 0;
}
