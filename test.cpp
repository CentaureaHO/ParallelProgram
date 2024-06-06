#include <mpi.h>
#include <iostream>
#include <vector>
#include <chrono>

void matrix_multiply(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B,
    std::vector<std::vector<double>>& C, int start_row, int end_row)
{
    int N = B[0].size();
    for (int i = start_row; i < end_row; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            C[i][j] = 0;
            for (int k = 0; k < N; ++k) { C[i][j] += A[i][k] * B[k][j]; }
        }
    }
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    const int N = 1000;  // Size of the matrix (NxN)

    // Initialize matrices
    std::vector<std::vector<double>> A(N, std::vector<double>(N, 1.0));
    std::vector<std::vector<double>> B(N, std::vector<double>(N, 1.0));
    std::vector<std::vector<double>> C(N, std::vector<double>(N, 0.0));

    int rows_per_proc = N / world_size;
    int start_row     = world_rank * rows_per_proc;
    int end_row       = (world_rank == world_size - 1) ? N : start_row + rows_per_proc;

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Perform matrix multiplication
    matrix_multiply(A, B, C, start_row, end_row);

    // Gather results
    std::vector<double> local_result(rows_per_proc * N);
    for (int i = start_row; i < end_row; ++i)
    {
        for (int j = 0; j < N; ++j) { local_result[(i - start_row) * N + j] = C[i][j]; }
    }

    std::vector<double> global_result;
    if (world_rank == 0) { global_result.resize(N * N); }

    MPI_Gather(local_result.data(),
        rows_per_proc * N,
        MPI_DOUBLE,
        global_result.data(),
        rows_per_proc * N,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD);

    // End timing
    auto                                     end     = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::nano> elapsed = end - start;

    if (world_rank == 0) 
    { 
        std::cout << "First element of the result matrix: " << global_result[0] << std::endl;
        std::cout << "Elapsed time: " << elapsed.count() << " ns" << std::endl; 
    }

    MPI_Finalize();
    return 0;
}
