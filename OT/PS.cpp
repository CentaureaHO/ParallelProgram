#include <mpi.h>
#include <bits/stdc++.h>
using namespace std;

int FakeRand(int& Seed)
{
    int Tmp1 = 0, Tmp2 = 0, Shift = 0;
    Shift = Seed >> 7;
    Tmp1  = Seed ^ Shift;
    Shift = Tmp1 << 9;
    Seed  = Tmp1 ^ Shift;
    Shift = Seed >> 6;
    Seed  = Tmp2 ^ Shift;
    return Seed;
}

void Server(int NumWorkers, int DataSize)
{
    vector<int> GlobalData(DataSize);

    int Seed = 114514;
    for (int& Val : GlobalData) Val = FakeRand(Seed) % 1000;

    auto Start     = chrono::high_resolution_clock::now();
    int  ChunkSize = DataSize / NumWorkers;
    int  Remainder = DataSize % NumWorkers;
    for (int i = 0; i < NumWorkers; ++i)
    {
        int SendSize = ChunkSize + (i < Remainder ? 1 : 0);
        MPI_Send(
            GlobalData.data() + i * ChunkSize + std::min(i, Remainder), SendSize, MPI_INT, i + 1, 0, MPI_COMM_WORLD);
    }

    vector<int> RecvBuffer(DataSize / NumWorkers + (Remainder ? 1 : 0));
    vector<int> SummedData(DataSize, 0);

    for (int i = 0; i < NumWorkers; ++i)
    {
        int RecvSize = ChunkSize + (i < Remainder ? 1 : 0);
        MPI_Recv(RecvBuffer.data(), RecvSize, MPI_INT, i + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (int j = 0; j < RecvSize; ++j) SummedData[i * ChunkSize + std::min(i, Remainder) + j] = RecvBuffer[j];
    }
    auto                                End      = chrono::high_resolution_clock::now();
    chrono::duration<double, std::nano> Duration = End - Start;
    cout << "Server elapsed time: " << Duration.count() << " ns\n";

    int TotalSum = std::accumulate(SummedData.begin(), SummedData.end(), 0);
    cout << "Total sum of global data: " << TotalSum;
}

void Worker(int Rank, int DataSize, int NumWorkers)
{
    int ChunkSize = DataSize / NumWorkers;
    int Remainder = DataSize % NumWorkers;
    int LocalSize = ChunkSize + (Rank <= Remainder ? 1 : 0);

    vector<int> LocalData(LocalSize);

    MPI_Recv(LocalData.data(), LocalSize, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    int LocalSum = std::accumulate(LocalData.begin(), LocalData.end(), 0);

    MPI_Send(&LocalSum, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int Rank, Size;
    MPI_Comm_rank(MPI_COMM_WORLD, &Rank);
    MPI_Comm_size(MPI_COMM_WORLD, &Size);

    int NumWorkers = Size - 1;
    int DataSize   = 10000000 * Size;

    if (Rank == 0)
        Server(NumWorkers, DataSize);
    else
        Worker(Rank - 1, DataSize, NumWorkers);

    MPI_Finalize();
}
