#include <mpi.h>
#include <bits/stdc++.h>
using namespace std;

void Ring_AllReduce(std::vector<int>& Data, int NumNodes)
{
    int Rank, Size;
    MPI_Comm_rank(MPI_COMM_WORLD, &Rank);
    MPI_Comm_size(MPI_COMM_WORLD, &Size);

    int         SizePerNode = Data.size();
    vector<int> RecvBuffer(SizePerNode);

    for (int step = 0; step < NumNodes - 1; ++step)
    {
        int SendTo   = (Rank + 1) % NumNodes;
        int RecvFrom = (Rank - 1 + NumNodes) % NumNodes;

        MPI_Sendrecv(Data.data(),
            SizePerNode,
            MPI_INT,
            SendTo,
            0,
            RecvBuffer.data(),
            SizePerNode,
            MPI_INT,
            RecvFrom,
            0,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE);

        for (int i = 0; i < SizePerNode; ++i) Data[i] += RecvBuffer[i];
    }

    MPI_Allreduce(MPI_IN_PLACE, Data.data(), SizePerNode, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
}

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

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int Rank, Size;
    MPI_Comm_rank(MPI_COMM_WORLD, &Rank);
    MPI_Comm_size(MPI_COMM_WORLD, &Size);

    int Nodes       = Size;
    int NumsPerNode = 10000000;
    int TotalSize   = NumsPerNode * Nodes;

    vector<int> GlobalData;
    vector<int> LocalData(NumsPerNode);

    if (Rank == 0)
    {
        GlobalData.resize(TotalSize);
        int Seed = 114514;
        for (int& val : GlobalData) { val = FakeRand(Seed) % 1000; }

        int Remainder = TotalSize % Nodes;
        if (Remainder != 0)
        {
            for (int i = 0; i < Nodes - Remainder; ++i) { GlobalData.push_back(0); }
        }
        TotalSize = GlobalData.size();
    }

    MPI_Scatter(GlobalData.data(), NumsPerNode, MPI_INT, LocalData.data(), NumsPerNode, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    auto Start = chrono::high_resolution_clock::now();

    Ring_AllReduce(LocalData, Nodes);

    auto                                     End      = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::nano> Duration = End - Start;

    MPI_Barrier(MPI_COMM_WORLD);
    for (int Node = 0; Node < Nodes; ++Node)
    {
        if (Rank == Node) { std::cout << "Rank " << Rank << " Elapsed: " << Duration.count() << " ns\n"; }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    if (Rank == 0)
    {
        int TotalSum = 0;
        for (int val : GlobalData) { TotalSum += val; }
        cout << "Total sum of global data: " << TotalSum;
    }

    MPI_Finalize();
}
