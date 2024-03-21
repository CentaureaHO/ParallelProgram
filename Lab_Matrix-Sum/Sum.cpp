#include <bits/stdc++.h>
using namespace std;
using ns = chrono::nanoseconds;
using hrClk = chrono::high_resolution_clock;

ns SequentialSum(const vector<int>& Nums, long long& Sum)
{
    auto Start = hrClk::now();
    Sum = 0;
    for (const int& Num : Nums) Sum += Num;
    auto End = hrClk::now();
    return chrono::duration_cast<ns>(End - Start);
}

void ParallelSum(const vector<int>& Nums, int Left, int Right, long long& Sum, int Depth = 0) 
{
    if (Left == Right) 
    {
        Sum += Nums[Left];
        return;
    }
    if ((Right - Left <= 100000) || (Depth >= 4))
    {
        for (int i = Left; i <= Right; i++) Sum += Nums[i];
        return;
    }
    int Mid = Left + (Right - Left) / 2;
    long long SumLeft = 0, SumRight = 0;
    thread LeftThread(ParallelSum, std::ref(Nums), Left, Mid, std::ref(SumLeft), Depth + 1);
    thread RightThread(ParallelSum, std::ref(Nums), Mid + 1, Right, std::ref(SumRight), Depth + 1);
    LeftThread.join();
    RightThread.join();
    Sum += SumLeft + SumRight;
}

ns ConcurrentSum(const vector<int>& Nums, long long& Sum) 
{
    auto Start = hrClk::now();
    Sum = 0;
    int n = Nums.size();
    ParallelSum(Nums, 0, Nums.size() - 1, Sum);
    auto End = hrClk::now();
    return chrono::duration_cast<ns>(End - Start);
}

ns UnrollSum(const vector<int>& Nums, long long& Sum) 
{
    auto Start = hrClk::now();
    Sum = 0;
    int i = 0, n = Nums.size();
    for (; i < n - 4; i += 4) 
    {
        Sum += Nums[i];
        Sum += Nums[i + 1];
        Sum += Nums[i + 2];
        Sum += Nums[i + 3];
    }
    for (; i < n; i++) Sum += Nums[i];
    auto End = hrClk::now();
    return chrono::duration_cast<ns>(End - Start);
}

int main()
{
    int Size = 1000000000, Repeat = 10;
    cout << "Enter the test size: ";
    cin >> Size;
    cout << "Enter the number of repetitions: ";
    cin >> Repeat;
    vector<int> Nums(Size);
    int Seed = 17171017;
    int Tmp1 = 0, Tmp2 = 0, Shift = 0;
    for (int i = 0; i < Size; i++)
    {
        Shift = Seed >> 7;
        Tmp1 = Seed ^ Shift;
        Shift = Tmp1 << 9;
        Seed = Tmp1 ^ Shift;
        Shift = Seed >> 6;
        Seed = Tmp2 ^ Shift;
        Nums[i] = Seed % 10000;
    }
    long long Sum = 0;
    cout << "Size: " << Size << "\n";
    ns CurTime, AvgTime, MinTime, MaxTime;
    
    AvgTime = MinTime = MaxTime = ns(0);
    for (int i = 0; i < Repeat; i++) 
    {
        CurTime = SequentialSum(Nums, Sum);
        if (i == 0) MinTime = MaxTime = CurTime;
        else 
        {
            MinTime = min(MinTime, CurTime);
            MaxTime = max(MaxTime, CurTime);
        }
        AvgTime += CurTime;
    }
    AvgTime /= Repeat;
    cout << "SequentialSum:\n";
    cout << "\tAverage time: " << AvgTime.count() << " ns\n";
    cout << "\tMinimum time: " << MinTime.count() << " ns\n";
    cout << "\tMaximum time: " << MaxTime.count() << " ns\n";

    AvgTime = MinTime = MaxTime = ns(0);
    for (int i = 0; i < Repeat; i++) 
    {
        CurTime = ConcurrentSum(Nums, Sum);
        if (i == 0) MinTime = MaxTime = CurTime;
        else 
        {
            MinTime = min(MinTime, CurTime);
            MaxTime = max(MaxTime, CurTime);
        }
        AvgTime += CurTime;
    }
    AvgTime /= Repeat;
    cout << "\nConcurrentSum:\n";
    cout << "\tAverage time: " << AvgTime.count() << " ns\n";
    cout << "\tMinimum time: " << MinTime.count() << " ns\n";
    cout << "\tMaximum time: " << MaxTime.count() << " ns\n";

    AvgTime = MinTime = MaxTime = ns(0);
    for (int i = 0; i < Repeat; i++) 
    {
        CurTime = UnrollSum(Nums, Sum);
        if (i == 0) MinTime = MaxTime = CurTime;
        else 
        {
            MinTime = min(MinTime, CurTime);
            MaxTime = max(MaxTime, CurTime);
        }
        AvgTime += CurTime;
    }
    AvgTime /= Repeat;
    cout << "\nUnrollSum:\n";
    cout << "\tAverage time: " << AvgTime.count() << " ns\n";
    cout << "\tMinimum time: " << MinTime.count() << " ns\n";
    cout << "\tMaximum time: " << MaxTime.count() << " ns\n";
}