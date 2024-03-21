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

ns VectorizedSum(const vector<int>& Nums, long long& Sum) 
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
    int Size = 1000000000;
    cout << "Enter the test size: ";
    cin >> Size;
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
    ns SeqTime = SequentialSum(Nums, Sum);
    cout << "Sequential Sum: " << Sum << " Time: " << SeqTime.count() << "ns\n";
    ns VecTime = VectorizedSum(Nums, Sum);
    cout << "Vectorized Sum: " << Sum << " Time: " << VecTime.count() << "ns\n";
    ns UnrollTime = UnrollSum(Nums, Sum);
    cout << "Unroll Sum: " << Sum << " Time: " << UnrollTime.count() << "ns\n";
}