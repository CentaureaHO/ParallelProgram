#include <bits/stdc++.h>
using namespace std;
using ns = chrono::nanoseconds;
using hrClk = chrono::high_resolution_clock;

ns Trivial(const vector<vector<int>> &Matrix, const vector<int> &Vec, vector<long long> &Result)
{
    fill(Result.begin(), Result.end(), 0);
    auto Start = hrClk::now();
    int n = Matrix.size();
    for (int Col = 0; Col < n; Col++)
    {
        Result[Col] = 0;
        for (int Row = 0; Row < n; ++Row) Result[Col] += Matrix[Row][Col] * Vec[Row];
    }
    auto End = hrClk::now();
    return chrono::duration_cast<ns>(End - Start);
}

ns Optimized(const vector<vector<int>>& Matrix, const vector<int>& Vec, vector<long long>& Result) 
{
    fill(Result.begin(), Result.end(), 0);
    auto Start = hrClk::now();
    int n = Matrix.size();
    for (int Row = 0; Row < n; ++Row) 
    {
        for (int Col = 0; Col < n; ++Col) Result[Col] += Matrix[Row][Col] * Vec[Row];
    }
    auto End = hrClk::now();
    return chrono::duration_cast<ns>(End - Start);
}

template<int N>
struct UnrollLoop 
{
    template<typename Func>
    static inline void Do(Func func, int Row, vector<long long>& Result) 
    {
        UnrollLoop<N-1>::Do(func, Row, Result);
        func(Row, N-1, Result);
    }
};

template<>
struct UnrollLoop<0> 
{
    template<typename Func>
    static inline void Do(Func func, int Row, vector<long long>& Result) {}
};

ns Unroll(const vector<vector<int>>& Matrix, const vector<int>& Vec, vector<long long>& Result) 
{
    fill(Result.begin(), Result.end(), 0);
    auto Start = hrClk::now();
    int n = Matrix.size();
    const int UnrollFactor = 8;
    auto lambda = [&](int Row, int Col, vector<long long>& Result) { if (Col < n) Result[Col] += Matrix[Row][Col] * Vec[Row]; };
    for (int Row = 0; Row < n; ++Row)
        for (int Col = 0; Col < n; Col += UnrollFactor)
            UnrollLoop<UnrollFactor>::Do([&](int R, int C, vector<long long>& Res) { lambda(R, Col + C, Res); }, Row, Result);
    auto End = hrClk::now();
    return chrono::duration_cast<ns>(End - Start);
}

int FakeRand(int& Seed)
{
    int Tmp1 = 0, Tmp2 = 0, Shift = 0;
    Shift = Seed >> 7;
    Tmp1 = Seed ^ Shift;
    Shift = Tmp1 << 9;
    Seed = Tmp1 ^ Shift;
    Shift = Seed >> 6;
    Seed = Tmp2 ^ Shift;
    return Seed;
}

int main()
{
    int Size = 30000, Repeat = 10;
    cout << "Enter the test size: ";
    cin >> Size;
    cout << "Enter the number of repetitions: ";
    cin >> Repeat;
    vector<int> Vec(Size);
    vector<long long> Result(Size);
    vector<vector<int>> Matrix(Size, vector<int>(Size));
    int Seed = 17171017;
    for (int i = 0; i < Size; i++)
    {
        Vec[i] = FakeRand(Seed) % 1000;
        for (int j = 0; j < Size; j++) Matrix[i][j] = FakeRand(Seed) % 1000;
    }
    ns CurTime, AvgTime, MinTime, MaxTime;

    AvgTime = MinTime = MaxTime = ns(0);
    for (int i = 0; i < Repeat; i++) 
    {
        CurTime = Trivial(Matrix, Vec, Result);
        if (i == 0) MinTime = MaxTime = CurTime;
        else 
        {
            if (CurTime < MinTime) MinTime = CurTime;
            if (CurTime > MaxTime) MaxTime = CurTime;
        }
        AvgTime += CurTime;
    }
    AvgTime /= Repeat;
    cout << "Trivial:\n";
    cout <<"\tAverage time: " << AvgTime.count() << " ns\n";
    cout <<"\tMinimum time: " << MinTime.count() << " ns\n";
    cout <<"\tMaximum time: " << MaxTime.count() << " ns\n";

    AvgTime = MinTime = MaxTime = ns(0);
    for (int i = 0; i < Repeat; i++) 
    {
        CurTime = Optimized(Matrix, Vec, Result);
        if (i == 0) MinTime = MaxTime = CurTime;
        else 
        {
            if (CurTime < MinTime) MinTime = CurTime;
            if (CurTime > MaxTime) MaxTime = CurTime;
        }
        AvgTime += CurTime;
    }
    AvgTime /= Repeat;
    cout << "\nOptimized:\n";
    cout <<"\tAverage time: " << AvgTime.count() << " ns\n";
    cout <<"\tMinimum time: " << MinTime.count() << " ns\n";
    cout <<"\tMaximum time: " << MaxTime.count() << " ns\n";

    AvgTime = MinTime = MaxTime = ns(0);
    for (int i = 0; i < Repeat; i++) 
    {
        CurTime = Unroll(Matrix, Vec, Result);
        if (i == 0) MinTime = MaxTime = CurTime;
        else 
        {
            if (CurTime < MinTime) MinTime = CurTime;
            if (CurTime > MaxTime) MaxTime = CurTime;
        }
        AvgTime += CurTime;
    }
    AvgTime /= Repeat;
    cout << "\nUnroll:\n";
    cout <<"\tAverage time: " << AvgTime.count() << " ns\n";
    cout <<"\tMinimum time: " << MinTime.count() << " ns\n";
    cout <<"\tMaximum time: " << MaxTime.count() << " ns\n";
}