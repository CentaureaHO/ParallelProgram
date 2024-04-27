#include <iostream>
#include <immintrin.h> // AVX、AVX2
#define n 1007
using namespace std;

__attribute__((aligned(32))) float A[n][n];

void init(float A[n][n])
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            A[i][j] = i + j;
}

void SIMD_AVX_gausseliminate(float A[n][n])
{
    __m256 t1, t2, t3; // 八位单精度构成的向量

    for (int k = 0; k < n; k++)
    {
        int preprocessnumber = (n - k - 1) % 8; // 预处理的数量,能被八整除
        int begin = k + 1 + preprocessnumber;
        __attribute__((aligned(32))) float head[8] = {A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k]};
        t2 = _mm256_load_ps(head);
        for (int j = k + 1; j < k + 1 + preprocessnumber; j++)
        {
            A[k][j] = A[k][j] / A[k][k];
        }
        for (int j = begin; j < n; j += 8)
        {
            t1 = _mm256_load_ps(A[k] + j);
            t1 = _mm256_div_ps(t1, t2);
            _mm256_store_ps(A[k] + j, t1);
        }
        A[k][k] = 0;
        t1 = _mm256_setzero_ps(); // 清零
        t2 = _mm256_setzero_ps();
        // 先去头，为了四个四个的处理

        for (int i = k + 1; i < n; i++)
        {
            for (int j = k + 1; j < k + 1 + preprocessnumber; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }

        for (int i = k + 1; i < n; i++)
        {
            __attribute__((aligned(32))) float head1[8] = {A[i][k], A[i][k], A[i][k], A[i][k], A[i][k], A[i][k], A[i][k], A[i][k]};
            t3 = _mm256_load_ps(head1);
            for (int j = begin; j < n; j += 8)
            {
                t1 = _mm256_load_ps(A[k] + j);
                t2 = _mm256_load_ps(A[i] + j);
                t1 = _mm256_mul_ps(t1, t3);
                t2 = _mm256_sub_ps(t2, t1);
                _mm256_store_ps(A[i] + j, t2);
            }
            A[i][k] = 0;
        }
    }
}

int main()
{
    int a = 0;
    cin >> a;
    cout << a / 16 * 16;
}