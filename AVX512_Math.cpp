#include <immintrin.h>
#include <cmath>
#include <cstdio>

float tarctan(float x)
{
    float sqr = x * x;
    float e   = x;
    float r   = 0;
    int   i   = 1;
    while (fabs(e / i) > 1e-15)
    {
        float f = e / i;
        r       = (i % 4 == 1) ? r + f : r - f;
        e *= sqr;
        i += 2;
    }
    return r;
}

float arctan(float x)
{
    if (abs(x - 1) <= 1e-9) return M_PI / 4;
    else if (abs(x + 1) <= 1e-9) return -M_PI / 4;
    if (x >= -1 && x <= 1)
        return tarctan(x);
    else
    {
        if (x > 0)
            return (M_PI / 2 - tarctan(1 / x));
        else
            return (-M_PI / 2 - tarctan(1 / x));
    }
}

float arctan2(float y, float x)
{
    double result = 0;
    if (x == 0 && y > 0) result = M_PI / 2;
    else if (x == 0 && y < 0) result = -M_PI / 2;
    else
    {
        result = arctan(y / x);
        if (x < 0)
        {
            if (y >= 0) result += M_PI;
            else if (y < 0) result -= M_PI;
        }
    }
    return result;
}

__m512 _mm512_tarctan(__m512 x);
__m512 _mm512_arctan(__m512 x);
__m512 _mm512_arctan2(__m512 y, __m512 x);

int main()
{
    printf("arctan(1) = %f\n", arctan(0.9999999999));
    printf("atan(1) = %f\n", atan(0.9999999999));
    double x = 0.5;
    double y = 0.5;
    printf("arctan2(%f, %f) = %f\n", y, x, arctan2(y, x));
    printf("atan2(%f, %f) = %f\n", y, x, atan2(y, x));
}