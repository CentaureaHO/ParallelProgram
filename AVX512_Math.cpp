#include <immintrin.h>
#include <cmath>
#include <cstdio>
#include <bits/stdc++.h>

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
    if (abs(x - 1) <= 1e-9)
        return M_PI / 4;
    else if (abs(x + 1) <= 1e-9)
        return -M_PI / 4;
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
    if (x == 0 && y > 0)
        result = M_PI / 2;
    else if (x == 0 && y < 0)
        result = -M_PI / 2;
    else
    {
        result = arctan(y / x);
        if (x < 0)
        {
            if (y >= 0)
                result += M_PI;
            else if (y < 0)
                result -= M_PI;
        }
    }
    return result;
}

__m512 _mm512_tarctan(__m512 x)
{
    const __m512  sqr       = _mm512_mul_ps(x, x);
    __m512        e         = x;
    __m512        r         = _mm512_setzero_ps();
    const __m512  threshold = _mm512_set1_ps(1e-15);
    __m512        f;
    __m512i       i    = _mm512_set1_epi32(1);
    const __m512i two  = _mm512_set1_epi32(2);
    __m512        sign = _mm512_set1_ps(1.0f);

    for (;;)
    {
        f = _mm512_div_ps(e, _mm512_cvtepi32_ps(i));
        if (_mm512_cmp_ps_mask(_mm512_abs_ps(f), threshold, _CMP_LE_OQ) == 0xFFFF) break;

        r = _mm512_add_ps(r, _mm512_mul_ps(sign, f));

        sign = _mm512_mul_ps(sign, _mm512_set1_ps(-1.0f));
        e    = _mm512_mul_ps(e, sqr);
        i    = _mm512_add_epi32(i, two);
    }

    return r;
}

__m512 _mm512_arctan(__m512 x)
{
    __m512 t0 = _mm512_abs_ps(x);

    __m512 t0_squared = _mm512_mul_ps(t0, t0);

    __m512 factor      = _mm512_set1_ps(0.28f);
    __m512 denominator = _mm512_fmadd_ps(t0_squared, factor, _mm512_set1_ps(1.0f));
    __m512 t1          = _mm512_div_ps(t0, denominator);

    __m512 t1_squared = _mm512_mul_ps(t1, t1);

    __m512 term = t1;
    __m512 t3   = term;
    term        = _mm512_mul_ps(term, t1_squared);
    t3          = _mm512_fnmadd_ps(term, _mm512_set1_ps(1.0f / 3.0f), t3);
    term        = _mm512_mul_ps(term, t1_squared);
    t3          = _mm512_fmadd_ps(term, _mm512_set1_ps(1.0f / 5.0f), t3);
    term        = _mm512_mul_ps(term, t1_squared);
    t3          = _mm512_fnmadd_ps(term, _mm512_set1_ps(1.0f / 7.0f), t3);
    term        = _mm512_mul_ps(term, t1_squared);
    t3          = _mm512_fmadd_ps(term, _mm512_set1_ps(1.0f / 9.0f), t3);

    __m512 pi_over_two = _mm512_set1_ps(1.570796327f);
    __m512 pi          = _mm512_set1_ps(3.141592654f);
    __m512 result      = _mm512_mask_sub_ps(t3, _mm512_cmp_ps_mask(x, _mm512_setzero_ps(), _CMP_LT_OQ), pi, t3);

    return result;
}

__m512 _mm512_arctan2(__m512 y, __m512 x)
{
    __m512 abs_y = _mm512_abs_ps(y);
    __m512 abs_x = _mm512_abs_ps(x);

    __m512 t0 = _mm512_max_ps(abs_x, abs_y);
    __m512 t1 = _mm512_min_ps(abs_x, abs_y);

    __m512 t3 = _mm512_div_ps(t1, t0);

    __m512 t4 = _mm512_mul_ps(t3, t3);
    t0        = _mm512_fmadd_ps(_mm512_set1_ps(-0.013480470), t4, _mm512_set1_ps(0.057477314));
    t0        = _mm512_fmsub_ps(t0, t4, _mm512_set1_ps(0.121239071));
    t0        = _mm512_fmadd_ps(t0, t4, _mm512_set1_ps(0.195635925));
    t0        = _mm512_fmsub_ps(t0, t4, _mm512_set1_ps(0.332994597));
    t0        = _mm512_fmadd_ps(t0, t4, _mm512_set1_ps(0.999995630));
    t3        = _mm512_mul_ps(t0, t3);

    __m512 pi_over_two = _mm512_set1_ps(1.570796327f);
    __m512 pi          = _mm512_set1_ps(3.141592654f);
    t3                 = _mm512_mask_sub_ps(t3, _mm512_cmp_ps_mask(x, _mm512_setzero_ps(), _CMP_LT_OQ), pi, t3);
    t3                 = _mm512_mask_sub_ps(t3, _mm512_cmp_ps_mask(abs_y, abs_x, _CMP_GT_OQ), pi_over_two, t3);
    t3 = _mm512_mask_mul_ps(t3, _mm512_cmp_ps_mask(y, _mm512_setzero_ps(), _CMP_LT_OQ), t3, _mm512_set1_ps(-1.0));

    __mmask16 mask_second_quad =
        _mm512_cmp_ps_mask(x, _mm512_setzero_ps(), _CMP_LT_OQ) & _mm512_cmp_ps_mask(y, _mm512_setzero_ps(), _CMP_GE_OQ);
    __mmask16 mask_third_quad =
        _mm512_cmp_ps_mask(x, _mm512_setzero_ps(), _CMP_LT_OQ) & _mm512_cmp_ps_mask(y, _mm512_setzero_ps(), _CMP_LT_OQ);

    t3 = _mm512_mask_add_ps(t3, _mm512_cmp_ps_mask(t3, _mm512_setzero_ps(), _CMP_LT_OQ) & mask_second_quad, t3, pi);
    t3 = _mm512_mask_sub_ps(t3, _mm512_cmp_ps_mask(t3, _mm512_setzero_ps(), _CMP_GT_OQ) & mask_third_quad, t3, pi);

    return t3;
}

int main()
{
    std::random_device                    rd;
    std::mt19937                          gen(rd());
    std::uniform_real_distribution<float> dis(-100.0, 100.0);

    __m512 y = _mm512_setzero_ps();
    __m512 x = _mm512_setzero_ps();

    for (int i = 0; i < 16; i++)
    {
        reinterpret_cast<float*>(&y)[i] = dis(gen);
        reinterpret_cast<float*>(&x)[i] = dis(gen);
    }
    __m512 lengths = _mm512_sqrt_ps(_mm512_add_ps(_mm512_mul_ps(y, y), _mm512_mul_ps(x, x)));
    y              = _mm512_div_ps(y, lengths);
    x              = _mm512_div_ps(x, lengths);

    auto   start_avx512  = std::chrono::high_resolution_clock::now();
    __m512 result_avx512 = _mm512_arctan2(y, x);
    auto   end_avx512    = std::chrono::high_resolution_clock::now();

    float result_scalar[16];
    auto  start_scalar = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 16; i++)
    {
        float len        = std::sqrt(reinterpret_cast<float*>(&y)[i] * reinterpret_cast<float*>(&y)[i] +
                              reinterpret_cast<float*>(&x)[i] * reinterpret_cast<float*>(&x)[i]);
        float norm_y     = reinterpret_cast<float*>(&y)[i] / len;
        float norm_x     = reinterpret_cast<float*>(&x)[i] / len;
        result_scalar[i] = std::atan2(norm_y, norm_x);
    }
    auto end_scalar = std::chrono::high_resolution_clock::now();

    std::cout << "y: ";
    for (int i = 0; i < 16; i++) { std::cout << reinterpret_cast<float*>(&y)[i] << " "; }
    std::cout << "\nx: ";
    for (int i = 0; i < 16; i++) { std::cout << reinterpret_cast<float*>(&x)[i] << " "; }
    std::cout << "\narctan: ";
    for (int i = 0; i < 16; i++)
    {
        std::cout << atan(reinterpret_cast<float*>(&y)[i] / reinterpret_cast<float*>(&x)[i]) << " ";
    }

    std::cout << "\nAVX-512 results:\n";
    for (int i = 0; i < 16; i++) { std::cout << reinterpret_cast<float*>(&result_avx512)[i] << " "; }
    std::cout << "\nScalar results:\n";
    for (int i = 0; i < 16; i++) { std::cout << result_scalar[i] << " "; }

    auto avx512_time = std::chrono::duration_cast<std::chrono::microseconds>(end_avx512 - start_avx512).count();
    auto scalar_time = std::chrono::duration_cast<std::chrono::microseconds>(end_scalar - start_scalar).count();
    std::cout << "\nTime taken (AVX-512): " << avx512_time << " microseconds\n";
    std::cout << "Time taken (Scalar): " << scalar_time << " microseconds\n";

    return 0;
}