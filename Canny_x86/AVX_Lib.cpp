#include "AVX_Lib.h"

__m512 _mm512_arctan2(__m512 y, __m512 x)
{
    __m512 absY = _mm512_abs_ps(y);
    __m512 absX = _mm512_abs_ps(x);

    __m512 t0 = _mm512_max_ps(absX, absY);
    __m512 t1 = _mm512_min_ps(absX, absY);

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
    t3                 = _mm512_mask_sub_ps(t3, _mm512_cmp_ps_mask(absY, absX, _CMP_GT_OQ), pi_over_two, t3);
    t3 = _mm512_mask_mul_ps(t3, _mm512_cmp_ps_mask(y, _mm512_setzero_ps(), _CMP_LT_OQ), t3, _mm512_set1_ps(-1.0));

    __mmask16 SecondQuadMask =
        _mm512_cmp_ps_mask(x, _mm512_setzero_ps(), _CMP_LT_OQ) & _mm512_cmp_ps_mask(y, _mm512_setzero_ps(), _CMP_GE_OQ);
    __mmask16 ThirdQuadMask =
        _mm512_cmp_ps_mask(x, _mm512_setzero_ps(), _CMP_LT_OQ) & _mm512_cmp_ps_mask(y, _mm512_setzero_ps(), _CMP_LT_OQ);

    t3 = _mm512_mask_add_ps(t3, _mm512_cmp_ps_mask(t3, _mm512_setzero_ps(), _CMP_LT_OQ) & SecondQuadMask, t3, pi);
    t3 = _mm512_mask_sub_ps(t3, _mm512_cmp_ps_mask(t3, _mm512_setzero_ps(), _CMP_GT_OQ) & ThirdQuadMask, t3, pi);

    return t3;
}

__m128i cvtepi32_epi8(__m512i v)
{
    __m256i v0_32 = _mm512_extracti32x8_epi32(v, 0);
    __m256i v1_32 = _mm512_extracti32x8_epi32(v, 1);
    __m128i v0_16 = _mm256_cvtepi32_epi16(v0_32);
    __m128i v1_16 = _mm256_cvtepi32_epi16(v1_32);
    __m128i v8    = _mm_packus_epi16(v0_16, v1_16);
    return v8;
}