#include <immintrin.h>
#include <math.h>
#include <pthread.h>
#include "AVX_Lib.h"
#include "PThread.h"

void PThread::ComputeGradients(float* Gradients, uint8_t* GradDires, const uint8_t* BlurredImage, int Width, int Height)
{
    const int8_t Gx[]   = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    const int8_t Gy[]   = {1, 2, 1, 0, 0, 0, -1, -2, -1};
    const int    Offset = 1;

    for (int y = Offset; y < Height - Offset; y++)
    {
        for (int x = Offset; x < Width - Offset; x += 16)
        {
            __m512 GradX = _mm512_setzero_ps();
            __m512 GradY = _mm512_setzero_ps();

            for (int ky = -Offset; ky <= Offset; ky++)
            {
                for (int kx = -Offset; kx <= Offset; kx++)
                {
                    int KernelIdx = (ky + Offset) * 3 + (kx + Offset);
                    int PixelIdx  = x + (y * Width) + kx + (ky * Width);

                    if (x + 15 < Width)
                    {
                        __m512i PixelValues =
                            _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(BlurredImage + PixelIdx)));
                        __m512i GxValue = _mm512_set1_epi32(Gx[KernelIdx]);
                        __m512i GyValue = _mm512_set1_epi32(Gy[KernelIdx]);

                        GradX = _mm512_add_ps(
                            GradX, _mm512_mul_ps(_mm512_cvtepi32_ps(PixelValues), _mm512_cvtepi32_ps(GxValue)));
                        GradY = _mm512_add_ps(
                            GradY, _mm512_mul_ps(_mm512_cvtepi32_ps(PixelValues), _mm512_cvtepi32_ps(GyValue)));
                    }
                }
            }

            __m512 Magnitude = _mm512_sqrt_ps(_mm512_add_ps(_mm512_mul_ps(GradX, GradX), _mm512_mul_ps(GradY, GradY)));
            _mm512_storeu_ps(Gradients + x + y * Width, Magnitude);

            __m512 Degrees = _mm512_arctan2(GradY, GradX) * _mm512_set1_ps(360.0 / (2.0 * M_PI));

            __m512i   Directions = _mm512_setzero_si512();
            __mmask16 DireMask1  = (_mm512_cmp_ps_mask(Degrees, _mm512_set1_ps(22.5), _CMP_LE_OS) &
                                      _mm512_cmp_ps_mask(Degrees, _mm512_set1_ps(-22.5), _CMP_NLE_US)) |
                                  _mm512_cmp_ps_mask(Degrees, _mm512_set1_ps(-157.5), _CMP_LE_OS) |
                                  _mm512_cmp_ps_mask(Degrees, _mm512_set1_ps(157.5), _CMP_NLE_US);
            __mmask16 DireMask2 = (_mm512_cmp_ps_mask(Degrees, _mm512_set1_ps(22.5), _CMP_GT_OS) &
                                      _mm512_cmp_ps_mask(Degrees, _mm512_set1_ps(67.5), _CMP_LE_OS)) |
                                  (_mm512_cmp_ps_mask(Degrees, _mm512_set1_ps(-157.5), _CMP_GT_OS) &
                                      _mm512_cmp_ps_mask(Degrees, _mm512_set1_ps(-112.5), _CMP_LE_OS));
            __mmask16 DireMask3 = (_mm512_cmp_ps_mask(Degrees, _mm512_set1_ps(67.5), _CMP_GT_OS) &
                                      _mm512_cmp_ps_mask(Degrees, _mm512_set1_ps(112.5), _CMP_LE_OS)) |
                                  (_mm512_cmp_ps_mask(Degrees, _mm512_set1_ps(-112.5), _CMP_GE_OS) &
                                      _mm512_cmp_ps_mask(Degrees, _mm512_set1_ps(-67.5), _CMP_LT_OS));
            __mmask16 DireMask4 = (_mm512_cmp_ps_mask(Degrees, _mm512_set1_ps(-67.5), _CMP_GE_OS) &
                                      _mm512_cmp_ps_mask(Degrees, _mm512_set1_ps(-22.5), _CMP_LT_OS)) |
                                  (_mm512_cmp_ps_mask(Degrees, _mm512_set1_ps(112.5), _CMP_GT_OS) &
                                      _mm512_cmp_ps_mask(Degrees, _mm512_set1_ps(157.5), _CMP_LT_OS));
            Directions = _mm512_mask_add_epi32(Directions, DireMask1, Directions, _mm512_set1_epi32(1));
            Directions = _mm512_mask_add_epi32(Directions, DireMask2, Directions, _mm512_set1_epi32(2));
            Directions = _mm512_mask_add_epi32(Directions, DireMask3, Directions, _mm512_set1_epi32(3));
            Directions = _mm512_mask_add_epi32(Directions, DireMask4, Directions, _mm512_set1_epi32(4));

            _mm_storeu_si128((__m128i*)(GradDires + x + y * Width), _mm512_cvtsepi32_epi8(Directions));
        }
    }
}