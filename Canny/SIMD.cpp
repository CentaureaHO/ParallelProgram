#include "SIMD.h"
#include "GaussDef.h"
#include <xmmintrin.h>
#include <immintrin.h>
#include <vector>
const int KernelRadius = 1;

void SSE::PerformGaussianBlur(uint8_t* Output, const uint8_t* OriImg, int Width, int Height)
{
    std::vector<float> Tmp(Width * Height, 0.0f);
    for (int y = 0; y < Height; y++)
    {
        for (int x = 0; x < Width; x++)
        {
            __m128 PixelVal  = _mm_setzero_ps();
            __m128 KernelSum = _mm_setzero_ps();
            for (int i = -KernelRadius; i <= KernelRadius; i++)
            {
                int nx = x + i;
                if (nx >= 0 && nx < Width)
                {
                    int    ImgIdx    = y * Width + nx;
                    __m128 ImgPixel  = _mm_set1_ps((float)OriImg[ImgIdx]);
                    __m128 KernelVal = _mm_set1_ps(GaussianKernel_1D[KernelRadius + i]);
                    PixelVal         = _mm_add_ps(PixelVal, _mm_mul_ps(ImgPixel, KernelVal));
                    KernelSum        = _mm_add_ps(KernelSum, KernelVal);
                }
            }
            PixelVal       = _mm_div_ps(PixelVal, KernelSum);
            int OutputIdx  = y * Width + x;
            Tmp[OutputIdx] = _mm_cvtss_f32(_mm_shuffle_ps(PixelVal, PixelVal, _MM_SHUFFLE(0, 0, 0, 0)));
        }
    }

    for (int x = 0; x < Width; x++)
    {
        for (int y = 0; y < Height; y++)
        {
            __m128 PixelVal  = _mm_setzero_ps();
            __m128 KernelSum = _mm_setzero_ps();
            for (int i = -KernelRadius; i <= KernelRadius; i++)
            {
                int ny = y + i;
                if (ny >= 0 && ny < Height)
                {
                    int    ImgIdx    = ny * Width + x;
                    __m128 ImgPixel  = _mm_set1_ps(Tmp[ImgIdx]);
                    __m128 KernelVal = _mm_set1_ps(GaussianKernel_1D[KernelRadius + i]);
                    PixelVal         = _mm_add_ps(PixelVal, _mm_mul_ps(ImgPixel, KernelVal));
                    KernelSum        = _mm_add_ps(KernelSum, KernelVal);
                }
            }
            PixelVal          = _mm_div_ps(PixelVal, KernelSum);
            int OutputIdx     = y * Width + x;
            Output[OutputIdx] = (uint8_t)std::min(
                std::max((int)(_mm_cvtss_f32(_mm_shuffle_ps(PixelVal, PixelVal, _MM_SHUFFLE(0, 0, 0, 0))) + 0.5f), 0),
                255);
        }
    }
}

void AVX::PerformGaussianBlur(uint8_t* Output, const uint8_t* OriImg, int Width, int Height)
{
    std::vector<float> Tmp(Width * Height, 0.0f);
    for (int y = 0; y < Height; y++)
    {
        for (int x = 0; x < Width; x++)
        {
            __m256 PixelVal  = _mm256_setzero_ps();
            __m256 KernelSum = _mm256_setzero_ps();
            for (int i = -KernelRadius; i <= KernelRadius; i++)
            {
                int nx = x + i;
                if (nx >= 0 && nx < Width)
                {
                    int    ImgIdx  = y * Width + nx;
                    __m256 ImgPixel  = _mm256_set1_ps((float)OriImg[ImgIdx]);
                    __m256 Kernel = _mm256_set1_ps(GaussianKernel_1D[KernelRadius + i]);
                    PixelVal         = _mm256_add_ps(PixelVal, _mm256_mul_ps(ImgPixel, Kernel));
                    KernelSum        = _mm256_add_ps(KernelSum, Kernel);
                }
            }
            PixelVal         = _mm256_div_ps(PixelVal, KernelSum);
            int OutputIdx  = y * Width + x;
            Tmp[OutputIdx] = _mm_cvtss_f32(_mm256_castps256_ps128(PixelVal));
        }
    }

    for (int x = 0; x < Width; x++)
    {
        for (int y = 0; y < Height; y++)
        {
            __m256 PixelVal  = _mm256_setzero_ps();
            __m256 KernelSum = _mm256_setzero_ps();
            for (int i = -KernelRadius; i <= KernelRadius; i++)
            {
                int ny = y + i;
                if (ny >= 0 && ny < Height)
                {
                    int    ImgIdx  = ny * Width + x;
                    __m256 ImgPixel  = _mm256_set1_ps(Tmp[ImgIdx]);
                    __m256 KernelVal = _mm256_set1_ps(GaussianKernel_1D[KernelRadius + i]);
                    PixelVal         = _mm256_add_ps(PixelVal, _mm256_mul_ps(ImgPixel, KernelVal));
                    KernelSum        = _mm256_add_ps(KernelSum, KernelVal);
                }
            }
            PixelVal            = _mm256_div_ps(PixelVal, KernelSum);
            PixelVal            = _mm256_min_ps(_mm256_max_ps(PixelVal, _mm256_setzero_ps()), _mm256_set1_ps(255.0f));
            int OutputIdx     = y * Width + x;
            Output[OutputIdx] = (uint8_t)_mm_cvtss_f32(_mm256_castps256_ps128(PixelVal));  // Correct conversion
        }
    }
}