#include <cstring>
#include <immintrin.h>
#include <iostream>
#include <math.h>
#include <algorithm>
#include <omp.h>
#include "AVX_Lib.h"
#include "ParmsDef.h"
#include "OpenMP.h"

OpenMP::OpenMP(unsigned int TN) : ThreadNum(static_cast<int>(TN)) { omp_set_num_threads(ThreadNum); }

OpenMP::~OpenMP() {}

OpenMP& OpenMP::GetInstance(unsigned int TN)
{
    static OpenMP Instance(TN);
    return Instance;
}

void OpenMP::PerformGaussianBlur(uint8_t* Output, const uint8_t* OriImg, int Width, int Height)
{
    int    PaddedWidth = (Width + 15) & ~15;
    float* Tmp         = (float*)_mm_malloc(PaddedWidth * Height * sizeof(float), 64);

    auto ProcessRow = [&](int y) {
        int x = 0;
        for (; x <= Width - 16; x += 16)
        {
            __m512 PixelVal  = _mm512_setzero_ps();
            __m512 KernelSum = _mm512_setzero_ps();
            for (int i = -KernelRadius; i <= KernelRadius; i++)
            {
                int nx = x + i;
                if (nx >= 0 && nx < Width)
                {
                    __m512i data      = _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)(OriImg + y * Width + nx)));
                    __m512  ImgPixel  = _mm512_cvtepi32_ps(data);
                    __m512  KernelVal = _mm512_set1_ps(GaussianKernel_1D[KernelRadius + i]);
                    PixelVal          = _mm512_add_ps(PixelVal, _mm512_mul_ps(ImgPixel, KernelVal));
                    KernelSum         = _mm512_add_ps(KernelSum, KernelVal);
                }
            }
            __m512 InvKernelSum = _mm512_div_ps(_mm512_set1_ps(1.0f), KernelSum);
            _mm512_store_ps(Tmp + y * PaddedWidth + x, _mm512_mul_ps(PixelVal, InvKernelSum));
        }
        for (; x < Width; x++)
        {
            float PixelVal = 0.0f, KernelSum = 0.0f;
            for (int i = -KernelRadius; i <= KernelRadius; i++)
            {
                int nx = x + i;
                if (nx >= 0 && nx < Width)
                {
                    float ImgPixel  = static_cast<float>(OriImg[y * Width + nx]);
                    float KernelVal = GaussianKernel_1D[KernelRadius + i];
                    PixelVal += ImgPixel * KernelVal;
                    KernelSum += KernelVal;
                }
            }
            Tmp[y * PaddedWidth + x] = PixelVal / KernelSum;
        }
    };

    auto ProcessColumn = [&](int y) {
        int x = 0;
        for (; x <= Width - 16; x += 16)
        {
            __m512 PixelVal  = _mm512_setzero_ps();
            __m512 KernelSum = _mm512_setzero_ps();
            for (int j = -KernelRadius; j <= KernelRadius; j++)
            {
                int ny = y + j;
                if (ny >= 0 && ny < Height)
                {
                    __m512 ImgPixel  = _mm512_load_ps(Tmp + ny * PaddedWidth + x);
                    __m512 KernelVal = _mm512_set1_ps(GaussianKernel_1D[KernelRadius + j]);
                    PixelVal         = _mm512_add_ps(PixelVal, _mm512_mul_ps(ImgPixel, KernelVal));
                    KernelSum        = _mm512_add_ps(KernelSum, KernelVal);
                }
            }
            __m512 InvKernelSum = _mm512_div_ps(_mm512_set1_ps(1.0f), KernelSum);
            _mm512_store_ps(Tmp + y * PaddedWidth + x, _mm512_mul_ps(PixelVal, InvKernelSum));
        }
        for (; x < Width; x++)
        {
            float PixelVal = 0.0f, KernelSum = 0.0f;
            for (int j = -KernelRadius; j <= KernelRadius; j++)
            {
                int ny = y + j;
                if (ny >= 0 && ny < Height)
                {
                    float ImgPixel  = Tmp[ny * PaddedWidth + x];
                    float KernelVal = GaussianKernel_1D[KernelRadius + j];
                    PixelVal += ImgPixel * KernelVal;
                    KernelSum += KernelVal;
                }
            }
            Tmp[y * PaddedWidth + x] = PixelVal / KernelSum;
        }
    };

#pragma omp parallel
    {
#pragma omp for schedule(guided) nowait
        for (int y = 0; y < Height; y++) { ProcessRow(y); }

#pragma omp for schedule(guided)
        for (int y = 0; y < Height; y++) { ProcessColumn(y); }
    }

    __m512 zero          = _mm512_set1_ps(0.0f);
    __m512 two_five_five = _mm512_set1_ps(255.0f);
#pragma omp parallel for
    for (int i = 0; i <= Width * Height - 16; i += 16)
    {
        __m512 Pixels      = _mm512_load_ps(Tmp + i);
        Pixels             = _mm512_max_ps(zero, _mm512_min_ps(Pixels, two_five_five));
        __m512i Pixels_i32 = _mm512_cvtps_epi32(Pixels);
        __m256i Pixels_i16 = _mm512_cvtepi32_epi16(Pixels_i32);
        __m128i Pixels_i8  = _mm256_cvtepi16_epi8(Pixels_i16);
        _mm_store_si128((__m128i*)(Output + i), Pixels_i8);
    }

    for (int i = (Width * Height / 16) * 16; i < Width * Height; i++)
        Output[i] = static_cast<uint8_t>(std::min(std::max(0.0f, Tmp[i]), 255.0f));

    _mm_free(Tmp);
}

void OpenMP::ComputeGradients(float* Gradients, uint8_t* GradDires, const uint8_t* BlurredImage, int Width, int Height)
{
    static const int8_t Gx[]   = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    static const int8_t Gy[]   = {1, 2, 1, 0, 0, 0, -1, -2, -1};
    static const int    Offset = 1;

#pragma omp parallel for
    for (int y = Offset; y < Height - Offset; y++)
    {
        int x = Offset;
        for (; x <= Width - 16; x += 16)
        {
            __m512 GradX = _mm512_setzero_ps();
            __m512 GradY = _mm512_setzero_ps();

            for (int ky = -Offset; ky <= Offset; ky++)
            {
                for (int kx = -Offset; kx <= Offset; kx++)
                {
                    int KernelIdx = (ky + Offset) * 3 + (kx + Offset);
                    int PixelIdx  = x + kx + (y + ky) * Width;

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
                                  (_mm512_cmp_ps_mask(Degrees, _mm512_set1_ps(-157.5), _CMP_LE_OS) |
                                      _mm512_cmp_ps_mask(Degrees, _mm512_set1_ps(157.5), _CMP_NLE_US));
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

        for (; x < Width - Offset; x++)
        {
            float GradX = 0.0f;
            float GradY = 0.0f;
            for (int ky = -Offset; ky <= Offset; ky++)
            {
                for (int kx = -Offset; kx <= Offset; kx++)
                {
                    int KernelIdx = (ky + Offset) * 3 + (kx + Offset);
                    int PixelIdx  = x + kx + (y + ky) * Width;
                    GradX += BlurredImage[PixelIdx] * Gx[KernelIdx];
                    GradY += BlurredImage[PixelIdx] * Gy[KernelIdx];
                }
            }
            Gradients[x + y * Width] = std::sqrt(GradX * GradX + GradY * GradY);
            float   Degree           = std::atan2(GradY, GradX) * (360.0 / (2.0 * M_PI));
            uint8_t Direction        = 0;
            if ((Degree <= 22.5 && Degree > -22.5) || (Degree <= -157.5 || Degree > 157.5))
                Direction = 1;
            else if ((Degree > 22.5 && Degree <= 67.5) || (Degree > -157.5 && Degree <= -112.5))
                Direction = 2;
            else if ((Degree > 67.5 && Degree <= 112.5) || (Degree > -112.5 && Degree <= -67.5))
                Direction = 3;
            else if ((Degree > 112.5 && Degree <= 157.5) || (Degree > -67.5 && Degree <= -22.5))
                Direction = 4;
            GradDires[x + y * Width] = Direction;
        }
    }
}

void OpenMP::ReduceNonMaximum(float* Magnitudes, float* Gradients, uint8_t* Direction, int Width, int Height) {}

void OpenMP::PerformDoubleThresholding(
    uint8_t* EdgedImg, float* Magnitudes, int HighThre, int LowThre, int Width, int Height)
{}

void OpenMP::PerformEdgeHysteresis(uint8_t* EdgedImg, uint8_t* InitialEdges, int Width, int Height) {}