#include <immintrin.h>
#include <thread>
#include <vector>
#include <xmmintrin.h>
#include <bits/stdc++.h>
#include "GaussDef.h"
#include "SIMD.h"

const int KernelRadius = 1;

void SIMD::SSE::PerformGaussianBlur(uint8_t* Output, const uint8_t* OriImg, int Width, int Height)
{
    float* Tmp = static_cast<float*>(_mm_malloc(Width * Height * sizeof(float), 16));

    for (int y = 0; y < Height; y++)
    {
        int x = 0;
        for (; x <= Width - 4; x += 4)
        {
            __m128 PixelVal0 = _mm_setzero_ps();
            __m128 PixelVal1 = _mm_setzero_ps();
            __m128 PixelVal2 = _mm_setzero_ps();
            __m128 PixelVal3 = _mm_setzero_ps();
            __m128 KernelSum = _mm_setzero_ps();

            for (int i = -KernelRadius; i <= KernelRadius; i++)
            {
                int nx0 = x + i;
                if (nx0 >= 0 && nx0 < Width)
                {
                    int    ImgIdx0   = y * Width + nx0;
                    __m128 ImgPixel0 = _mm_set1_ps(static_cast<float>(OriImg[ImgIdx0]));
                    __m128 KernelVal = _mm_set1_ps(GaussianKernel_1D[KernelRadius + i]);
                    PixelVal0        = _mm_add_ps(PixelVal0, _mm_mul_ps(ImgPixel0, KernelVal));
                    if (nx0 + 1 < Width)
                        PixelVal1 = _mm_add_ps(
                            PixelVal1, _mm_mul_ps(_mm_set1_ps(static_cast<float>(OriImg[ImgIdx0 + 1])), KernelVal));
                    if (nx0 + 2 < Width)
                        PixelVal2 = _mm_add_ps(
                            PixelVal2, _mm_mul_ps(_mm_set1_ps(static_cast<float>(OriImg[ImgIdx0 + 2])), KernelVal));
                    if (nx0 + 3 < Width)
                        PixelVal3 = _mm_add_ps(
                            PixelVal3, _mm_mul_ps(_mm_set1_ps(static_cast<float>(OriImg[ImgIdx0 + 3])), KernelVal));
                    KernelSum = _mm_add_ps(KernelSum, KernelVal);
                }
            }

            __m128 InvKernelSum    = _mm_div_ps(_mm_set1_ps(1.0f), KernelSum);
            Tmp[y * Width + x]     = _mm_cvtss_f32(_mm_mul_ps(PixelVal0, InvKernelSum));
            Tmp[y * Width + x + 1] = _mm_cvtss_f32(_mm_mul_ps(PixelVal1, InvKernelSum));
            Tmp[y * Width + x + 2] = _mm_cvtss_f32(_mm_mul_ps(PixelVal2, InvKernelSum));
            Tmp[y * Width + x + 3] = _mm_cvtss_f32(_mm_mul_ps(PixelVal3, InvKernelSum));
        }

        for (; x < Width; x++)
        {
            __m128 PixelVal  = _mm_setzero_ps();
            __m128 KernelSum = _mm_setzero_ps();
            for (int i = -KernelRadius; i <= KernelRadius; i++)
            {
                int nx = x + i;
                if (nx >= 0 && nx < Width)
                {
                    int    ImgIdx    = y * Width + nx;
                    __m128 ImgPixel  = _mm_set1_ps(static_cast<float>(OriImg[ImgIdx]));
                    __m128 KernelVal = _mm_set1_ps(GaussianKernel_1D[KernelRadius + i]);
                    PixelVal         = _mm_add_ps(PixelVal, _mm_mul_ps(ImgPixel, KernelVal));
                    KernelSum        = _mm_add_ps(KernelSum, KernelVal);
                }
            }
            PixelVal           = _mm_div_ps(PixelVal, KernelSum);
            Tmp[y * Width + x] = _mm_cvtss_f32(_mm_shuffle_ps(PixelVal, PixelVal, _MM_SHUFFLE(0, 0, 0, 0)));
        }
    }

    for (int y = 0; y < Height; y++)
    {
        int x = 0;
        for (; x <= Width - 4; x += 4)
        {
            __m128 PixelVal0 = _mm_setzero_ps();
            __m128 PixelVal1 = _mm_setzero_ps();
            __m128 PixelVal2 = _mm_setzero_ps();
            __m128 PixelVal3 = _mm_setzero_ps();
            __m128 KernelSum = _mm_setzero_ps();

            for (int i = -KernelRadius; i <= KernelRadius; i++)
            {
                int nx0 = x + i;
                if (nx0 >= 0 && nx0 < Width)
                {
                    int    ImgIdx0   = y * Width + nx0;
                    __m128 ImgPixel0 = _mm_set1_ps(static_cast<float>(Tmp[ImgIdx0]));
                    __m128 KernelVal = _mm_set1_ps(GaussianKernel_1D[KernelRadius + i]);
                    PixelVal0        = _mm_add_ps(PixelVal0, _mm_mul_ps(ImgPixel0, KernelVal));
                    if (nx0 + 1 < Width)
                        PixelVal1 = _mm_add_ps(
                            PixelVal1, _mm_mul_ps(_mm_set1_ps(static_cast<float>(Tmp[ImgIdx0 + 1])), KernelVal));
                    if (nx0 + 2 < Width)
                        PixelVal2 = _mm_add_ps(
                            PixelVal2, _mm_mul_ps(_mm_set1_ps(static_cast<float>(Tmp[ImgIdx0 + 2])), KernelVal));
                    if (nx0 + 3 < Width)
                        PixelVal3 = _mm_add_ps(
                            PixelVal3, _mm_mul_ps(_mm_set1_ps(static_cast<float>(Tmp[ImgIdx0 + 3])), KernelVal));
                    KernelSum = _mm_add_ps(KernelSum, KernelVal);
                }
            }

            __m128 InvKernelSum       = _mm_div_ps(_mm_set1_ps(1.0f), KernelSum);
            Output[y * Width + x]     = _mm_cvtss_f32(_mm_mul_ps(PixelVal0, InvKernelSum));
            Output[y * Width + x + 1] = _mm_cvtss_f32(_mm_mul_ps(PixelVal1, InvKernelSum));
            Output[y * Width + x + 2] = _mm_cvtss_f32(_mm_mul_ps(PixelVal2, InvKernelSum));
            Output[y * Width + x + 3] = _mm_cvtss_f32(_mm_mul_ps(PixelVal3, InvKernelSum));
        }

        for (; x < Width; x++)
        {
            __m128 PixelVal  = _mm_setzero_ps();
            __m128 KernelSum = _mm_setzero_ps();
            for (int i = -KernelRadius; i <= KernelRadius; i++)
            {
                int nx = x + i;
                if (nx >= 0 && nx < Width)
                {
                    int    ImgIdx    = y * Width + nx;
                    __m128 ImgPixel  = _mm_set1_ps(static_cast<float>(Tmp[ImgIdx]));
                    __m128 KernelVal = _mm_set1_ps(GaussianKernel_1D[KernelRadius + i]);
                    PixelVal         = _mm_add_ps(PixelVal, _mm_mul_ps(ImgPixel, KernelVal));
                    KernelSum        = _mm_add_ps(KernelSum, KernelVal);
                }
            }
            PixelVal              = _mm_div_ps(PixelVal, KernelSum);
            Output[y * Width + x] = _mm_cvtss_f32(_mm_shuffle_ps(PixelVal, PixelVal, _MM_SHUFFLE(0, 0, 0, 0)));
        }
    }
    _mm_free(Tmp);
}

void SIMD::AVX::A256::PerformGaussianBlur(uint8_t* Output, const uint8_t* OriImg, int Width, int Height)
{
    float* Tmp = static_cast<float*>(_mm_malloc(Width * Height * sizeof(float), 32));

    for (int y = 0; y < Height; y++)
    {
        int x = 0;
        for (; x <= Width - 8; x += 8)
        {
            __m256 PixelVal  = _mm256_setzero_ps();
            __m256 KernelSum = _mm256_setzero_ps();

            for (int i = -KernelRadius; i <= KernelRadius; i++)
            {
                int nx = x + i;
                if (nx >= 0 && nx < Width)
                {
                    int    ImgIdx    = y * Width + nx;
                    __m256 ImgPixel  = _mm256_set_ps(static_cast<float>(OriImg[ImgIdx + 7]),
                        static_cast<float>(OriImg[ImgIdx + 6]),
                        static_cast<float>(OriImg[ImgIdx + 5]),
                        static_cast<float>(OriImg[ImgIdx + 4]),
                        static_cast<float>(OriImg[ImgIdx + 3]),
                        static_cast<float>(OriImg[ImgIdx + 2]),
                        static_cast<float>(OriImg[ImgIdx + 1]),
                        static_cast<float>(OriImg[ImgIdx]));
                    __m256 KernelVal = _mm256_set1_ps(GaussianKernel_1D[KernelRadius + i]);
                    PixelVal         = _mm256_add_ps(PixelVal, _mm256_mul_ps(ImgPixel, KernelVal));
                    KernelSum        = _mm256_add_ps(KernelSum, KernelVal);
                }
            }

            __m256 InvKernelSum = _mm256_div_ps(_mm256_set1_ps(1.0f), KernelSum);
            _mm256_store_ps(&Tmp[y * Width + x], _mm256_mul_ps(PixelVal, InvKernelSum));
        }

        for (; x < Width; x++)
        {
            float PixelVal  = 0.0f;
            float KernelSum = 0.0f;
            for (int i = -KernelRadius; i <= KernelRadius; i++)
            {
                int nx = x + i;
                if (nx >= 0 && nx < Width)
                {
                    int   ImgIdx    = y * Width + nx;
                    float ImgPixel  = static_cast<float>(OriImg[ImgIdx]);
                    float KernelVal = GaussianKernel_1D[KernelRadius + i];
                    PixelVal += ImgPixel * KernelVal;
                    KernelSum += KernelVal;
                }
            }
            Tmp[y * Width + x] = PixelVal / KernelSum;
        }
    }

    for (int y = 0; y < Height; y++)
    {
        int x = 0;
        for (; x <= Width - 8; x += 8)
        {
            __m256 PixelVal  = _mm256_setzero_ps();
            __m256 KernelSum = _mm256_setzero_ps();

            for (int i = -KernelRadius; i <= KernelRadius; i++)
            {
                int nx = x + i;
                if (nx >= 0 && nx < Width)
                {
                    int    ImgIdx    = y * Width + nx;
                    __m256 ImgPixel  = _mm256_loadu_ps(&Tmp[ImgIdx]);
                    __m256 KernelVal = _mm256_set1_ps(GaussianKernel_1D[KernelRadius + i]);
                    PixelVal         = _mm256_add_ps(PixelVal, _mm256_mul_ps(ImgPixel, KernelVal));
                    KernelSum        = _mm256_add_ps(KernelSum, KernelVal);
                }
            }

            __m256 InvKernelSum = _mm256_div_ps(_mm256_set1_ps(1.0f), KernelSum);
            _mm256_store_ps(&Tmp[y * Width + x], _mm256_mul_ps(PixelVal, InvKernelSum));
        }

        for (; x < Width; x++)
        {
            float PixelVal  = 0.0f;
            float KernelSum = 0.0f;
            for (int i = -KernelRadius; i <= KernelRadius; i++)
            {
                int nx = x + i;
                if (nx >= 0 && nx < Width)
                {
                    int   ImgIdx    = y * Width + nx;
                    float ImgPixel  = Tmp[ImgIdx];
                    float KernelVal = GaussianKernel_1D[KernelRadius + i];
                    PixelVal += ImgPixel * KernelVal;
                    KernelSum += KernelVal;
                }
            }
            Tmp[y * Width + x] = PixelVal / KernelSum;
        }
    }

    for (int i = 0; i < Width * Height; i++) Output[i] = static_cast<uint8_t>(Tmp[i]);
    _mm_free(Tmp);
}

void SIMD::AVX::A512::PerformGaussianBlur(uint8_t* Output, const uint8_t* OriImg, int Width, int Height)
{
    float* Tmp = (float*)_mm_malloc(Width * Height * sizeof(float), 64);

    for (int y = 0; y < Height; y++)
    {
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
            _mm512_store_ps(Tmp + y * Width + x, _mm512_mul_ps(PixelVal, InvKernelSum));
        }
        if (x < Width)
        {
            for (; x < Width; x++)
            {
                float PixelVal  = 0.0f;
                float KernelSum = 0.0f;
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
                Tmp[y * Width + x] = PixelVal / KernelSum;
            }
        }
    }

    for (int y = 0; y < Height; y++)
    {
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
                    __m512 ImgPixel  = _mm512_load_ps(Tmp + ny * Width + x);
                    __m512 KernelVal = _mm512_set1_ps(GaussianKernel_1D[KernelRadius + j]);
                    PixelVal         = _mm512_add_ps(PixelVal, _mm512_mul_ps(ImgPixel, KernelVal));
                    KernelSum        = _mm512_add_ps(KernelSum, KernelVal);
                }
            }
            __m512 InvKernelSum = _mm512_div_ps(_mm512_set1_ps(1.0f), KernelSum);
            _mm512_store_ps(Tmp + y * Width + x, _mm512_mul_ps(PixelVal, InvKernelSum));
        }
        if (x < Width)
        {
            for (; x < Width; x++)
            {
                float PixelVal  = 0.0f;
                float KernelSum = 0.0f;
                for (int j = -KernelRadius; j <= KernelRadius; j++)
                {
                    int ny = y + j;
                    if (ny >= 0 && ny < Height)
                    {
                        float ImgPixel  = Tmp[ny * Width + x];
                        float KernelVal = GaussianKernel_1D[KernelRadius + j];
                        PixelVal += ImgPixel * KernelVal;
                        KernelSum += KernelVal;
                    }
                }
                Tmp[y * Width + x] = PixelVal / KernelSum;
            }
        }
    }

    for (int i = 0; i < Width * Height; i++) Output[i] = static_cast<uint8_t>(std::min(std::max(0.0f, Tmp[i]), 255.0f));
    _mm_free(Tmp);
}

void SIMD::AVX::A512::ComputeGradients(
    float* Gradients, uint8_t* GradDires, const uint8_t* BlurredImage, int Width, int Height)
{
    const int8_t Gx[9]  = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    const int8_t Gy[9]  = {1, 2, 1, 0, 0, 0, -1, -2, -1};
    const int    Offset = 1;

    for (int y = Offset; y < Height - Offset; y++)
    {
        for (int x = Offset; x < Width - Offset; x += 16)
        {
            __m512 gradX = _mm512_setzero_ps();
            __m512 gradY = _mm512_setzero_ps();

            int PixelIdx = x + (y * Width);

            for (int ky = -Offset; ky <= Offset; ky++)
            {
                for (int kx = -Offset; kx <= Offset; kx++)
                {
                    int kernelIndex = (ky + Offset) * 3 + (kx + Offset);

                    __m512 kernelValueX = _mm512_set1_ps(static_cast<float>(Gx[kernelIndex]));
                    __m512 kernelValueY = _mm512_set1_ps(static_cast<float>(Gy[kernelIndex]));

                    __m512i pixels = _mm512_cvtepu8_epi32(
                        _mm_loadu_si128((__m128i*)(BlurredImage + PixelIdx + (kx + (ky * Width)))));
                    __m512 pixels_f = _mm512_cvtepi32_ps(pixels);

                    gradX = _mm512_fmadd_ps(pixels_f, kernelValueX, gradX);
                    gradY = _mm512_fmadd_ps(pixels_f, kernelValueY, gradY);
                }
            }

            __m512 gradMagnitude =
                _mm512_sqrt_ps(_mm512_add_ps(_mm512_mul_ps(gradX, gradX), _mm512_mul_ps(gradY, gradY)));
            _mm512_storeu_ps(Gradients + PixelIdx, gradMagnitude);

            for (int i = 0; i < 16; ++i)
            {
                float   angle = std::atan2(gradY[i], gradX[i]) * (360.0 / (2.0 * M_PI));
                uint8_t Dire  = 0;
                if ((angle <= 22.5 && angle >= -22.5) || (angle <= -157.5) || (angle >= 157.5))
                    Dire = 1;
                else if ((angle > 22.5 && angle <= 67.5) || (angle > -157.5 && angle <= -112.5))
                    Dire = 2;
                else if ((angle > 67.5 && angle <= 112.5) || (angle >= -112.5 && angle < -67.5))
                    Dire = 3;
                else if ((angle >= -67.5 && angle < -22.5) || (angle > 112.5 && angle < 157.5))
                    Dire = 4;

                GradDires[PixelIdx + i] = Dire;
            }
        }
    }
}