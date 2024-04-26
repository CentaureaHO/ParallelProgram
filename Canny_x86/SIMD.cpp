#include <immintrin.h>
#include <thread>
#include <vector>
#include <xmmintrin.h>
#include "GaussDef.h"
#include "SIMD.h"

const int KernelRadius = 1;

void SSE::PerformGaussianBlur(uint8_t* Output, const uint8_t* OriImg, int Width, int Height)
{
    std::vector<float>       Tmp(Width * Height, 0.0f);
    int                      ThreadsNum = std::thread::hardware_concurrency();
    std::vector<std::thread> Threads(ThreadsNum);

    for (int t = 0; t < ThreadsNum; ++t)
    {
        Threads[t] = std::thread([&, t]() {
            int RowPerThread = Height / ThreadsNum;
            int Start        = t * RowPerThread;
            int End          = (t == ThreadsNum - 1) ? Height : Start + RowPerThread;
            for (int y = Start; y < End; y++)
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
                                PixelVal1 = _mm_add_ps(PixelVal1,
                                    _mm_mul_ps(_mm_set1_ps(static_cast<float>(OriImg[ImgIdx0 + 1])), KernelVal));
                            if (nx0 + 2 < Width)
                                PixelVal2 = _mm_add_ps(PixelVal2,
                                    _mm_mul_ps(_mm_set1_ps(static_cast<float>(OriImg[ImgIdx0 + 2])), KernelVal));
                            if (nx0 + 3 < Width)
                                PixelVal3 = _mm_add_ps(PixelVal3,
                                    _mm_mul_ps(_mm_set1_ps(static_cast<float>(OriImg[ImgIdx0 + 3])), KernelVal));
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
        });
    }
    for (auto& Th : Threads) Th.join();

    for (int t = 0; t < ThreadsNum; ++t)
    {
        Threads[t] = std::thread([&, t]() {
            int ColPerThread = Width / ThreadsNum;
            int Start        = t * ColPerThread;
            int End          = (t == ThreadsNum - 1) ? Width : Start + ColPerThread;
            for (int x = Start; x < End; x++)
            {
                int y = 0;
                for (; y <= Height - 4; y += 4)
                {
                    __m128 PixelVal0 = _mm_setzero_ps();
                    __m128 PixelVal1 = _mm_setzero_ps();
                    __m128 PixelVal2 = _mm_setzero_ps();
                    __m128 PixelVal3 = _mm_setzero_ps();
                    __m128 KernelSum = _mm_setzero_ps();

                    for (int i = -KernelRadius; i <= KernelRadius; i++)
                    {
                        int ny0 = y + i;
                        if (ny0 >= 0 && ny0 < Height)
                        {
                            int    ImgIdx0   = ny0 * Width + x;
                            __m128 ImgPixel0 = _mm_set1_ps(Tmp[ImgIdx0]);
                            __m128 KernelVal = _mm_set1_ps(GaussianKernel_1D[KernelRadius + i]);
                            PixelVal0        = _mm_add_ps(PixelVal0, _mm_mul_ps(ImgPixel0, KernelVal));
                            if (ny0 + 1 < Height)
                                PixelVal1 =
                                    _mm_add_ps(PixelVal1, _mm_mul_ps(_mm_set1_ps(Tmp[ImgIdx0 + Width]), KernelVal));
                            if (ny0 + 2 < Height)
                                PixelVal2 =
                                    _mm_add_ps(PixelVal2, _mm_mul_ps(_mm_set1_ps(Tmp[ImgIdx0 + 2 * Width]), KernelVal));
                            if (ny0 + 3 < Height)
                                PixelVal3 =
                                    _mm_add_ps(PixelVal3, _mm_mul_ps(_mm_set1_ps(Tmp[ImgIdx0 + 3 * Width]), KernelVal));
                            KernelSum = _mm_add_ps(KernelSum, KernelVal);
                        }
                    }

                    __m128 InvKernelSum   = _mm_div_ps(_mm_set1_ps(1.0f), KernelSum);
                    Output[y * Width + x] = (uint8_t)std::min(
                        std::max((int)(_mm_cvtss_f32(_mm_mul_ps(PixelVal0, InvKernelSum)) + 0.5f), 0), 255);
                    Output[(y + 1) * Width + x] = (uint8_t)std::min(
                        std::max((int)(_mm_cvtss_f32(_mm_mul_ps(PixelVal1, InvKernelSum)) + 0.5f), 0), 255);
                    Output[(y + 2) * Width + x] = (uint8_t)std::min(
                        std::max((int)(_mm_cvtss_f32(_mm_mul_ps(PixelVal2, InvKernelSum)) + 0.5f), 0), 255);
                    Output[(y + 3) * Width + x] = (uint8_t)std::min(
                        std::max((int)(_mm_cvtss_f32(_mm_mul_ps(PixelVal3, InvKernelSum)) + 0.5f), 0), 255);
                }

                for (; y < Height; y++)
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
                    PixelVal              = _mm_div_ps(PixelVal, KernelSum);
                    Output[y * Width + x] = (uint8_t)std::min(
                        std::max(
                            (int)(_mm_cvtss_f32(_mm_shuffle_ps(PixelVal, PixelVal, _MM_SHUFFLE(0, 0, 0, 0))) + 0.5f),
                            0),
                        255);
                }
            }
        });
    }
    for (auto& Th : Threads) Th.join();
}

void AVX::A256::PerformGaussianBlur(uint8_t* Output, const uint8_t* OriImg, int Width, int Height)
{
    std::vector<float>       Tmp(Width * Height, 0.0f);
    int                      ThreadsNum = std::thread::hardware_concurrency();
    std::vector<std::thread> Threads(ThreadsNum);

    for (int t = 0; t < ThreadsNum; ++t)
    {
        Threads[t] = std::thread([&, t]() {
            int RowPerThread = Height / ThreadsNum;
            int Start        = t * RowPerThread;
            int End          = (t == ThreadsNum - 1) ? Height : Start + RowPerThread;
            for (int y = Start; y < End; y++)
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
                            __m256 ImgPixel = _mm256_set1_ps(static_cast<float>(OriImg[y * Width + nx]));
                            __m256 Kernel   = _mm256_set1_ps(GaussianKernel_1D[KernelRadius + i]);
                            PixelVal        = _mm256_add_ps(PixelVal, _mm256_mul_ps(ImgPixel, Kernel));
                            KernelSum       = _mm256_add_ps(KernelSum, Kernel);
                        }
                    }

                    PixelVal          = _mm256_div_ps(PixelVal, KernelSum);
                    __m256i Pixels_u8 = _mm256_cvtps_epi32(
                        _mm256_min_ps(_mm256_max_ps(PixelVal, _mm256_setzero_ps()), _mm256_set1_ps(255.0f)));
                    _mm_storeu_si128((__m128i*)&Output[y * Width + x], _mm256_extractf128_si256(Pixels_u8, 0));
                    _mm_storeu_si128((__m128i*)&Output[y * Width + x + 4], _mm256_extractf128_si256(Pixels_u8, 1));
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
                            float ImgPixel = static_cast<float>(OriImg[y * Width + nx]);
                            float Kernel   = GaussianKernel_1D[KernelRadius + i];
                            PixelVal += ImgPixel * Kernel;
                            KernelSum += Kernel;
                        }
                    }
                    Output[y * Width + x] =
                        static_cast<uint8_t>(std::min(std::max(int(PixelVal / KernelSum + 0.5f), 0), 255));
                }
            }
        });
    }
    for (auto& Th : Threads) Th.join();

    for (int t = 0; t < ThreadsNum; ++t)
    {
        Threads[t] = std::thread([&, t]() {
            int ColPerThread = Width / ThreadsNum;
            int Start        = t * ColPerThread;
            int End          = (t == ThreadsNum - 1) ? Width : Start + ColPerThread;
            for (int x = Start; x < End; x++)
            {
                int y = 0;
                for (; y <= Height - 8; y += 8)
                {
                    __m256 PixelVal  = _mm256_setzero_ps();
                    __m256 KernelSum = _mm256_setzero_ps();
                    for (int i = -KernelRadius; i <= KernelRadius; i++)
                    {
                        int ny = y + i;
                        if (ny >= 0 && ny < Height)
                        {
                            __m256 ImgPixel = _mm256_set1_ps(Tmp[ny * Width + x]);
                            __m256 Kernel   = _mm256_set1_ps(GaussianKernel_1D[KernelRadius + i]);
                            PixelVal        = _mm256_add_ps(PixelVal, _mm256_mul_ps(ImgPixel, Kernel));
                            KernelSum       = _mm256_add_ps(KernelSum, Kernel);
                        }
                    }
                    PixelVal          = _mm256_div_ps(PixelVal, KernelSum);
                    __m256i Pixels_u8 = _mm256_cvtps_epi32(
                        _mm256_min_ps(_mm256_max_ps(PixelVal, _mm256_setzero_ps()), _mm256_set1_ps(255.0f)));
                    _mm_storeu_si128((__m128i*)&Output[y * Width + x], _mm256_extractf128_si256(Pixels_u8, 0));
                    _mm_storeu_si128((__m128i*)&Output[y * Width + x + 4], _mm256_extractf128_si256(Pixels_u8, 1));
                }
                for (; y < Height; y++)
                {
                    float PixelVal  = 0.0f;
                    float KernelSum = 0.0f;
                    for (int i = -KernelRadius; i <= KernelRadius; i++)
                    {
                        int ny = y + i;
                        if (ny >= 0 && ny < Height)
                        {
                            float ImgPixel = Tmp[ny * Width + x];
                            float Kernel   = GaussianKernel_1D[KernelRadius + i];
                            PixelVal += ImgPixel * Kernel;
                            KernelSum += Kernel;
                        }
                    }
                    Output[y * Width + x] =
                        static_cast<uint8_t>(std::min(std::max(int(PixelVal / KernelSum + 0.5f), 0), 255));
                }
            }
        });
    }
    for (auto& Th : Threads) Th.join();
}
