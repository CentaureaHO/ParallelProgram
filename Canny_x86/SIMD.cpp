#include <immintrin.h>
#include <thread>
#include <vector>
#include <xmmintrin.h>
#include <bits/stdc++.h>
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
                            __m128 ImgPixel0 = _mm_set1_ps(static_cast<float>(Tmp[ImgIdx0]));
                            __m128 KernelVal = _mm_set1_ps(GaussianKernel_1D[KernelRadius + i]);
                            PixelVal0        = _mm_add_ps(PixelVal0, _mm_mul_ps(ImgPixel0, KernelVal));
                            if (nx0 + 1 < Width)
                                PixelVal1 = _mm_add_ps(PixelVal1,
                                    _mm_mul_ps(_mm_set1_ps(static_cast<float>(Tmp[ImgIdx0 + 1])), KernelVal));
                            if (nx0 + 2 < Width)
                                PixelVal2 = _mm_add_ps(PixelVal2,
                                    _mm_mul_ps(_mm_set1_ps(static_cast<float>(Tmp[ImgIdx0 + 2])), KernelVal));
                            if (nx0 + 3 < Width)
                                PixelVal3 = _mm_add_ps(PixelVal3,
                                    _mm_mul_ps(_mm_set1_ps(static_cast<float>(Tmp[ImgIdx0 + 3])), KernelVal));
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
                    _mm256_storeu_ps(&Tmp[y * Width + x], _mm256_mul_ps(PixelVal, InvKernelSum));
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
        });
    }
    for (auto& Th : Threads) Th.join();

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
                            int    ImgIdx    = y * Width + nx;
                            __m256 ImgPixel  = _mm256_loadu_ps(&Tmp[ImgIdx]);
                            __m256 KernelVal = _mm256_set1_ps(GaussianKernel_1D[KernelRadius + i]);
                            PixelVal         = _mm256_add_ps(PixelVal, _mm256_mul_ps(ImgPixel, KernelVal));
                            KernelSum        = _mm256_add_ps(KernelSum, KernelVal);
                        }
                    }

                    __m256 InvKernelSum = _mm256_div_ps(_mm256_set1_ps(1.0f), KernelSum);
                    _mm256_storeu_ps(&Tmp[y * Width + x], _mm256_mul_ps(PixelVal, InvKernelSum));
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
        });
    }
    for (auto& Th : Threads) Th.join();

    for (int i = 0; i < Width * Height; i++) Output[i] = static_cast<uint8_t>(Tmp[i]);
}

void AVX::A512::PerformGaussianBlur(uint8_t* Output, const uint8_t* OriImg, int Width, int Height)
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
                for (; x <= Width - 16; x += 16)
                {
                    __m512 PixelVal  = _mm512_setzero_ps();
                    __m512 KernelSum = _mm512_setzero_ps();

                    for (int i = -KernelRadius; i <= KernelRadius; i++)
                    {
                        int nx = x + i;
                        if (nx >= 0 && nx < Width)
                        {
                            int    ImgIdx = y * Width + nx;
                            __m512 ImgPixel =
                                _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)&OriImg[ImgIdx])));
                            __m512 KernelVal = _mm512_set1_ps(GaussianKernel_1D[KernelRadius + i]);
                            PixelVal         = _mm512_add_ps(PixelVal, _mm512_mul_ps(ImgPixel, KernelVal));
                            KernelSum        = _mm512_add_ps(KernelSum, KernelVal);
                        }
                    }

                    __m512 InvKernelSum = _mm512_div_ps(_mm512_set1_ps(1.0f), KernelSum);
                    _mm512_storeu_ps(&Tmp[y * Width + x], _mm512_mul_ps(PixelVal, InvKernelSum));
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
        });
    }
    for (auto& Th : Threads) Th.join();

    for (int t = 0; t < ThreadsNum; ++t)
    {
        Threads[t] = std::thread([&, t]() {
            int RowPerThread = Height / ThreadsNum;
            int Start        = t * RowPerThread;
            int End          = (t == ThreadsNum - 1) ? Height : Start + RowPerThread;
            for (int y = Start; y < End; y++)
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
                            int    ImgIdx = y * Width + nx;
                            __m512 ImgPixel =
                                _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)&OriImg[ImgIdx])));
                            __m512 KernelVal = _mm512_set1_ps(GaussianKernel_1D[KernelRadius + i]);
                            PixelVal         = _mm512_add_ps(PixelVal, _mm512_mul_ps(ImgPixel, KernelVal));
                            KernelSum        = _mm512_add_ps(KernelSum, KernelVal);
                        }
                    }

                    __m512 InvKernelSum = _mm512_div_ps(_mm512_set1_ps(1.0f), KernelSum);
                    _mm512_storeu_ps(&Tmp[y * Width + x], _mm512_mul_ps(PixelVal, InvKernelSum));
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
        });
    }
    for (auto& Th : Threads) Th.join();

    std::vector<float> OutputTmp(Width * Height, 0.0f);
    for (int t = 0; t < ThreadsNum; ++t)
    {
        Threads[t] = std::thread([&, t]() {
            int RowPerThread = Height / ThreadsNum;
            int Start        = t * RowPerThread;
            int End          = (t == ThreadsNum - 1) ? Height : Start + RowPerThread;
            for (int y = Start; y < End; y++)
            {
                int x = 0;
                for (; x <= Width - 16; x += 16)
                {
                    __m512 PixelVal  = _mm512_setzero_ps();
                    __m512 KernelSum = _mm512_setzero_ps();

                    for (int i = -KernelRadius; i <= KernelRadius; i++)
                    {
                        int ny = y + i;
                        if (ny >= 0 && ny < Height)
                        {
                            int    ImgIdx    = ny * Width + x;
                            __m512 ImgPixel  = _mm512_loadu_ps(&Tmp[ImgIdx]);
                            __m512 KernelVal = _mm512_set1_ps(GaussianKernel_1D[KernelRadius + i]);
                            PixelVal         = _mm512_add_ps(PixelVal, _mm512_mul_ps(ImgPixel, KernelVal));
                            KernelSum        = _mm512_add_ps(KernelSum, KernelVal);
                        }
                    }
                    __m512 InvKernelSum = _mm512_div_ps(_mm512_set1_ps(1.0f), KernelSum);
                    _mm512_storeu_ps(&OutputTmp[y * Width + x], _mm512_mul_ps(PixelVal, InvKernelSum));
                }

                for (; x < Width; x++)
                {
                    float PixelVal  = 0.0f;
                    float KernelSum = 0.0f;
                    for (int i = -KernelRadius; i <= KernelRadius; i++)
                    {
                        int ny = y + i;
                        if (ny >= 0 && ny < Height)
                        {
                            int   ImgIdx    = ny * Width + x;
                            float ImgPixel  = Tmp[ImgIdx];
                            float KernelVal = GaussianKernel_1D[KernelRadius + i];
                            PixelVal += ImgPixel * KernelVal;
                            KernelSum += KernelVal;
                        }
                    }
                    OutputTmp[y * Width + x] = PixelVal / KernelSum;
                }
            }
        });
    }
    for (auto& Th : Threads) Th.join();
    for (int i = 0; i < Width * Height; i++) Output[i] = static_cast<uint8_t>(OutputTmp[i]);
}