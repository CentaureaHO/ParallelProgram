#include <immintrin.h>
#include <iostream>
#include <math.h>
#include <pthread.h>
#include <vector>
#include "AVX_Lib.h"
#include "ParmsDef.h"
#include "PThread.h"
#include "ThreadPool.h"

// 以下为无线程池版本

PThread::PThread(unsigned int TN) : ThreadNum(static_cast<int>(TN)) {}

PThread::~PThread() {}

PThread& PThread::GetInstance(unsigned int TN)
{
    static PThread Instance(TN);
    return Instance;
}

void PThread::PerformGaussianBlur(uint8_t* Output, const uint8_t* OriImg, int Width, int Height)
{
    int    PaddedWidth = (Width + 15) & ~15;
    float* Temp        = (float*)_mm_malloc(PaddedWidth * Height * sizeof(float), 64);

    auto ProcessRow = [&](int startY, int endY) {
        for (int y = startY; y < endY; ++y)
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
                _mm512_store_ps(Temp + y * PaddedWidth + x, _mm512_mul_ps(PixelVal, InvKernelSum));
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
                Temp[y * PaddedWidth + x] = PixelVal / KernelSum;
            }
        }
    };

    auto ProcessColumn = [&](int startY, int endY) {
        for (int y = startY; y < endY; ++y)
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
                        __m512 ImgPixel  = _mm512_load_ps(Temp + ny * PaddedWidth + x);
                        __m512 KernelVal = _mm512_set1_ps(GaussianKernel_1D[KernelRadius + j]);
                        PixelVal         = _mm512_add_ps(PixelVal, _mm512_mul_ps(ImgPixel, KernelVal));
                        KernelSum        = _mm512_add_ps(KernelSum, KernelVal);
                    }
                }
                __m512 InvKernelSum = _mm512_div_ps(_mm512_set1_ps(1.0f), KernelSum);
                _mm512_store_ps(Temp + y * PaddedWidth + x, _mm512_mul_ps(PixelVal, InvKernelSum));
            }
            for (; x < Width; x++)
            {
                float PixelVal = 0.0f, KernelSum = 0.0f;
                for (int j = -KernelRadius; j <= KernelRadius; j++)
                {
                    int ny = y + j;
                    if (ny >= 0 && ny < Height)
                    {
                        float ImgPixel  = Temp[ny * PaddedWidth + x];
                        float KernelVal = GaussianKernel_1D[KernelRadius + j];
                        PixelVal += ImgPixel * KernelVal;
                        KernelSum += KernelVal;
                    }
                }
                Temp[y * PaddedWidth + x] = PixelVal / KernelSum;
            }
        }
    };

    int                      RegionHeight = (Height + ThreadNum - 1) / ThreadNum;
    std::vector<std::thread> Threads;

    for (int i = 0; i < ThreadNum; i++)
    {
        int StartY = i * RegionHeight;
        int EndY   = std::min(StartY + RegionHeight, Height);
        Threads.emplace_back(ProcessRow, StartY, EndY);
    }
    for (auto& t : Threads) t.join();

    Threads.clear();

    for (int i = 0; i < ThreadNum; i++)
    {
        int StartY = i * RegionHeight + KernelRadius;
        int EndY   = std::min(StartY + RegionHeight - KernelRadius, Height - KernelRadius);
        Threads.emplace_back(ProcessColumn, StartY, EndY);
    }
    for (auto& t : Threads) t.join();

    auto ProcessBorders = [&](int startY, int endY) {
        if (startY < KernelRadius) { ProcessColumn(startY, std::min(startY + KernelRadius, Height)); }
        if (endY > Height - KernelRadius) { ProcessColumn(std::max(endY - KernelRadius, 0), endY); }
    };

    Threads.clear();

    for (int i = 0; i < ThreadNum; i++)
    {
        int startY = i * RegionHeight;
        int endY   = std::min(startY + RegionHeight, Height);
        Threads.emplace_back(ProcessBorders, startY, endY);
    }
    for (auto& t : Threads) t.join();

    auto FinalizeOutput = [&](int start, int end) {
        __m512 zero          = _mm512_set1_ps(0.0f);
        __m512 two_five_five = _mm512_set1_ps(255.0f);
        for (int i = start; i <= end - 16; i += 16)
        {
            __m512 Pixels      = _mm512_load_ps(Temp + i);
            Pixels             = _mm512_max_ps(zero, _mm512_min_ps(Pixels, two_five_five));
            __m512i Pixels_i32 = _mm512_cvtps_epi32(Pixels);
            __m256i Pixels_i16 = _mm512_cvtepi32_epi16(Pixels_i32);
            __m128i Pixels_i8  = _mm256_cvtepi16_epi8(Pixels_i16);
            _mm_store_si128((__m128i*)(Output + i), Pixels_i8);
        }
        for (int i = end - (end % 16); i < end; i++)
            Output[i] = static_cast<uint8_t>(std::min(std::max(0.0f, Temp[i]), 255.0f));
    };

    int PixelsPerThread = (Width * Height + ThreadNum - 1) / ThreadNum;
    Threads.clear();

    for (int i = 0; i < ThreadNum; i++)
    {
        int Start = i * PixelsPerThread;
        int End   = std::min(Start + PixelsPerThread, Width * Height);
        Threads.emplace_back(FinalizeOutput, Start, End);
    }
    for (auto& t : Threads) t.join();

    _mm_free(Temp);
}

void PThread::ComputeGradients(float* Gradients, uint8_t* GradDires, const uint8_t* BlurredImage, int Width, int Height)
{
    struct ThreadData
    {
        float*         Gradients;
        uint8_t*       GradDires;
        const uint8_t* BlurredImage;
        int            Width;
        int            Height;
        int            RFrom;
        int            REnd;
        const int8_t*  Gx;
        const int8_t*  Gy;
    };
    int                 RPT  = Height / ThreadNum;
    static const int8_t Gx[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    static const int8_t Gy[] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

    std::vector<std::thread> Threads;

    for (int i = 0; i < ThreadNum; i++)
    {
        int RFrom = i * RPT + 1;
        int REnd  = (i == ThreadNum - 1) ? (Height - 1) : (RFrom + RPT);

        ThreadData* ThData = new ThreadData{Gradients, GradDires, BlurredImage, Width, Height, RFrom, REnd, Gx, Gy};

        auto ThreadFunc = [](ThreadData* Data) {
            const int Offset = 1;

            for (int y = Data->RFrom; y < Data->REnd; y++)
            {
                int x = Offset;
                for (; x <= Data->Width - 16; x += 16)
                {
                    __m512 GradX = _mm512_setzero_ps();
                    __m512 GradY = _mm512_setzero_ps();

                    for (int ky = -Offset; ky <= Offset; ky++)
                    {
                        for (int kx = -Offset; kx <= Offset; kx++)
                        {
                            int KernelIdx = (ky + Offset) * 3 + (kx + Offset);
                            int PixelIdx  = x + (y * Data->Width) + kx + (ky * Data->Width);

                            if (x + 15 < Data->Width)
                            {
                                __m512i PixelValues = _mm512_cvtepu8_epi32(
                                    _mm_loadu_si128((const __m128i*)(Data->BlurredImage + PixelIdx)));
                                __m512i GxValue = _mm512_set1_epi32(Data->Gx[KernelIdx]);
                                __m512i GyValue = _mm512_set1_epi32(Data->Gy[KernelIdx]);

                                GradX = _mm512_add_ps(
                                    GradX, _mm512_mul_ps(_mm512_cvtepi32_ps(PixelValues), _mm512_cvtepi32_ps(GxValue)));
                                GradY = _mm512_add_ps(
                                    GradY, _mm512_mul_ps(_mm512_cvtepi32_ps(PixelValues), _mm512_cvtepi32_ps(GyValue)));
                            }
                        }
                    }

                    __m512 Magnitude =
                        _mm512_sqrt_ps(_mm512_add_ps(_mm512_mul_ps(GradX, GradX), _mm512_mul_ps(GradY, GradY)));
                    _mm512_storeu_ps(Data->Gradients + x + y * Data->Width, Magnitude);

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

                    _mm_storeu_si128(
                        (__m128i*)(Data->GradDires + x + y * Data->Width), _mm512_cvtsepi32_epi8(Directions));
                }

                if (x < Data->Width - Offset)
                {
                    for (; x < Data->Width - Offset; x++)
                    {
                        float GradX = 0.0f;
                        float GradY = 0.0f;
                        for (int ky = -Offset; ky <= Offset; ky++)
                        {
                            for (int kx = -Offset; kx <= Offset; kx++)
                            {
                                int KernelIdx = (ky + Offset) * 3 + (kx + Offset);
                                int PixelIdx  = x + kx + (y + ky) * Data->Width;
                                GradX += Data->BlurredImage[PixelIdx] * Data->Gx[KernelIdx];
                                GradY += Data->BlurredImage[PixelIdx] * Data->Gy[KernelIdx];
                            }
                        }
                        Data->Gradients[x + y * Data->Width] = std::sqrt(GradX * GradX + GradY * GradY);
                        float   Degree                       = std::atan2(GradY, GradX) * (360.0 / (2.0 * M_PI));
                        uint8_t Direction                    = 0;
                        if ((Degree <= 22.5 && Degree > -22.5) || (Degree <= -157.5 || Degree > 157.5))
                            Direction = 1;
                        else if ((Degree > 22.5 && Degree <= 67.5) || (Degree > -157.5 && Degree <= -112.5))
                            Direction = 2;
                        else if ((Degree > 67.5 && Degree <= 112.5) || (Degree > -112.5 && Degree <= -67.5))
                            Direction = 3;
                        else if ((Degree > 112.5 && Degree <= 157.5) || (Degree > -67.5 && Degree <= -22.5))
                            Direction = 4;
                        Data->GradDires[x + y * Data->Width] = Direction;
                    }
                }
            }

            delete Data;
        };

        Threads.emplace_back(ThreadFunc, ThData);
    }

    for (auto& t : Threads)
    {
        if (t.joinable()) t.join();
    }
}

void PThread::ReduceNonMaximum(float* Magnitudes, float* Gradients, uint8_t* Direction, int Width, int Height) {}

void PThread::PerformDoubleThresholding(
    uint8_t* EdgedImg, float* Magnitudes, int HighThre, int LowThre, int Width, int Height)
{}

void PThread::PerformEdgeHysteresis(uint8_t* EdgedImg, uint8_t* InitialEdges, int Width, int Height) {}

// 以下为使用线程池的版本，减少因线程创建和销毁带来的开销

PThreadWithPool::PThreadWithPool(unsigned int TN) : Pool(TN), ThreadNum(static_cast<int>(TN)) {}

PThreadWithPool::~PThreadWithPool() {}

PThreadWithPool& PThreadWithPool::GetInstance(unsigned int TN)
{
    static PThreadWithPool Instance(TN);
    return Instance;
}

void PThreadWithPool::PerformGaussianBlur(uint8_t* Output, const uint8_t* OriImg, int Width, int Height)
{
    int    PaddedWidth = (Width + 15) & ~15;
    float* Temp        = (float*)_mm_malloc(PaddedWidth * Height * sizeof(float), 64);

    auto ProcessRow = [&](int startY, int endY) {
        for (int y = startY; y < endY; ++y)
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
                _mm512_store_ps(Temp + y * PaddedWidth + x, _mm512_mul_ps(PixelVal, InvKernelSum));
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
                Temp[y * PaddedWidth + x] = PixelVal / KernelSum;
            }
        }
    };

    auto ProcessColumn = [&](int startY, int endY) {
        for (int y = startY; y < endY; ++y)
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
                        __m512 ImgPixel  = _mm512_load_ps(Temp + ny * PaddedWidth + x);
                        __m512 KernelVal = _mm512_set1_ps(GaussianKernel_1D[KernelRadius + j]);
                        PixelVal         = _mm512_add_ps(PixelVal, _mm512_mul_ps(ImgPixel, KernelVal));
                        KernelSum        = _mm512_add_ps(KernelSum, KernelVal);
                    }
                }
                __m512 InvKernelSum = _mm512_div_ps(_mm512_set1_ps(1.0f), KernelSum);
                _mm512_store_ps(Temp + y * PaddedWidth + x, _mm512_mul_ps(PixelVal, InvKernelSum));
            }
            for (; x < Width; x++)
            {
                float PixelVal = 0.0f, KernelSum = 0.0f;
                for (int j = -KernelRadius; j <= KernelRadius; j++)
                {
                    int ny = y + j;
                    if (ny >= 0 && ny < Height)
                    {
                        float ImgPixel  = Temp[ny * PaddedWidth + x];
                        float KernelVal = GaussianKernel_1D[KernelRadius + j];
                        PixelVal += ImgPixel * KernelVal;
                        KernelSum += KernelVal;
                    }
                }
                Temp[y * PaddedWidth + x] = PixelVal / KernelSum;
            }
        }
    };

    int RegionHeight = (Height + ThreadNum - 1) / ThreadNum;

    auto ProcessRegion = [&](int ThID) {
        int StartY = ThID * RegionHeight;
        int EndY   = std::min(StartY + RegionHeight, Height);

        ProcessRow(StartY, EndY);

        int InnerStartY = StartY + KernelRadius;
        int InnerEndY   = EndY - KernelRadius;
        ProcessColumn(InnerStartY, InnerEndY);
    };

    for (int i = 0; i < ThreadNum; i++) { Pool.EnQueue(ProcessRegion, i); }
    Pool.Sync();

    auto ProcessBorders = [&](int ThID) {
        int startY = ThID * RegionHeight;
        int endY   = std::min(startY + RegionHeight, Height);
        if (startY < KernelRadius) { ProcessColumn(startY, std::min(startY + KernelRadius, Height)); }
        if (endY > Height - KernelRadius) { ProcessColumn(std::max(endY - KernelRadius, 0), endY); }
    };

    for (int i = 0; i < ThreadNum; i++) { Pool.EnQueue(ProcessBorders, i); }
    Pool.Sync();

    auto FinalizeOutput = [&](int start, int end) {
        __m512 zero          = _mm512_set1_ps(0.0f);
        __m512 two_five_five = _mm512_set1_ps(255.0f);
        for (int i = start; i <= end - 16; i += 16)
        {
            __m512 Pixels      = _mm512_load_ps(Temp + i);
            Pixels             = _mm512_max_ps(zero, _mm512_min_ps(Pixels, two_five_five));
            __m512i Pixels_i32 = _mm512_cvtps_epi32(Pixels);
            __m256i Pixels_i16 = _mm512_cvtepi32_epi16(Pixels_i32);
            __m128i Pixels_i8  = _mm256_cvtepi16_epi8(Pixels_i16);
            _mm_store_si128((__m128i*)(Output + i), Pixels_i8);
        }
        for (int i = end - (end % 16); i < end; i++)
            Output[i] = static_cast<uint8_t>(std::min(std::max(0.0f, Temp[i]), 255.0f));
    };

    int PixelsPerThread = (Width * Height + ThreadNum - 1) / ThreadNum;
    for (int i = 0; i < ThreadNum; i++)
    {
        int Start = i * PixelsPerThread;
        int End   = std::min(Start + PixelsPerThread, Width * Height);
        Pool.EnQueue(FinalizeOutput, Start, End);
    }
    Pool.Sync();

    _mm_free(Temp);
}

void PThreadWithPool::ComputeGradients(
    float* Gradients, uint8_t* GradDires, const uint8_t* BlurredImage, int Width, int Height)
{
    const int8_t     Gx[]   = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    const int8_t     Gy[]   = {1, 2, 1, 0, 0, 0, -1, -2, -1};
    static const int Offset = 1;

    int RPT = Height / ThreadNum;

    for (int i = 0; i < ThreadNum; i++)
    {
        int RFrom = i * RPT + 1;
        int REnd  = (i == ThreadNum - 1) ? (Height - 1) : (RFrom + RPT);

        auto ThreadFunc = [Gradients, GradDires, BlurredImage, Width, Height, RFrom, REnd, Gx, Gy]() {
            for (int y = RFrom; y < REnd; y++)
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

                    __m512 Magnitude =
                        _mm512_sqrt_ps(_mm512_add_ps(_mm512_mul_ps(GradX, GradX), _mm512_mul_ps(GradY, GradY)));
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
                    else if ((Degree > 112.5 and Degree <= 157.5) || (Degree > -67.5 and Degree <= -22.5))
                        Direction = 4;
                    GradDires[x + y * Width] = Direction;
                }
            }
        };

        Pool.EnQueue(ThreadFunc);
    }

    Pool.Sync();
}

void PThreadWithPool::ReduceNonMaximum(float* Magnitudes, float* Gradients, uint8_t* Direction, int Width, int Height)
{}

void PThreadWithPool::PerformDoubleThresholding(
    uint8_t* EdgedImg, float* Magnitudes, int HighThre, int LowThre, int Width, int Height)
{}

void PThreadWithPool::PerformEdgeHysteresis(uint8_t* EdgedImg, uint8_t* InitialEdges, int Width, int Height) {}