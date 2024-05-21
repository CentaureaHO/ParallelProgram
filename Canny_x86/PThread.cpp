#include <cstring>
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
                for (; x < Offset + (16 - Offset % 16); x++)
                {
                    if (x >= Data->Width - Offset) break;
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

                for (; x <= Data->Width - 16 - Offset; x += 16)
                {
                    __m512 GradX = _mm512_setzero_ps();
                    __m512 GradY = _mm512_setzero_ps();

                    for (int ky = -Offset; ky <= Offset; ky++)
                    {
                        for (int kx = -Offset; kx <= Offset; kx++)
                        {
                            int KernelIdx = (ky + Offset) * 3 + (kx + Offset);
                            int PixelIdx  = x + (y * Data->Width) + kx + (ky * Data->Width);

                            __m512i PixelValues =
                                _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(Data->BlurredImage + PixelIdx)));
                            __m512i GxValue = _mm512_set1_epi32(Data->Gx[KernelIdx]);
                            __m512i GyValue = _mm512_set1_epi32(Data->Gy[KernelIdx]);

                            GradX = _mm512_add_ps(
                                GradX, _mm512_mul_ps(_mm512_cvtepi32_ps(PixelValues), _mm512_cvtepi32_ps(GxValue)));
                            GradY = _mm512_add_ps(
                                GradY, _mm512_mul_ps(_mm512_cvtepi32_ps(PixelValues), _mm512_cvtepi32_ps(GyValue)));
                        }
                    }

                    __m512 Magnitude =
                        _mm512_sqrt_ps(_mm512_add_ps(_mm512_mul_ps(GradX, GradX), _mm512_mul_ps(GradY, GradY)));
                    _mm512_store_ps(Data->Gradients + x + y * Data->Width, Magnitude);

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

                    _mm_store_si128(
                        (__m128i*)(Data->GradDires + x + y * Data->Width), _mm512_cvtsepi32_epi8(Directions));
                }

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

            delete Data;
        };

        Threads.emplace_back(ThreadFunc, ThData);
    }

    for (auto& t : Threads)
        if (t.joinable()) t.join();
}

void PThread::ReduceNonMaximum(float* Magnitudes, float* Gradients, uint8_t* Direction, int Width, int Height)
{
    struct ThreadData
    {
        float*   Magnitudes;
        float*   Gradients;
        uint8_t* Direction;
        int      Width;
        int      Height;
        int      StartY;
        int      EndY;
    };
    int                     rowsPerThread = (Height - 2) / ThreadNum;
    std::vector<pthread_t>  threads(ThreadNum);
    std::vector<ThreadData> threadData(ThreadNum);

    auto Reduce = [](void* arg) -> void* {
        ThreadData* data       = static_cast<ThreadData*>(arg);
        float*      Magnitudes = data->Magnitudes;
        float*      Gradients  = data->Gradients;
        uint8_t*    Direction  = data->Direction;
        int         Width      = data->Width;
        int         StartY     = data->StartY;
        int         EndY       = data->EndY;

        memcpy(Magnitudes + StartY * Width, Gradients + StartY * Width, (EndY - StartY) * Width * sizeof(float));

        __m512i Dir1 = _mm512_set1_epi8(1);
        __m512i Dir2 = _mm512_set1_epi8(2);
        __m512i Dir3 = _mm512_set1_epi8(3);
        __m512i Dir4 = _mm512_set1_epi8(4);

        for (int y = StartY; y < EndY; y++)
        {
            int x = 1;
            for (; x <= Width - 17; x += 16)
            {
                int Pos = x + (y * Width);

                __m512i Directions = _mm512_loadu_si512((__m512i*)&Direction[Pos]);
                __m512  Grads      = _mm512_loadu_ps(&Gradients[Pos]);
                __m512  Magn       = Grads;

                __mmask16 Mask1 = _mm512_cmpeq_epi8_mask(Directions, Dir1);
                __mmask16 Mask2 = _mm512_cmpeq_epi8_mask(Directions, Dir2);
                __mmask16 Mask3 = _mm512_cmpeq_epi8_mask(Directions, Dir3);
                __mmask16 Mask4 = _mm512_cmpeq_epi8_mask(Directions, Dir4);

                __m512 GradsLeft        = _mm512_loadu_ps(&Gradients[Pos - 1]);
                __m512 GradsRight       = _mm512_loadu_ps(&Gradients[Pos + 1]);
                __m512 GradsTopLeft     = _mm512_loadu_ps(&Gradients[Pos - (Width + 1)]);
                __m512 GradsTopRight    = _mm512_loadu_ps(&Gradients[Pos + (Width + 1)]);
                __m512 GradsTop         = _mm512_loadu_ps(&Gradients[Pos - Width]);
                __m512 GradsBottom      = _mm512_loadu_ps(&Gradients[Pos + Width]);
                __m512 GradsBottomLeft  = _mm512_loadu_ps(&Gradients[Pos - (Width - 1)]);
                __m512 GradsBottomRight = _mm512_loadu_ps(&Gradients[Pos + (Width - 1)]);

                __mmask16 MaskDir1 = _mm512_kor(_mm512_cmp_ps_mask(GradsLeft, Grads, _CMP_GE_OQ),
                    _mm512_cmp_ps_mask(GradsRight, Grads, _CMP_GT_OQ));
                __mmask16 MaskDir2 = _mm512_kor(_mm512_cmp_ps_mask(GradsTopLeft, Grads, _CMP_GE_OQ),
                    _mm512_cmp_ps_mask(GradsBottomRight, Grads, _CMP_GT_OQ));
                __mmask16 MaskDir3 = _mm512_kor(_mm512_cmp_ps_mask(GradsTop, Grads, _CMP_GE_OQ),
                    _mm512_cmp_ps_mask(GradsBottom, Grads, _CMP_GT_OQ));
                __mmask16 MaskDir4 = _mm512_kor(_mm512_cmp_ps_mask(GradsTopRight, Grads, _CMP_GE_OQ),
                    _mm512_cmp_ps_mask(GradsBottomLeft, Grads, _CMP_GT_OQ));

                __mmask16 FinalMask1 = _mm512_kand(Mask1, MaskDir1);
                __mmask16 FinalMask2 = _mm512_kand(Mask2, MaskDir2);
                __mmask16 FinalMask3 = _mm512_kand(Mask3, MaskDir3);
                __mmask16 FinalMask4 = _mm512_kand(Mask4, MaskDir4);

                __mmask16 FinalMask =
                    _mm512_kor(_mm512_kor(FinalMask1, FinalMask2), _mm512_kor(FinalMask3, FinalMask4));

                Magn = _mm512_mask_blend_ps(FinalMask, Magn, _mm512_set1_ps(0.0f));
                _mm512_storeu_ps(&Magnitudes[Pos], Magn);
            }

            for (; x < Width - 1; x++)
            {
                int Pos = x + (y * Width);
                switch (Direction[Pos])
                {
                    case 1:
                        if (Gradients[Pos - 1] >= Gradients[Pos] || Gradients[Pos + 1] > Gradients[Pos])
                            Magnitudes[Pos] = 0;
                        break;
                    case 2:
                        if (Gradients[Pos - (Width - 1)] >= Gradients[Pos] ||
                            Gradients[Pos + (Width - 1)] > Gradients[Pos])
                            Magnitudes[Pos] = 0;
                        break;
                    case 3:
                        if (Gradients[Pos - Width] >= Gradients[Pos] || Gradients[Pos + Width] > Gradients[Pos])
                            Magnitudes[Pos] = 0;
                        break;
                    case 4:
                        if (Gradients[Pos - (Width + 1)] >= Gradients[Pos] ||
                            Gradients[Pos + (Width + 1)] > Gradients[Pos])
                            Magnitudes[Pos] = 0;
                        break;
                    default: Magnitudes[Pos] = 0; break;
                }
            }
        }
        return nullptr;
    };

    for (int i = 0; i < ThreadNum; ++i)
    {
        int startY    = 1 + i * rowsPerThread;
        int endY      = (i == ThreadNum - 1) ? Height - 1 : startY + rowsPerThread;
        threadData[i] = {Magnitudes, Gradients, Direction, Width, Height, startY, endY};
        pthread_create(&threads[i], nullptr, Reduce, &threadData[i]);
    }

    for (int i = 0; i < ThreadNum; ++i) pthread_join(threads[i], nullptr);
}

void PThread::PerformDoubleThresholding(
    uint8_t* EdgedImg, float* Magnitudes, int HighThre, int LowThre, int Width, int Height)
{
    struct ThreadData
    {
        uint8_t* EdgedImg;
        float*   Magnitudes;
        int      HighThre;
        int      LowThre;
        int      Width;
        int      RFrom;
        int      REnd;
    };

    int RPT = Height / ThreadNum;

    std::vector<pthread_t>  threads(ThreadNum);
    std::vector<ThreadData> threadData(ThreadNum);

    for (int i = 0; i < ThreadNum; ++i)
    {
        int RFrom = i * RPT;
        int REnd  = (i == ThreadNum - 1) ? Height : (RFrom + RPT);

        threadData[i] = {EdgedImg, Magnitudes, HighThre, LowThre, Width, RFrom, REnd};

        auto ThreadFunc = [](void* arg) -> void* {
            ThreadData* data = static_cast<ThreadData*>(arg);

            __m512         HighThreshold = _mm512_set1_ps(static_cast<float>(data->HighThre));
            __m512         LowThreshold  = _mm512_set1_ps(static_cast<float>(data->LowThre));
            static __m512i MaxVal        = _mm512_set1_epi32(255);
            static __m512i MidVal        = _mm512_set1_epi32(100);
            static __m512i Zero          = _mm512_set1_epi32(0);

            for (int y = data->RFrom; y < data->REnd; ++y)
            {
                for (int x = 0; x < data->Width; x += 16)
                {
                    int    PixelIdx = x + (y * data->Width);
                    __m512 Mag      = _mm512_load_ps(&data->Magnitudes[PixelIdx]);

                    __mmask16 HighMask = _mm512_cmp_ps_mask(Mag, HighThreshold, _CMP_GT_OQ);
                    __mmask16 LowMask  = _mm512_cmp_ps_mask(Mag, LowThreshold, _CMP_GT_OQ);

                    _mm_store_si128((__m128i*)&data->EdgedImg[PixelIdx],
                        _mm256_cvtepi16_epi8(_mm512_cvtepi32_epi16(_mm512_mask_blend_epi32(
                            HighMask, _mm512_mask_blend_epi32(LowMask, Zero, MidVal), MaxVal))));
                }
            }

            pthread_exit(nullptr);
            return nullptr;
        };

        pthread_create(&threads[i], nullptr, ThreadFunc, &threadData[i]);
    }

    for (int i = 0; i < ThreadNum; ++i) pthread_join(threads[i], nullptr);
}

void PThread::PerformEdgeHysteresis(uint8_t* EdgedImg, uint8_t* InitialEdges, int Width, int Height)
{
    struct ThreadData
    {
        uint8_t* EdgedImg;
        uint8_t* InitialEdges;
        int      Width;
        int      Height;
        int      StartY;
        int      EndY;
    };

    static __m512i LowThreshold  = _mm512_set1_epi8(static_cast<uint8_t>(100));
    static __m512i HighThreshold = _mm512_set1_epi8(static_cast<uint8_t>(255));

    memcpy(EdgedImg, InitialEdges, Width * Height * sizeof(uint8_t));

    auto ThreadFunc = [](void* arg) -> void* {
        ThreadData* data         = static_cast<ThreadData*>(arg);
        uint8_t*    EdgedImg     = data->EdgedImg;
        uint8_t*    InitialEdges = data->InitialEdges;
        int         Width        = data->Width;
        int         StartY       = data->StartY;
        int         EndY         = data->EndY;

        for (int y = StartY; y < EndY; y++)
        {
            int x = 1;
            for (; x <= Width - 65; x += 64)
            {
                __m512i   CurPixels = _mm512_loadu_si512((__m512i*)&InitialEdges[x + y * Width]);
                __mmask64 Has100    = _mm512_cmpeq_epi8_mask(CurPixels, LowThreshold);

                if (Has100)
                {
                    __m512i Neighbors[8];
                    Neighbors[0] = _mm512_loadu_si512((__m512i*)&InitialEdges[x - 1 + (y - 1) * Width]);
                    Neighbors[1] = _mm512_loadu_si512((__m512i*)&InitialEdges[x + (y - 1) * Width]);
                    Neighbors[2] = _mm512_loadu_si512((__m512i*)&InitialEdges[x + 1 + (y - 1) * Width]);
                    Neighbors[3] = _mm512_loadu_si512((__m512i*)&InitialEdges[x - 1 + y * Width]);
                    Neighbors[4] = _mm512_loadu_si512((__m512i*)&InitialEdges[x + 1 + y * Width]);
                    Neighbors[5] = _mm512_loadu_si512((__m512i*)&InitialEdges[x - 1 + (y + 1) * Width]);
                    Neighbors[6] = _mm512_loadu_si512((__m512i*)&InitialEdges[x + (y + 1) * Width]);
                    Neighbors[7] = _mm512_loadu_si512((__m512i*)&InitialEdges[x + 1 + (y + 1) * Width]);

                    __m512i Res = _mm512_set1_epi8(0);
                    for (int i = 0; i < 8; i++)
                    {
                        __mmask64 LocalMask = _mm512_cmpeq_epi8_mask(Neighbors[i], HighThreshold);
                        Res                 = _mm512_mask_blend_epi8(LocalMask, Res, HighThreshold);
                    }
                    _mm512_mask_storeu_epi8(&EdgedImg[x + y * Width], Has100, Res);
                }
            }

            for (; x < Width - 1; x++)
            {
                if (InitialEdges[x + y * Width] == 100)
                {
                    bool EdgePresent = 0;
                    for (int ny = -1; ny <= 1 && !EdgePresent; ny++)
                    {
                        for (int nx = -1; nx <= 1; nx++)
                        {
                            if (InitialEdges[x + nx + (y + ny) * Width] == 255)
                            {
                                EdgePresent = 1;
                                break;
                            }
                        }
                    }
                    EdgedImg[x + y * Width] = EdgePresent ? 255 : 0;
                }
            }
        }
        return nullptr;
    };

    std::vector<pthread_t>  Threads(ThreadNum);
    std::vector<ThreadData> threadData(ThreadNum);

    int rowsPerThread = (Height - 2) / ThreadNum;
    for (int i = 0; i < ThreadNum; i++)
    {
        threadData[i] = {EdgedImg,
            InitialEdges,
            Width,
            Height,
            1 + i * rowsPerThread,
            (i == ThreadNum - 1) ? Height - 1 : 1 + (i + 1) * rowsPerThread};
        pthread_create(&Threads[i], nullptr, ThreadFunc, &threadData[i]);
    }

    for (int i = 0; i < ThreadNum; i++) { pthread_join(Threads[i], nullptr); }
}

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
                for (; x < Offset + (16 - Offset % 16); x++)
                {
                    if (x >= Width - Offset) break;
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

                for (; x <= Width - 16 - Offset; x += 16)
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
                    _mm512_store_ps(Gradients + x + y * Width, Magnitude);

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

                    _mm_store_si128((__m128i*)(GradDires + x + y * Width), _mm512_cvtsepi32_epi8(Directions));
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
{
    int                            RowPerThread = (Height - 2) / ThreadNum;
    std::vector<std::future<void>> futures;

    auto Reduce = [](float* Magnitudes, float* Gradients, uint8_t* Direction, int Width, int StartY, int EndY) {
        memcpy(Magnitudes + StartY * Width, Gradients + StartY * Width, (EndY - StartY) * Width * sizeof(float));

        __m512i Dir1 = _mm512_set1_epi8(1);
        __m512i Dir2 = _mm512_set1_epi8(2);
        __m512i Dir3 = _mm512_set1_epi8(3);
        __m512i Dir4 = _mm512_set1_epi8(4);

        for (int y = StartY; y < EndY; y++)
        {
            int x = 1;
            for (; x <= Width - 17; x += 16)
            {
                int Pos = x + (y * Width);

                __m512i Directions = _mm512_loadu_si512((__m512i*)&Direction[Pos]);
                __m512  Grads      = _mm512_loadu_ps(&Gradients[Pos]);
                __m512  Magn       = Grads;

                __mmask16 Mask1 = _mm512_cmpeq_epi8_mask(Directions, Dir1);
                __mmask16 Mask2 = _mm512_cmpeq_epi8_mask(Directions, Dir2);
                __mmask16 Mask3 = _mm512_cmpeq_epi8_mask(Directions, Dir3);
                __mmask16 Mask4 = _mm512_cmpeq_epi8_mask(Directions, Dir4);

                __m512 GradsLeft        = _mm512_loadu_ps(&Gradients[Pos - 1]);
                __m512 GradsRight       = _mm512_loadu_ps(&Gradients[Pos + 1]);
                __m512 GradsTopLeft     = _mm512_loadu_ps(&Gradients[Pos - (Width + 1)]);
                __m512 GradsTopRight    = _mm512_loadu_ps(&Gradients[Pos + (Width + 1)]);
                __m512 GradsTop         = _mm512_loadu_ps(&Gradients[Pos - Width]);
                __m512 GradsBottom      = _mm512_loadu_ps(&Gradients[Pos + Width]);
                __m512 GradsBottomLeft  = _mm512_loadu_ps(&Gradients[Pos - (Width - 1)]);
                __m512 GradsBottomRight = _mm512_loadu_ps(&Gradients[Pos + (Width - 1)]);

                __mmask16 MaskDir1 = _mm512_kor(_mm512_cmp_ps_mask(GradsLeft, Grads, _CMP_GE_OQ),
                    _mm512_cmp_ps_mask(GradsRight, Grads, _CMP_GT_OQ));
                __mmask16 MaskDir2 = _mm512_kor(_mm512_cmp_ps_mask(GradsTopLeft, Grads, _CMP_GE_OQ),
                    _mm512_cmp_ps_mask(GradsBottomRight, Grads, _CMP_GT_OQ));
                __mmask16 MaskDir3 = _mm512_kor(_mm512_cmp_ps_mask(GradsTop, Grads, _CMP_GE_OQ),
                    _mm512_cmp_ps_mask(GradsBottom, Grads, _CMP_GT_OQ));
                __mmask16 MaskDir4 = _mm512_kor(_mm512_cmp_ps_mask(GradsTopRight, Grads, _CMP_GE_OQ),
                    _mm512_cmp_ps_mask(GradsBottomLeft, Grads, _CMP_GT_OQ));

                __mmask16 FinalMask1 = _mm512_kand(Mask1, MaskDir1);
                __mmask16 FinalMask2 = _mm512_kand(Mask2, MaskDir2);
                __mmask16 FinalMask3 = _mm512_kand(Mask3, MaskDir3);
                __mmask16 FinalMask4 = _mm512_kand(Mask4, MaskDir4);

                __mmask16 FinalMask =
                    _mm512_kor(_mm512_kor(FinalMask1, FinalMask2), _mm512_kor(FinalMask3, FinalMask4));

                Magn = _mm512_mask_blend_ps(FinalMask, Magn, _mm512_set1_ps(0.0f));
                _mm512_storeu_ps(&Magnitudes[Pos], Magn);
            }

            for (; x < Width - 1; x++)
            {
                int Pos = x + (y * Width);
                switch (Direction[Pos])
                {
                    case 1:
                        if (Gradients[Pos - 1] >= Gradients[Pos] || Gradients[Pos + 1] > Gradients[Pos])
                            Magnitudes[Pos] = 0;
                        break;
                    case 2:
                        if (Gradients[Pos - (Width - 1)] >= Gradients[Pos] ||
                            Gradients[Pos + (Width - 1)] > Gradients[Pos])
                            Magnitudes[Pos] = 0;
                        break;
                    case 3:
                        if (Gradients[Pos - Width] >= Gradients[Pos] || Gradients[Pos + Width] > Gradients[Pos])
                            Magnitudes[Pos] = 0;
                        break;
                    case 4:
                        if (Gradients[Pos - (Width + 1)] >= Gradients[Pos] ||
                            Gradients[Pos + (Width + 1)] > Gradients[Pos])
                            Magnitudes[Pos] = 0;
                        break;
                    default: Magnitudes[Pos] = 0; break;
                }
            }
        }
    };

    for (int i = 0; i < ThreadNum; ++i)
    {
        int startY = 1 + i * RowPerThread;
        int endY   = (i == ThreadNum - 1) ? Height - 1 : startY + RowPerThread;
        futures.emplace_back(Pool.EnQueue(Reduce, Magnitudes, Gradients, Direction, Width, startY, endY));
    }

    for (auto& future : futures) { future.get(); }

    Pool.Sync();
}

void PThreadWithPool::PerformDoubleThresholding(
    uint8_t* EdgedImg, float* Magnitudes, int HighThre, int LowThre, int Width, int Height)
{
    __m512         HighThreshold = _mm512_set1_ps(static_cast<float>(HighThre));
    __m512         LowThreshold  = _mm512_set1_ps(static_cast<float>(LowThre));
    static __m512i MaxVal        = _mm512_set1_epi32(255);
    static __m512i MidVal        = _mm512_set1_epi32(100);
    static __m512i Zero          = _mm512_set1_epi32(0);

    int RPT = Height / ThreadNum;

    for (int i = 0; i < ThreadNum; i++)
    {
        int RFrom = i * RPT;
        int REnd  = (i == ThreadNum - 1) ? Height : (RFrom + RPT);

        auto ThreadFunc = [EdgedImg, Magnitudes, HighThre, LowThre, Width, RFrom, REnd, HighThreshold, LowThreshold]() {
            for (int y = RFrom; y < REnd; ++y)
            {
                for (int x = 0; x < Width; x += 16)
                {
                    int    PixelIdx = x + (y * Width);
                    __m512 Mag      = _mm512_load_ps(&Magnitudes[PixelIdx]);

                    __mmask16 HighMask = _mm512_cmp_ps_mask(Mag, HighThreshold, _CMP_GT_OQ);
                    __mmask16 LowMask  = _mm512_cmp_ps_mask(Mag, LowThreshold, _CMP_GT_OQ);

                    _mm_store_si128((__m128i*)&EdgedImg[PixelIdx],
                        _mm256_cvtepi16_epi8(_mm512_cvtepi32_epi16(_mm512_mask_blend_epi32(
                            HighMask, _mm512_mask_blend_epi32(LowMask, Zero, MidVal), MaxVal))));
                }
            }
        };

        Pool.EnQueue(ThreadFunc);
    }

    Pool.Sync();
}

void PThreadWithPool::PerformEdgeHysteresis(uint8_t* EdgedImg, uint8_t* InitialEdges, int Width, int Height)
{
    static __m512i LowThreshold  = _mm512_set1_epi8(static_cast<uint8_t>(100));
    static __m512i HighThreshold = _mm512_set1_epi8(static_cast<uint8_t>(255));

    memcpy(EdgedImg, InitialEdges, Width * Height * sizeof(uint8_t));

    int RowPerThread = (Height - 2) / ThreadNum;

    auto ThreadFunc = [](uint8_t* EdgedImg, uint8_t* InitialEdges, int Width, int StartY, int EndY) {
        for (int y = StartY; y < EndY; y++)
        {
            int x = 1;
            for (; x <= Width - 65; x += 64)
            {
                __m512i   CurPixels = _mm512_loadu_si512((__m512i*)&InitialEdges[x + y * Width]);
                __mmask64 Has100    = _mm512_cmpeq_epi8_mask(CurPixels, LowThreshold);

                if (Has100)
                {
                    __m512i Neighbors[8];
                    Neighbors[0] = _mm512_loadu_si512((__m512i*)&InitialEdges[x - 1 + (y - 1) * Width]);
                    Neighbors[1] = _mm512_loadu_si512((__m512i*)&InitialEdges[x + (y - 1) * Width]);
                    Neighbors[2] = _mm512_loadu_si512((__m512i*)&InitialEdges[x + 1 + (y - 1) * Width]);
                    Neighbors[3] = _mm512_loadu_si512((__m512i*)&InitialEdges[x - 1 + y * Width]);
                    Neighbors[4] = _mm512_loadu_si512((__m512i*)&InitialEdges[x + 1 + y * Width]);
                    Neighbors[5] = _mm512_loadu_si512((__m512i*)&InitialEdges[x - 1 + (y + 1) * Width]);
                    Neighbors[6] = _mm512_loadu_si512((__m512i*)&InitialEdges[x + (y + 1) * Width]);
                    Neighbors[7] = _mm512_loadu_si512((__m512i*)&InitialEdges[x + 1 + (y + 1) * Width]);

                    __m512i Res = _mm512_set1_epi8(0);
                    for (int i = 0; i < 8; i++)
                    {
                        __mmask64 LocalMask = _mm512_cmpeq_epi8_mask(Neighbors[i], HighThreshold);
                        Res                 = _mm512_mask_blend_epi8(LocalMask, Res, HighThreshold);
                    }
                    _mm512_mask_storeu_epi8(&EdgedImg[x + y * Width], Has100, Res);
                }
            }

            for (; x < Width - 1; x++)
            {
                if (InitialEdges[x + y * Width] == 100)
                {
                    bool EdgePresent = 0;
                    for (int ny = -1; ny <= 1 && !EdgePresent; ny++)
                    {
                        for (int nx = -1; nx <= 1; nx++)
                        {
                            if (InitialEdges[x + nx + (y + ny) * Width] == 255)
                            {
                                EdgePresent = 1;
                                break;
                            }
                        }
                    }
                    EdgedImg[x + y * Width] = EdgePresent ? 255 : 0;
                }
            }
        }
    };

    std::vector<std::future<void>> futures;

    for (int i = 0; i < ThreadNum; i++)
    {
        int startY = 1 + i * RowPerThread;
        int endY   = (i == ThreadNum - 1) ? Height - 1 : startY + RowPerThread;
        futures.emplace_back(Pool.EnQueue(ThreadFunc, EdgedImg, InitialEdges, Width, startY, endY));
    }

    for (auto& future : futures) future.get();

    Pool.Sync();
}
