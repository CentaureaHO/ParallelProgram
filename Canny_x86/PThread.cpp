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
    struct ThreadData
    {
        int            From;
        int            To;
        int            W;
        int            H;
        const uint8_t* InputImg;
        uint8_t*       OutputImg;
    };

    auto Gauss = [](void* arg) -> void* {
        ThreadData*    data      = static_cast<ThreadData*>(arg);
        int            start     = data->From;
        int            end       = data->To;
        int            width     = data->W;
        const uint8_t* inputImg  = data->InputImg;
        uint8_t*       OutputImg = data->OutputImg;

        for (int y = start; y < end; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                float PixelVal = 0.0f, KernelSum = 0.0f;
                for (int i = -KernelRadius; i <= KernelRadius; i++)
                {
                    int nx = x + i;
                    if (nx >= 0 && nx < width)
                    {
                        PixelVal += GaussianKernel_1D[i + KernelRadius] * inputImg[y * width + nx];
                        KernelSum += GaussianKernel_1D[i + KernelRadius];
                    }
                }
                OutputImg[y * width + x] = (uint8_t)(PixelVal / KernelSum);
            }
        }
        return NULL;
    };

    pthread_t*  Threads       = new pthread_t[ThreadNum];
    ThreadData* ThreadDatas   = new ThreadData[ThreadNum];
    int         RowsPerThread = Height / ThreadNum;

    for (int i = 0; i < ThreadNum; i++)
    {
        ThreadDatas[i].From = i * RowsPerThread;
        ThreadDatas[i].To   = (i + 1) * RowsPerThread;
        if (i == ThreadNum - 1) ThreadDatas[i].To = Height;
        ThreadDatas[i].W         = Width;
        ThreadDatas[i].H         = Height;
        ThreadDatas[i].InputImg  = OriImg;
        ThreadDatas[i].OutputImg = Output;

        pthread_create(&Threads[i], NULL, Gauss, &ThreadDatas[i]);
    }

    for (int i = 0; i < ThreadNum; i++) { pthread_join(Threads[i], NULL); }
    delete[] Threads;
    delete[] ThreadDatas;
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

        _mm512_memcpy(Magnitudes + StartY * Width, Gradients + StartY * Width, (EndY - StartY) * Width);

        __m512i Dir1 = _mm512_set1_epi8(1);
        __m512i Dir2 = _mm512_set1_epi8(2);
        __m512i Dir3 = _mm512_set1_epi8(3);
        __m512i Dir4 = _mm512_set1_epi8(4);

        for (int y = StartY; y < EndY; y++)
        {
            int x = 1;
            for (; x < Width - 1 && ((uintptr_t)&Gradients[x + y * Width] & 63) != 0; x++)
            {
                int Pos = x + (y * Width);

                float   Grad = Gradients[Pos];
                uint8_t Dir  = Direction[Pos];

                switch (Dir)
                {
                    case 1:
                        if (Gradients[Pos - 1] >= Grad || Gradients[Pos + 1] > Grad) Magnitudes[Pos] = 0;
                        break;
                    case 2:
                        if (Gradients[Pos - (Width - 1)] >= Grad || Gradients[Pos + (Width - 1)] > Grad)
                            Magnitudes[Pos] = 0;
                        break;
                    case 3:
                        if (Gradients[Pos - Width] >= Grad || Gradients[Pos + Width] > Grad) Magnitudes[Pos] = 0;
                        break;
                    case 4:
                        if (Gradients[Pos - (Width + 1)] >= Grad || Gradients[Pos + (Width + 1)] > Grad)
                            Magnitudes[Pos] = 0;
                        break;
                    default: Magnitudes[Pos] = 0; break;
                }
            }

            for (; x < Width - 1; x += 16)
            {
                int Pos = x + (y * Width);

                __m512  Grad = _mm512_load_ps(&Gradients[Pos]);
                __m512i Dir  = _mm512_loadu_si512((__m512i*)&Direction[Pos]);

                __mmask16 Mask1 = _mm512_cmpeq_epi8_mask(Dir, Dir1);
                __mmask16 Mask2 = _mm512_cmpeq_epi8_mask(Dir, Dir2);
                __mmask16 Mask3 = _mm512_cmpeq_epi8_mask(Dir, Dir3);
                __mmask16 Mask4 = _mm512_cmpeq_epi8_mask(Dir, Dir4);

                __m512 GradML    = _mm512_loadu_ps(&Gradients[Pos - 1]);
                __m512 GradPL    = _mm512_loadu_ps(&Gradients[Pos + 1]);
                __m512 GradMWL   = _mm512_loadu_ps(&Gradients[Pos - (Width - 1)]);
                __m512 GradPWL   = _mm512_loadu_ps(&Gradients[Pos + (Width - 1)]);
                __m512 GradMW    = _mm512_loadu_ps(&Gradients[Pos - Width]);
                __m512 GradPW    = _mm512_loadu_ps(&Gradients[Pos + Width]);
                __m512 GradMWLPL = _mm512_loadu_ps(&Gradients[Pos - (Width + 1)]);
                __m512 GradPWLPL = _mm512_loadu_ps(&Gradients[Pos + (Width + 1)]);

                __mmask16 ResMask1 = _mm512_kand(Mask1,
                    _mm512_kor(
                        _mm512_cmp_ps_mask(GradML, Grad, _CMP_GE_OQ), _mm512_cmp_ps_mask(GradPL, Grad, _CMP_GT_OQ)));
                __mmask16 ResMask2 = _mm512_kand(Mask2,
                    _mm512_kor(
                        _mm512_cmp_ps_mask(GradMWL, Grad, _CMP_GE_OQ), _mm512_cmp_ps_mask(GradPWL, Grad, _CMP_GT_OQ)));
                __mmask16 ResMask3 = _mm512_kand(Mask3,
                    _mm512_kor(
                        _mm512_cmp_ps_mask(GradMW, Grad, _CMP_GE_OQ), _mm512_cmp_ps_mask(GradPW, Grad, _CMP_GT_OQ)));
                __mmask16 ResMask4 = _mm512_kand(Mask4,
                    _mm512_kor(_mm512_cmp_ps_mask(GradMWLPL, Grad, _CMP_GE_OQ),
                        _mm512_cmp_ps_mask(GradPWLPL, Grad, _CMP_GT_OQ)));
                __mmask16 ResMask  = _mm512_kor(_mm512_kor(ResMask1, ResMask2), _mm512_kor(ResMask3, ResMask4));

                _mm512_mask_store_ps(&Magnitudes[Pos], ResMask, _mm512_setzero_ps());
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
    struct ThreadArgs
    {
        uint8_t* EdgedImg;
        uint8_t* InitialEdges;
        int      Width;
        int      Height;
        int      ThreadNum;
        int      ThreadId;
    };
    _mm512_memcpy(EdgedImg, InitialEdges, Width * Height);

    auto ThreadFunction = [](void* arg) -> void* {
        ThreadArgs* args         = static_cast<ThreadArgs*>(arg);
        uint8_t*    EdgedImg     = args->EdgedImg;
        uint8_t*    InitialEdges = args->InitialEdges;
        int         Width        = args->Width;
        int         Height       = args->Height;
        int         ThreadId     = args->ThreadId;
        int         ThreadNum    = args->ThreadNum;

        int blockSize = (Height - 2) / ThreadNum;
        int startY    = ThreadId * blockSize + 1;
        int endY      = (ThreadId == ThreadNum - 1) ? Height - 1 : (ThreadId + 1) * blockSize + 1;

        std::vector<uint8_t> localVisited((endY - startY + 2) * Width, 0);
        std::vector<int>     localEdgeQueue;

        for (int y = startY; y < endY; y++)
        {
            for (int x = 1; x < Width - 1; x += 16)
            {
                int     Pos          = x + y * Width;
                int     localPos     = x + (y - startY + 1) * Width;
                __m512i initialEdges = _mm512_loadu_si512((__m512i*)&InitialEdges[Pos]);
                __m512i visited      = _mm512_loadu_si512((__m512i*)&localVisited[localPos]);

                __mmask64 edgeMask    = _mm512_cmpeq_epi8_mask(initialEdges, _mm512_set1_epi8(100));
                __mmask64 visitedMask = _mm512_cmpeq_epi8_mask(visited, _mm512_set1_epi8(1));

                __mmask64 combinedMask = _kandn_mask64(visitedMask, edgeMask);

                for (int i = 0; i < 16; i++)
                {
                    int PixelIdx = Pos + i;
                    int localIdx = localPos + i;
                    if (combinedMask & (1ULL << i) && !localVisited[localIdx])
                    {
                        bool HasStrongNeighbor =
                            (InitialEdges[PixelIdx - 1] == 255 || InitialEdges[PixelIdx + 1] == 255 ||
                                InitialEdges[PixelIdx - Width] == 255 || InitialEdges[PixelIdx + Width] == 255 ||
                                InitialEdges[PixelIdx - Width - 1] == 255 ||
                                InitialEdges[PixelIdx - Width + 1] == 255 ||
                                InitialEdges[PixelIdx + Width - 1] == 255 || InitialEdges[PixelIdx + Width + 1] == 255);
                        if (HasStrongNeighbor)
                        {
                            localEdgeQueue.push_back(PixelIdx);
                            localVisited[localIdx] = 1;
                        }
                        else { EdgedImg[PixelIdx] = 0; }
                    }
                }
            }
        }

        for (size_t i = 0; i < localEdgeQueue.size(); ++i)
        {
            int PixelIdx = localEdgeQueue[i];

            for (int dx = -1; dx <= 1; dx++)
            {
                for (int dy = -1; dy <= 1; dy++)
                {
                    int newX        = (PixelIdx % Width) + dx;
                    int newY        = (PixelIdx / Width) + dy;
                    int newPixelIdx = newX + newY * Width;
                    int localNewIdx = newX + (newY - startY + 1) * Width;

                    if (newX >= 0 && newX < Width && newY >= startY - 1 && newY < endY + 1 &&
                        !localVisited[localNewIdx] && InitialEdges[newPixelIdx] == 100)
                    {
                        localEdgeQueue.push_back(newPixelIdx);
                        localVisited[localNewIdx] = 1;
                        EdgedImg[newPixelIdx]     = 255;
                    }
                }
            }
        }
        return nullptr;
    };

    std::vector<pthread_t>  Threads(ThreadNum);
    std::vector<ThreadArgs> Args(ThreadNum);

    for (int t = 0; t < ThreadNum; ++t)
    {
        Args[t] = {EdgedImg, InitialEdges, Width, Height, ThreadNum, t};
        pthread_create(&Threads[t], nullptr, ThreadFunction, &Args[t]);
    }

    for (int t = 0; t < ThreadNum; ++t) { pthread_join(Threads[t], nullptr); }
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
                        __m512 ImgPixel  = _mm512_loadu_ps(Temp + ny * PaddedWidth + x);
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

    int NumBlocks = ThreadNum * 4;
    int BlockSize = (Height + NumBlocks - 1) / NumBlocks;

    std::mutex                       TaskMutex;
    std::vector<std::pair<int, int>> Tasks;

    for (int t = 0; t < NumBlocks; ++t)
    {
        int startY = t * BlockSize;
        int endY   = std::min((t + 1) * BlockSize, Height);
        Tasks.push_back({startY, endY});
    }

    for (int t = 0; t < ThreadNum; ++t)
    {
        Pool.EnQueue([&, t] {
            while (true)
            {
                std::pair<int, int> task;

                {
                    std::lock_guard<std::mutex> lock(TaskMutex);
                    if (Tasks.empty()) break;
                    task = Tasks.back();
                    Tasks.pop_back();
                }

                int startY = task.first;
                int endY   = task.second;

                ProcessRow(startY, endY);
                ProcessColumn(startY + KernelRadius, endY - KernelRadius);
            }
        });
    }

    Pool.Sync();

    auto FinalizeOutput = [&](int start, int end) {
        __m512 zero          = _mm512_set1_ps(0.0f);
        __m512 two_five_five = _mm512_set1_ps(255.0f);
        for (int i = start; i <= end - 16; i += 16)
        {
            __m512 Pixels      = _mm512_loadu_ps(Temp + i);
            Pixels             = _mm512_max_ps(zero, _mm512_min_ps(Pixels, two_five_five));
            __m512i Pixels_i32 = _mm512_cvtps_epi32(Pixels);
            __m256i Pixels_i16 = _mm512_cvtepi32_epi16(Pixels_i32);
            __m128i Pixels_i8  = _mm256_cvtepi16_epi8(Pixels_i16);
            _mm_storeu_si128((__m128i*)(Output + i), Pixels_i8);
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

    int NumBlocks = ThreadNum * 4;
    int BlockSize = (Height + NumBlocks - 1) / NumBlocks;

    std::mutex                       TaskMutex;
    std::vector<std::pair<int, int>> Tasks;

    for (int t = 0; t < NumBlocks; ++t)
    {
        int startY = t * BlockSize;
        int endY   = std::min((t + 1) * BlockSize, Height);
        Tasks.push_back({startY, endY});
    }

    auto ProcessBlock = [&](int start, int end) {
        for (int y = start; y < end; y++)
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
                        if (x + kx >= 0 && x + kx < Width && y + ky >= 0 && y + ky < Height)
                        {
                            int PixelIdx = x + kx + (y + ky) * Width;
                            GradX += BlurredImage[PixelIdx] * Gx[KernelIdx];
                            GradY += BlurredImage[PixelIdx] * Gy[KernelIdx];
                        }
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

                        if (x + 15 < Width && (x + kx >= 0 && x + kx < Width && y + ky >= 0 && y + ky < Height))
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
    for (int t = 0; t < ThreadNum; ++t)
    {
        Pool.EnQueue([&, t] {
            while (true)
            {
                std::pair<int, int> task;

                {
                    std::lock_guard<std::mutex> lock(TaskMutex);
                    if (Tasks.empty()) break;
                    task = Tasks.back();
                    Tasks.pop_back();
                }

                int startY = task.first;
                int endY   = task.second;

                ProcessBlock(startY, endY);
            }
        });
    }

    Pool.Sync();
}

void PThreadWithPool::ReduceNonMaximum(float* Magnitudes, float* Gradients, uint8_t* Direction, int Width, int Height)
{
    int                            RowPerThread = (Height - 2) / ThreadNum;
    std::vector<std::future<void>> futures;

    auto Reduce = [](float* Magnitudes, float* Gradients, uint8_t* Direction, int Width, int StartY, int EndY) {
        _mm512_memcpy(Magnitudes + StartY * Width, Gradients + StartY * Width, (EndY - StartY) * Width);

        __m512i Dir1 = _mm512_set1_epi8(1);
        __m512i Dir2 = _mm512_set1_epi8(2);
        __m512i Dir3 = _mm512_set1_epi8(3);
        __m512i Dir4 = _mm512_set1_epi8(4);

        for (int y = StartY; y < EndY; y++)
        {
            int x = 1;
            for (; x < Width - 1 && ((uintptr_t)&Gradients[x + y * Width] & 63) != 0; x++)
            {
                int Pos = x + (y * Width);

                float   Grad = Gradients[Pos];
                uint8_t Dir  = Direction[Pos];

                switch (Dir)
                {
                    case 1:
                        if (Gradients[Pos - 1] >= Grad || Gradients[Pos + 1] > Grad) Magnitudes[Pos] = 0;
                        break;
                    case 2:
                        if (Gradients[Pos - (Width - 1)] >= Grad || Gradients[Pos + (Width - 1)] > Grad)
                            Magnitudes[Pos] = 0;
                        break;
                    case 3:
                        if (Gradients[Pos - Width] >= Grad || Gradients[Pos + Width] > Grad) Magnitudes[Pos] = 0;
                        break;
                    case 4:
                        if (Gradients[Pos - (Width + 1)] >= Grad || Gradients[Pos + (Width + 1)] > Grad)
                            Magnitudes[Pos] = 0;
                        break;
                    default: Magnitudes[Pos] = 0; break;
                }
            }

            for (; x < Width - 1; x += 16)
            {
                int Pos = x + (y * Width);

                __m512  Grad = _mm512_load_ps(&Gradients[Pos]);
                __m512i Dir  = _mm512_loadu_si512((__m512i*)&Direction[Pos]);

                __mmask16 Mask1 = _mm512_cmpeq_epi8_mask(Dir, Dir1);
                __mmask16 Mask2 = _mm512_cmpeq_epi8_mask(Dir, Dir2);
                __mmask16 Mask3 = _mm512_cmpeq_epi8_mask(Dir, Dir3);
                __mmask16 Mask4 = _mm512_cmpeq_epi8_mask(Dir, Dir4);

                __m512 GradML    = _mm512_loadu_ps(&Gradients[Pos - 1]);
                __m512 GradPL    = _mm512_loadu_ps(&Gradients[Pos + 1]);
                __m512 GradMWL   = _mm512_loadu_ps(&Gradients[Pos - (Width - 1)]);
                __m512 GradPWL   = _mm512_loadu_ps(&Gradients[Pos + (Width - 1)]);
                __m512 GradMW    = _mm512_loadu_ps(&Gradients[Pos - Width]);
                __m512 GradPW    = _mm512_loadu_ps(&Gradients[Pos + Width]);
                __m512 GradMWLPL = _mm512_loadu_ps(&Gradients[Pos - (Width + 1)]);
                __m512 GradPWLPL = _mm512_loadu_ps(&Gradients[Pos + (Width + 1)]);

                __mmask16 ResMask1 = _mm512_kand(Mask1,
                    _mm512_kor(
                        _mm512_cmp_ps_mask(GradML, Grad, _CMP_GE_OQ), _mm512_cmp_ps_mask(GradPL, Grad, _CMP_GT_OQ)));
                __mmask16 ResMask2 = _mm512_kand(Mask2,
                    _mm512_kor(
                        _mm512_cmp_ps_mask(GradMWL, Grad, _CMP_GE_OQ), _mm512_cmp_ps_mask(GradPWL, Grad, _CMP_GT_OQ)));
                __mmask16 ResMask3 = _mm512_kand(Mask3,
                    _mm512_kor(
                        _mm512_cmp_ps_mask(GradMW, Grad, _CMP_GE_OQ), _mm512_cmp_ps_mask(GradPW, Grad, _CMP_GT_OQ)));
                __mmask16 ResMask4 = _mm512_kand(Mask4,
                    _mm512_kor(_mm512_cmp_ps_mask(GradMWLPL, Grad, _CMP_GE_OQ),
                        _mm512_cmp_ps_mask(GradPWLPL, Grad, _CMP_GT_OQ)));
                __mmask16 ResMask  = _mm512_kor(_mm512_kor(ResMask1, ResMask2), _mm512_kor(ResMask3, ResMask4));

                _mm512_mask_store_ps(&Magnitudes[Pos], ResMask, _mm512_setzero_ps());
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
    _mm512_memcpy(EdgedImg, InitialEdges, Width * Height);

    std::vector<std::future<void>>   Futures;
    int                              NumBlocks = ThreadNum * 4;
    int                              BlockSize = (Height - 2) / NumBlocks;
    std::mutex                       TaskMutex;
    std::vector<std::pair<int, int>> Tasks;

    for (int t = 0; t < NumBlocks; ++t)
    {
        int startY = t * BlockSize + 1;
        int endY   = (t == NumBlocks - 1) ? Height - 1 : (t + 1) * BlockSize + 1;
        Tasks.push_back({startY, endY});
    }

    for (int t = 0; t < ThreadNum; ++t)
    {
        Futures.push_back(Pool.EnQueue([&, t] {
            while (true)
            {
                std::pair<int, int> task;

                {
                    std::lock_guard<std::mutex> lock(TaskMutex);
                    if (Tasks.empty()) break;
                    task = Tasks.back();
                    Tasks.pop_back();
                }

                int startY = task.first;
                int endY   = task.second;

                std::vector<uint8_t> localVisited((endY - startY + 2) * Width, 0);
                std::vector<int>     localEdgeQueue;

                for (int y = startY; y < endY; y++)
                {
                    for (int x = 1; x < Width - 1; x += 16)
                    {
                        int     Pos          = x + y * Width;
                        int     localPos     = x + (y - startY + 1) * Width;
                        __m512i initialEdges = _mm512_loadu_si512((__m512i*)&InitialEdges[Pos]);
                        __m512i visited      = _mm512_loadu_si512((__m512i*)&localVisited[localPos]);

                        __mmask64 edgeMask    = _mm512_cmpeq_epi8_mask(initialEdges, _mm512_set1_epi8(100));
                        __mmask64 visitedMask = _mm512_cmpeq_epi8_mask(visited, _mm512_set1_epi8(1));

                        __mmask64 combinedMask = _kandn_mask64(visitedMask, edgeMask);

                        for (int i = 0; i < 16; i++)
                        {
                            int PixelIdx = Pos + i;
                            int localIdx = localPos + i;
                            if (combinedMask & (1ULL << i) && !localVisited[localIdx])
                            {
                                bool HasStrongNeighbor =
                                    (InitialEdges[PixelIdx - 1] == 255 || InitialEdges[PixelIdx + 1] == 255 ||
                                        InitialEdges[PixelIdx - Width] == 255 ||
                                        InitialEdges[PixelIdx + Width] == 255 ||
                                        InitialEdges[PixelIdx - Width - 1] == 255 ||
                                        InitialEdges[PixelIdx - Width + 1] == 255 ||
                                        InitialEdges[PixelIdx + Width - 1] == 255 ||
                                        InitialEdges[PixelIdx + Width + 1] == 255);
                                if (HasStrongNeighbor)
                                {
                                    localEdgeQueue.push_back(PixelIdx);
                                    localVisited[localIdx] = 1;
                                }
                                else { EdgedImg[PixelIdx] = 0; }
                            }
                        }
                    }
                }

                for (size_t i = 0; i < localEdgeQueue.size(); ++i)
                {
                    int PixelIdx = localEdgeQueue[i];

                    for (int dx = -1; dx <= 1; dx++)
                    {
                        for (int dy = -1; dy <= 1; dy++)
                        {
                            int newX        = (PixelIdx % Width) + dx;
                            int newY        = (PixelIdx / Width) + dy;
                            int newPixelIdx = newX + newY * Width;
                            int localNewIdx = newX + (newY - startY + 1) * Width;

                            if (newX >= 0 && newX < Width && newY >= startY - 1 && newY < endY + 1 &&
                                !localVisited[localNewIdx] && InitialEdges[newPixelIdx] == 100)
                            {
                                localEdgeQueue.push_back(newPixelIdx);
                                localVisited[localNewIdx] = 1;
                                EdgedImg[newPixelIdx]     = 255;
                            }
                        }
                    }
                }
            }
        }));
    }

    for (auto& future : Futures) { future.get(); }

    Pool.Sync();
}
