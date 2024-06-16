#include <bits/stdc++.h>
#include <immintrin.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <mpi.h>
using namespace std;

const int ThreadNum            = 15;
float     GaussianKernel_1D[3] = {1, 2, 1};
float     GaussianKernel_2D[9] = {1, 2, 1, 2, 4, 2, 1, 2, 1};
const int KernelRadius         = 1;
const int KernelSize           = 3;

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

template <typename T>
void _mm512_memcpy(T* dest, const T* src, size_t size)
{
    const size_t ElePerReg = 64 / sizeof(T);

    size_t FullReg = size / ElePerReg;
    for (size_t i = 0; i < FullReg; ++i)
    {
        __m512i Data = _mm512_load_si512(reinterpret_cast<const __m512i*>(src) + i);
        _mm512_store_si512(reinterpret_cast<__m512i*>(dest) + i, Data);
    }

    size_t RemainEles = size % ElePerReg;
    size_t OffSet     = FullReg * ElePerReg;

    for (size_t i = 0; i < RemainEles; ++i) { dest[OffSet + i] = src[OffSet + i]; }
}

template <class F, class... Args>
using RetType = typename std::invoke_result<F, Args...>::type;

void* ThreadEntry(void* args);

class ThreadPool
{
  private:
    std::vector<pthread_t>            Workers;
    std::queue<std::function<void()>> Tasks;
    std::mutex                        QueueMutex;
    std::condition_variable           CondVar;
    std::condition_variable           FinishedVar;
    bool                              Stop;
    unsigned int                      ActiveTasks;

  public:
    ThreadPool(unsigned int ThreadNum) : Stop(0), ActiveTasks(0)
    {
        Workers.resize(ThreadNum);
        for (auto& Worker : Workers) { pthread_create(&Worker, nullptr, ThreadEntry, this); }
    }
    ~ThreadPool()
    {
        {
            std::unique_lock<std::mutex> Lock(QueueMutex);
            Stop = 1;
        }
        CondVar.notify_all();
        for (auto& Worker : Workers) { pthread_join(Worker, nullptr); }
    }

    template <class F, class... Args>
    std::future<RetType<F, Args...>> EnQueue(F&& ThFunc, Args&&... args)
    {
        auto Task = std::make_shared<std::packaged_task<RetType<F, Args...>()>>(
            std::bind(std::forward<F>(ThFunc), std::forward<Args>(args)...));

        std::future<RetType<F, Args...>> Res = Task->get_future();
        {
            std::unique_lock<std::mutex> Lock(QueueMutex);
            Tasks.emplace([Task] { (*Task)(); });
        }
        CondVar.notify_one();
        return Res;
    }

    void Run()
    {
        while (true)
        {
            std::function<void()> Task;
            {
                std::unique_lock<std::mutex> Lock(QueueMutex);
                CondVar.wait(Lock, [this] { return Stop || !Tasks.empty(); });
                if (Stop && Tasks.empty()) break;
                Task = std::move(Tasks.front());
                Tasks.pop();
                ++ActiveTasks;
            }
            Task();
            {
                std::unique_lock<std::mutex> Lock(QueueMutex);
                --ActiveTasks;
                if (ActiveTasks == 0 && Tasks.empty()) FinishedVar.notify_all();
            }
        }
    }

    void Sync()
    {
        std::unique_lock<std::mutex> Lock(QueueMutex);
        FinishedVar.wait(Lock, [this] { return Tasks.empty() && ActiveTasks == 0; });
    }
};

void* ThreadEntry(void* args)
{
    auto* Pool = static_cast<ThreadPool*>(args);
    Pool->Run();
    return nullptr;
}

ThreadPool Pool(ThreadNum);

void PerformGaussianBlur(uint8_t* Output, const uint8_t* OriImg, int Width, int Height)
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

void ComputeGradients(float* Gradients, uint8_t* GradDires, const uint8_t* BlurredImage, int Width, int Height)
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

void ReduceNonMaximum(float* Magnitudes, float* Gradients, uint8_t* Direction, int Width, int Height)
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

void PerformDoubleThresholding(uint8_t* EdgedImg, float* Magnitudes, int HighThre, int LowThre, int Width, int Height)
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

void PerformEdgeHysteresis(uint8_t* EdgedImg, uint8_t* InitialEdges, int Width, int Height)
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

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Determine the number of pipelines possible
    int pipelines = size / 4;
    if (size < 4)
    {
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    int      pipeline_id = rank / 4;
    MPI_Comm pipeline_comm;
    MPI_Comm_split(MPI_COMM_WORLD, pipeline_id, rank, &pipeline_comm);

    int pipeline_rank, pipeline_size;
    MPI_Comm_rank(pipeline_comm, &pipeline_rank);
    MPI_Comm_size(pipeline_comm, &pipeline_size);

    const string         ImgPath = "../Images/";
    const string         OutPath = "../Output/";
    const vector<string> Images  = {"bakery.jpg",
         "build.jpg",
         "car.jpg",
         "car2.jpg",
         "cat.jpg",
         "cube.jpg",
         "cupoi.jpg",
         "dog.jpg",
         "earth.jpg",
         "female.jpg",
         "female2.jpg",
         "lake.jpg",
         "machine.jpg",
         "mountain.jpg",
         "platinum.jpg",
         "street.jpg"};


    int img_index = pipeline_id;

    while (img_index < Images.size())
    {
        string  ImgName = ImgPath + Images[img_index];
        string  OutName = OutPath + "Sobel_MPI_" + Images[img_index];
        cv::Mat Img, GreyImg, EdgedImg;
        cout << "Processing image: " << ImgName << endl;
        int OriginalWidth, OriginalHeight, TmpWidth, TmpHeight;

        cout << "Rank " << rank << " reachead ckpt 1\n";
        if (pipeline_rank == 0)
        {
            // Node nk reads the image
            Img = cv::imread(ImgName, cv::IMREAD_COLOR);
            if (Img.empty())
            {
                cerr << "Error loading image: " << ImgName << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            OriginalWidth  = Img.cols;
            OriginalHeight = Img.rows;
            TmpWidth       = Img.cols < 16 ? 16 : Img.cols / 16 * 16;
            TmpHeight      = Img.rows < 16 ? 16 : Img.rows / 16 * 16;

            cout << "Rank " << rank << " reachead ckpt 2\n";
            cv::cvtColor(Img, GreyImg, cv::COLOR_BGR2GRAY);
            cv::resize(GreyImg, GreyImg, cv::Size(TmpWidth, TmpHeight));

            // Broadcast the sizes to the next nodes in the pipeline
            MPI_Send(&OriginalWidth, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
            MPI_Send(&OriginalHeight, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
            MPI_Send(&TmpWidth, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
            MPI_Send(&TmpHeight, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
            MPI_Send(GreyImg.data, TmpWidth * TmpHeight, MPI_UINT8_T, rank + 1, 0, MPI_COMM_WORLD);
            cout << "Rank " << rank << " reachead ckpt 3\n";
        }
        else
        {
            cout << "Rank " << rank << " reachead ckpt 2\n";
            // Other nodes receive the dimensions and allocate memory
            MPI_Recv(&OriginalWidth, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&OriginalHeight, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&TmpWidth, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&TmpHeight, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            GreyImg.create(TmpHeight, TmpWidth, CV_8UC1);
            MPI_Recv(GreyImg.data, TmpWidth * TmpHeight, MPI_UINT8_T, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            cout << "Rank " << rank << " reachead ckpt 3\n";
        }
        MPI_Barrier(pipeline_comm);

        // Define buffer for each processing step
        uint8_t* GaussianImageArray = (uint8_t*)_mm_malloc(TmpWidth * TmpHeight * sizeof(uint8_t), 64);
        float*   GradientPixels     = (float*)_mm_malloc(TmpWidth * TmpHeight * sizeof(float), 64);
        float*   MatrixPixels       = (float*)_mm_malloc(TmpWidth * TmpHeight * sizeof(float), 64);
        uint8_t* SegmentPixels      = (uint8_t*)_mm_malloc(TmpWidth * TmpHeight * sizeof(uint8_t), 64);
        uint8_t* DoubleThrePixels   = (uint8_t*)_mm_malloc(TmpWidth * TmpHeight * sizeof(uint8_t), 64);
        uint8_t* EdgeArray          = (uint8_t*)_mm_malloc(TmpWidth * TmpHeight * sizeof(uint8_t), 64);

        // Processing according to the position in the pipeline
        if (pipeline_rank == 0)
        {
            // Gaussian blur
            PerformGaussianBlur(GaussianImageArray, GreyImg.data, TmpWidth, TmpHeight);
            MPI_Send(GaussianImageArray, TmpWidth * TmpHeight, MPI_UINT8_T, rank + 1, 0, MPI_COMM_WORLD);
        }
        else if (pipeline_rank == 1)
        {
            // Gradient computation
            MPI_Recv(
                GaussianImageArray, TmpWidth * TmpHeight, MPI_UINT8_T, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            ComputeGradients(GradientPixels, SegmentPixels, GaussianImageArray, TmpWidth, TmpHeight);
            MPI_Send(GradientPixels, TmpWidth * TmpHeight * sizeof(float), MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD);
            MPI_Send(SegmentPixels, TmpWidth * TmpHeight, MPI_UINT8_T, rank + 1, 0, MPI_COMM_WORLD);
        }
        else if (pipeline_rank == 2)
        {
            // Non-maximum suppression and double thresholding
            MPI_Recv(GradientPixels,
                TmpWidth * TmpHeight * sizeof(float),
                MPI_FLOAT,
                rank - 1,
                0,
                MPI_COMM_WORLD,
                MPI_STATUS_IGNORE);
            MPI_Recv(SegmentPixels, TmpWidth * TmpHeight, MPI_UINT8_T, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            ReduceNonMaximum(MatrixPixels, GradientPixels, SegmentPixels, TmpWidth, TmpHeight);
            PerformDoubleThresholding(DoubleThrePixels, MatrixPixels, 90, 30, TmpWidth, TmpHeight);
            MPI_Send(DoubleThrePixels, TmpWidth * TmpHeight, MPI_UINT8_T, rank + 1, 0, MPI_COMM_WORLD);
        }
        else if (pipeline_rank == 3)
        {
            MPI_Recv(
                DoubleThrePixels, TmpWidth * TmpHeight, MPI_UINT8_T, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            PerformEdgeHysteresis(EdgeArray, DoubleThrePixels, TmpWidth, TmpHeight);
            EdgedImg.create(OriginalHeight, OriginalWidth, CV_8UC1);
            cv::resize(EdgedImg, EdgedImg, cv::Size(OriginalWidth, OriginalHeight));
            cv::imwrite(OutName, EdgedImg);
        }

        _mm_free(GaussianImageArray);
        _mm_free(GradientPixels);
        _mm_free(MatrixPixels);
        _mm_free(SegmentPixels);
        _mm_free(DoubleThrePixels);
        _mm_free(EdgeArray);

        img_index += pipelines;
    }

    MPI_Comm_free(&pipeline_comm);
    MPI_Finalize();
    return 0;
}