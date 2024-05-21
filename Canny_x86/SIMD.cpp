#include <algorithm>
#include <cstring>
#include <immintrin.h>
#include <math.h>
#include <xmmintrin.h>
#include "AVX_Lib.h"
#include "ParmsDef.h"
#include "SIMD.h"

namespace SIMD
{
    namespace AVX
    {
        A512::A512() {}
        A512::~A512() {}

        A512& A512::GetInstance()
        {
            static A512 instance;
            return instance;
        }

        void A512::PerformGaussianBlur(uint8_t* Output, const uint8_t* OriImg, int Width, int Height)
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
                            __m512i data = _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)(OriImg + y * Width + nx)));
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

            __m512 zero          = _mm512_set1_ps(0.0f);
            __m512 two_five_five = _mm512_set1_ps(255.0f);
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

        void A512::ComputeGradients(
            float* Gradients, uint8_t* GradDires, const uint8_t* BlurredImage, int Width, int Height)
        {
            static const int8_t Gx[]   = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
            static const int8_t Gy[]   = {1, 2, 1, 0, 0, 0, -1, -2, -1};
            static const int    Offset = 1;

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

                    __m512 Magnitude =
                        _mm512_sqrt_ps(_mm512_add_ps(_mm512_mul_ps(GradX, GradX), _mm512_mul_ps(GradY, GradY)));
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

        void A512::ReduceNonMaximum(float* Magnitudes, float* Gradients, uint8_t* Direction, int Width, int Height)
        {
            //
            memcpy(Magnitudes, Gradients, Width * Height * sizeof(float));
            for (int x = 1; x < Width - 1; x++)
            {
                for (int y = 1; y < Height - 1; y++)
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
        }

        void A512::PerformDoubleThresholding(
            uint8_t* EdgedImg, float* Magnitudes, int HighThre, int LowThre, int Width, int Height)
        {
            __m512         HighThreshold = _mm512_set1_ps(static_cast<float>(HighThre));
            __m512         LowThreshold  = _mm512_set1_ps(static_cast<float>(LowThre));
            static __m512i MaxVal        = _mm512_set1_epi32(255);
            static __m512i MidVal        = _mm512_set1_epi32(100);
            static __m512i Zero          = _mm512_set1_epi32(0);

            for (int y = 0; y < Height; ++y)
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
        }

        void A512::PerformEdgeHysteresis(uint8_t* EdgedImg, uint8_t* InitialEdges, int Width, int Height)
        {
            static __m512i LowThreshold  = _mm512_set1_epi8(static_cast<uint8_t>(100));
            static __m512i HighThreshold = _mm512_set1_epi8(static_cast<uint8_t>(255));

            memcpy(EdgedImg, InitialEdges, Width * Height * sizeof(uint8_t));

            for (int y = 1; y < Height - 1; y++)
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

                if (x < Width - 1)
                {
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
            }
        }
    }  // namespace AVX
}  // namespace SIMD

// 往下为SSE与AVX-2的实现，不再继续填充

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