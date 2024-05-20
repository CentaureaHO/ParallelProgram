#pragma once
#include <cstdint>
#include "CannyBase.h"

namespace SIMD
{
    class SSE
    {
      public:
        static void PerformGaussianBlur(uint8_t* Output, const uint8_t* OriImg, int Width, int Height);
    };

    namespace AVX
    {
        class A256
        {
          public:
            static void PerformGaussianBlur(uint8_t* Output, const uint8_t* OriImg, int Width, int Height);
        };

        class A512 : public Canny
        {
          public:
            static A512& GetInstance();

            void PerformGaussianBlur(uint8_t* Output, const uint8_t* OriImg, int Width, int Height) override;
            void ComputeGradients(
                float* Gradients, uint8_t* GradDires, const uint8_t* BlurredImage, int Width, int Height) override;
            void ReduceNonMaximum(
                float* Magnitudes, float* Gradients, uint8_t* Direction, int Width, int Height) override;
            void PerformDoubleThresholding(
                uint8_t* EdgedImg, float* Magnitudes, int HighThre, int LowThre, int Width, int Height) override;
            void PerformEdgeHysteresis(uint8_t* EdgedImg, uint8_t* InitialEdges, int Width, int Height) override;

          private:
            A512();
            ~A512();
        };  // namespace A512
    }  // namespace AVX
}  // namespace SIMD