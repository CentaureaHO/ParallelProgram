#pragma once
#include <cstdint>

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

        class A512
        {
          public:
            static void PerformGaussianBlur(uint8_t* Output, const uint8_t* OriImg, int Width, int Height);
            static void ComputeGradients(
                float* Gradients, uint8_t* GradDires, const uint8_t* BlurredImage, int Width, int Height);
        };  // namespace A512
    }  // namespace AVX
}  // namespace SIMD