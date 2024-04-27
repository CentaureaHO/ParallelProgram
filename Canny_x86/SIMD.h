#pragma once
#include <cstdint>

namespace SIMD
{
    namespace SSE
    {
        void PerformGaussianBlur(uint8_t* Output, const uint8_t* OriImg, int Width, int Height);
    }

    namespace AVX
    {
        namespace A256
        {
            void PerformGaussianBlur(uint8_t* Output, const uint8_t* OriImg, int Width, int Height);
        }

        namespace A512
        {
            void PerformGaussianBlur(uint8_t* Output, const uint8_t* OriImg, int Width, int Height);
        }
    }  // namespace AVX
}  // namespace SIMD

namespace ThSIMD
{
    namespace SSE
    {
        void PerformGaussianBlur(uint8_t* Output, const uint8_t* OriImg, int Width, int Height);
    }

    namespace AVX
    {
        namespace A256
        {
            void PerformGaussianBlur(uint8_t* Output, const uint8_t* OriImg, int Width, int Height);
        }

        namespace A512
        {
            void PerformGaussianBlur(uint8_t* Output, const uint8_t* OriImg, int Width, int Height);
        }
    }  // namespace AVX
}  // namespace ThSIMD