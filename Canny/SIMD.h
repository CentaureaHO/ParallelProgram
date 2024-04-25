#pragma once
#include <cstdint>

namespace SSE
{
    void PerformGaussianBlur(uint8_t* Output, const uint8_t* OriImg, int Width, int Height);
}

namespace AVX
{
    void PerformGaussianBlur(uint8_t* Output, const uint8_t* OriImg, int Width, int Height);
}