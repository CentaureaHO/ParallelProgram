#pragma once
#include <cstdint>

namespace NEON
{
    void PerformGaussianBlur(uint8_t* Output, const uint8_t* OriImg, int Width, int Height);
}
