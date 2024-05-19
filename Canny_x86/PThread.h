#pragma once
#include <cstdint>

class PThread
{
  public:
    static void PerformGaussianBlur(uint8_t* Output, const uint8_t* OriImg, int Width, int Height);
    static void ComputeGradients(
        float* Gradients, uint8_t* GradDires, const uint8_t* BlurredImage, int Width, int Height);
};