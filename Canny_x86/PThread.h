#pragma once
#include <cstdint>

namespace PThread
{
    void ComputeGradients(float* Gradients, uint8_t* GradDires, const uint8_t* BlurredImage, int Width, int Height);
}