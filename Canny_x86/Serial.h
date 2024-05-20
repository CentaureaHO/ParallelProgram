#pragma once
#include <cstdint>
#include "CannyBase.h"

namespace Serial
{
    void PerformGaussianBlur(uint8_t* Output, const uint8_t* OriImg, int Width, int Height);
    void ComputeGradients(float* Gradients, uint8_t* GradDires, const uint8_t* BlurredImage, int Width, int Height);
    void ReduceNonMaximum(float* Magnitudes, float* Gradients, uint8_t* Direction, int Width, int Height);
    void PerformDoubleThresholding(
        uint8_t* EdgedImg, float* Magnitudes, int HighThre, int LowThre, int Width, int Height);
    void PerformEdgeHysteresis(uint8_t* EdgedImg, uint8_t* InitialEdges, int Width, int Height);
}  // namespace Serial