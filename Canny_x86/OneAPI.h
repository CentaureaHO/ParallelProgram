#pragma once
#include <cstdint>
#include "CannyBase.h"
#include "TbbPool.h"

class OneAPI : public Canny
{
  public:
    static OneAPI& GetInstance(unsigned int TN = 16);

    void PerformGaussianBlur(uint8_t* Output, const uint8_t* OriImg, int Width, int Height) override;
    void ComputeGradients(
        float* Gradients, uint8_t* GradDires, const uint8_t* BlurredImage, int Width, int Height) override;
    void ReduceNonMaximum(float* Magnitudes, float* Gradients, uint8_t* Direction, int Width, int Height) override;
    void PerformDoubleThresholding(
        uint8_t* EdgedImg, float* Magnitudes, int HighThre, int LowThre, int Width, int Height) override;
    void PerformEdgeHysteresis(uint8_t* EdgedImg, uint8_t* InitialEdges, int Width, int Height) override;

  private:
    OneAPI(unsigned int TN);
    ~OneAPI();

    int ThreadNum;
};