#pragma once
#include <cstdint>
#include "ThreadPool.h"

class PThread
{
  public:
    static PThread& GetInstabce();
    void            PerformGaussianBlur(uint8_t* Output, const uint8_t* OriImg, int Width, int Height);
    void ComputeGradients(float* Gradients, uint8_t* GradDires, const uint8_t* BlurredImage, int Width, int Height);

  private:
    PThread();
    ~PThread();

    ThreadPool       Pool;
    static const int ThreadNum = 16;
};