#pragma once
#include <cstdint>
#include "CannyBase.h"
#include "ThreadPool.h"

class PThread : public Canny
{
  public:
    static PThread& GetInstance(unsigned int TN = 16);

    void PerformGaussianBlur(uint8_t* Output, const uint8_t* OriImg, int Width, int Height) override;
    void ComputeGradients(
        float* Gradients, uint8_t* GradDires, const uint8_t* BlurredImage, int Width, int Height) override;

  private:
    PThread(unsigned int TN);
    ~PThread();

    int ThreadNum;
};

class PThreadWithPool : public Canny
{
  public:
    static PThreadWithPool& GetInstance(unsigned int TN = 16);

    void PerformGaussianBlur(uint8_t* Output, const uint8_t* OriImg, int Width, int Height) override;
    void ComputeGradients(
        float* Gradients, uint8_t* GradDires, const uint8_t* BlurredImage, int Width, int Height) override;

  private:
    PThreadWithPool(unsigned int TN);
    ~PThreadWithPool();

    ThreadPool Pool;
    int        ThreadNum;
};
