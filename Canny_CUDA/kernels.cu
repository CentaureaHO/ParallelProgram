#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <queue>

__constant__ float GaussianKernel_1D[3] = {1, 2, 1};

__global__ void GaussianBlurHorizontal(uint8_t* Output, const uint8_t* OriImg, int Width, int Height, int Offset)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < Width && y < Height)
    {
        float PixelVal = 0, KernelSum = 0;
        for (int i = -Offset; i <= Offset; i++)
        {
            int ReadX = x + i;
            if (ReadX >= 0 && ReadX < Width)
            {
                PixelVal += GaussianKernel_1D[i + Offset] * OriImg[y * Width + ReadX];
                KernelSum += GaussianKernel_1D[i + Offset];
            }
        }
        Output[y * Width + x] = (uint8_t)(PixelVal / KernelSum);
    }
}

__global__ void GaussianBlurVertical(uint8_t* Output, const uint8_t* TempImg, int Width, int Height, int Offset)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < Width && y < Height)
    {
        float PixelVal = 0, KernelSum = 0;
        for (int j = -Offset; j <= Offset; j++)
        {
            int ReadY = y + j;
            if (ReadY >= 0 && ReadY < Height)
            {
                PixelVal += GaussianKernel_1D[j + Offset] * TempImg[ReadY * Width + x];
                KernelSum += GaussianKernel_1D[j + Offset];
            }
        }
        Output[y * Width + x] = (uint8_t)(PixelVal / KernelSum);
    }
}

__global__ void ComputeGradientsKernel(
    float* Gradients, uint8_t* GradDires, const uint8_t* BlurredImage, int Width, int Height)
{
    static const int8_t Gx[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    static const int8_t Gy[] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
    int                 x    = blockIdx.x * blockDim.x + threadIdx.x;
    int                 y    = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < Width - 1 && y > 0 && y < Height - 1)
    {
        float GradX = 0.0, GradY = 0.0;
        int   KernelIdx = 0, PixelIdx = x + (y * Width);

        for (int ky = -1; ky <= 1; ky++)
        {
            for (int kx = -1; kx <= 1; kx++)
            {
                GradX += BlurredImage[PixelIdx + (kx + (ky * Width))] * Gx[KernelIdx];
                GradY += BlurredImage[PixelIdx + (kx + (ky * Width))] * Gy[KernelIdx];
                KernelIdx++;
            }
        }

        int Dire = 0;
        if (GradX == 0.0 || GradY == 0.0) { Gradients[PixelIdx] = 0; }
        else
        {
            Gradients[PixelIdx] = (sqrt((GradX * GradX) + (GradY * GradY)));
            float Angle         = atan2(GradY, GradX) * (360.0 / (2.0 * M_PI));

            if ((Angle <= 22.5 && Angle >= -22.5) || (Angle <= -157.5) || (Angle >= 157.5))
                Dire = 1;
            else if ((Angle > 22.5 && Angle <= 67.5) || (Angle > -157.5 && Angle <= -112.5))
                Dire = 2;
            else if ((Angle > 67.5 && Angle <= 112.5) || (Angle >= -112.5 && Angle < -67.5))
                Dire = 3;
            else if ((Angle >= -67.5 && Angle < -22.5) || (Angle > 112.5 && Angle < 157.5))
                Dire = 4;
            else
                printf("error %f\n", Angle);
        }
        GradDires[PixelIdx] = (uint8_t)Dire;
    }
}

__global__ void ReduceNonMaximumKernel(float* Magnitudes, float* Gradients, uint8_t* Direction, int Width, int Height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < Width - 1 && y > 0 && y < Height - 1)
    {
        int Pos         = x + (y * Width);
        Magnitudes[Pos] = Gradients[Pos];
        switch (Direction[Pos])
        {
            case 1:
                if (Gradients[Pos - 1] >= Gradients[Pos] || Gradients[Pos + 1] > Gradients[Pos]) Magnitudes[Pos] = 0;
                break;
            case 2:
                if (Gradients[Pos - (Width - 1)] >= Gradients[Pos] || Gradients[Pos + (Width - 1)] > Gradients[Pos])
                    Magnitudes[Pos] = 0;
                break;
            case 3:
                if (Gradients[Pos - Width] >= Gradients[Pos] || Gradients[Pos + Width] > Gradients[Pos])
                    Magnitudes[Pos] = 0;
                break;
            case 4:
                if (Gradients[Pos - (Width + 1)] >= Gradients[Pos] || Gradients[Pos + (Width + 1)] > Gradients[Pos])
                    Magnitudes[Pos] = 0;
                break;
            default: Magnitudes[Pos] = 0; break;
        }
    }
}

__global__ void PerformDoubleThresholdingKernel(
    uint8_t* EdgedImg, float* Magnitudes, int HighThre, int LowThre, int Width, int Height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < Width && y < Height)
    {
        int PixelIdx = x + (y * Width);
        if (Magnitudes[PixelIdx] > HighThre)
            EdgedImg[PixelIdx] = 255;
        else if (Magnitudes[PixelIdx] > LowThre)
            EdgedImg[PixelIdx] = 100;
        else
            EdgedImg[PixelIdx] = 0;
    }
}

__global__ void PerformEdgeHysteresisKernel(uint8_t* EdgedImg, uint8_t* InitialEdges, int Width, int Height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int PixelIdx = x + y * Width;
    if (InitialEdges[PixelIdx] == 100)
    {
        bool hasStrongNeighbor =
            (InitialEdges[PixelIdx - 1] == 255 || InitialEdges[PixelIdx + 1] == 255 ||
                InitialEdges[PixelIdx - Width] == 255 || InitialEdges[PixelIdx + Width] == 255 ||
                InitialEdges[PixelIdx - Width - 1] == 255 || InitialEdges[PixelIdx - Width + 1] == 255 ||
                InitialEdges[PixelIdx + Width - 1] == 255 || InitialEdges[PixelIdx + Width + 1] == 255);
        if (hasStrongNeighbor) { EdgedImg[PixelIdx] = 255; }
        else { EdgedImg[PixelIdx] = 0; }
    }
}

void PerformCannyEdgeDetection(uint8_t* Output, const uint8_t* OriImg, int Width, int Height)
{
    static const int Offset = 1;
    uint8_t*         d_OriImg;
    uint8_t*         d_TempImg;
    uint8_t*         d_Output;
    float*           d_Gradients;
    uint8_t*         d_GradDires;
    float*           d_Magnitudes;

    cudaMalloc(&d_OriImg, Width * Height * sizeof(uint8_t));
    cudaMalloc(&d_TempImg, Width * Height * sizeof(uint8_t));
    cudaMalloc(&d_Output, Width * Height * sizeof(uint8_t));
    cudaMalloc(&d_Gradients, Width * Height * sizeof(float));
    cudaMalloc(&d_GradDires, Width * Height * sizeof(uint8_t));
    cudaMalloc(&d_Magnitudes, Width * Height * sizeof(float));

    cudaMemcpy(d_OriImg, OriImg, Width * Height * sizeof(uint8_t), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((Width + blockSize.x - 1) / blockSize.x, (Height + blockSize.y - 1) / blockSize.y);

    GaussianBlurHorizontal<<<gridSize, blockSize>>>(d_TempImg, d_OriImg, Width, Height, Offset);
    cudaDeviceSynchronize();

    GaussianBlurVertical<<<gridSize, blockSize>>>(d_Output, d_TempImg, Width, Height, Offset);
    cudaDeviceSynchronize();

    ComputeGradientsKernel<<<gridSize, blockSize>>>(d_Gradients, d_GradDires, d_Output, Width, Height);
    cudaDeviceSynchronize();

    ReduceNonMaximumKernel<<<gridSize, blockSize>>>(d_Magnitudes, d_Gradients, d_GradDires, Width, Height);
    cudaDeviceSynchronize();

    PerformDoubleThresholdingKernel<<<gridSize, blockSize>>>(d_Output, d_Magnitudes, 90, 30, Width, Height);
    cudaDeviceSynchronize();

    PerformEdgeHysteresisKernel<<<gridSize, blockSize>>>(d_Output, d_Output, Width, Height);
    cudaDeviceSynchronize();

    cudaMemcpy(Output, d_Output, Width * Height * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    cudaFree(d_OriImg);
    cudaFree(d_TempImg);
    cudaFree(d_Output);
    cudaFree(d_Gradients);
    cudaFree(d_GradDires);
    cudaFree(d_Magnitudes);
}
