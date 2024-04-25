#include "Serial.h"
#include "GaussDef.h"
#include <cmath>
#include <iostream>
#include <cstring>
const int KernelSize = 3;

void Serial::PerformGaussianBlur(uint8_t* Output, const uint8_t* OriImg, int Width, int Height)
{
    const int Offset = ((KernelSize - 1) / 2);

    for (int PixNum = 0; PixNum < Height * Width; PixNum++)
    {
        float KernelSum = 0, PixelVal = 0;
        for (int i = 0; i < KernelSize; ++i)
            for (int j = 0; j < KernelSize; ++j)
                if (((PixNum + ((i - Offset) * Width) + j - Offset) >= 0) &&
                    ((PixNum + ((i - Offset) * Width) + j - Offset) <= Height * Width - 1) &&
                    (((PixNum % Width) + j - Offset) >= 0) && (((PixNum % Width) + j - Offset) <= (Width - 1)))
                {
                    PixelVal +=
                        GaussianKernel_2D[i * KernelSize + j] * OriImg[PixNum + ((i - Offset) * Width) + j - Offset];
                    KernelSum += GaussianKernel_2D[i * KernelSize + j];
                }
        Output[PixNum] = (uint8_t)(PixelVal / KernelSum);
    }
}

void Serial::ComputeGradients(float* Gradients, uint8_t* GradDires, const uint8_t* BlurredImage, int Width, int Height)
{
    const int8_t Gx[]   = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    const int8_t Gy[]   = {1, 2, 1, 0, 0, 0, -1, -2, -1};
    int          Offset = 1;
    for (int x = Offset; x < Width - Offset; x++)
    {
        for (int y = Offset; y < Height - Offset; y++)
        {
            float GradX = 0.0, GradY = 0.0;
            int   KernelIdx = 0, PixelIdx = x + (y * Width);

            for (int ky = -Offset; ky <= Offset; ky++)
            {
                for (int kx = -Offset; kx <= Offset; kx++)
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
                Gradients[PixelIdx] = (std::sqrt((GradX * GradX) + (GradY * GradY)));
                float Angle         = std::atan2(GradY, GradX) * (360.0 / (2.0 * M_PI));

                if ((Angle <= 22.5 && Angle >= -22.5) || (Angle <= -157.5) || (Angle >= 157.5))
                    Dire = 1;
                else if ((Angle > 22.5 && Angle <= 67.5) || (Angle > -157.5 && Angle <= -112.5))
                    Dire = 2;
                else if ((Angle > 67.5 && Angle <= 112.5) || (Angle >= -112.5 && Angle < -67.5))
                    Dire = 3;
                else if ((Angle >= -67.5 && Angle < -22.5) || (Angle > 112.5 && Angle < 157.5))
                    Dire = 4;
                else
                    std::cerr << "error " << Angle << std::endl;
            }
            GradDires[PixelIdx] = (uint8_t)Dire;
        }
    }
}

void Serial::ReduceNonMaximum(float* Magnitudes, float* Gradients, uint8_t* Direction, int Width, int Height)
{
    memcpy(Magnitudes, Gradients, Width * Height * sizeof(float));
    for (int x = 1; x < Width - 1; x++)
    {
        for (int y = 1; y < Height - 1; y++)
        {
            int Pos = x + (y * Width);
            switch (Direction[Pos])
            {
                case 1:
                    if (Gradients[Pos - 1] >= Gradients[Pos] || Gradients[Pos + 1] > Gradients[Pos])
                        Magnitudes[Pos] = 0;
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
}

void Serial::PerformDoubleThresholding(
    uint8_t* EdgedImg, float* Magnitudes, int HighThre, int LowThre, int Width, int Height)
{
    for (int x = 0; x < Width; x++)
    {
        for (int y = 0; y < Height; y++)
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
}

void Serial::PerformEdgeHysteresis(uint8_t* EdgedImg, uint8_t* InitialEdges, int Width, int Height)
{
    memcpy(EdgedImg, InitialEdges, Width * Height * sizeof(uint8_t));
    for (int x = 1; x < Width - 1; x++)
    {
        for (int y = 1; y < Height - 1; y++)
        {
            int PixelIdx = x + (y * Width);
            if (InitialEdges[PixelIdx] == 100)
            {
                if (InitialEdges[PixelIdx - 1] == 255 || InitialEdges[PixelIdx + 1] == 255 ||
                    InitialEdges[PixelIdx - Width] == 255 || InitialEdges[PixelIdx + Width] == 255 ||
                    InitialEdges[PixelIdx - Width - 1] == 255 || InitialEdges[PixelIdx - Width + 1] == 255 ||
                    InitialEdges[PixelIdx + Width - 1] == 255 || InitialEdges[PixelIdx + Width + 1] == 255)
                    EdgedImg[PixelIdx] = 255;
                else
                    EdgedImg[PixelIdx] = 0;
            }
        }
    }
}