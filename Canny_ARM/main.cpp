#include <bits/stdc++.h>
#include <filesystem>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "Solutions.h"

namespace fs = std::filesystem;
using ns     = std::chrono::nanoseconds;
using hrClk  = std::chrono::high_resolution_clock;
using Str    = std::string;
template <typename T>
using Vec = std::vector<T>;

int main()
{
    Vec<std::tuple<Str, std::pair<int, int>, ns>> Cases;

    Str       ImgPath      = "../Images/";
    Str       OutputPath   = "../Output/";
    Str       GaussianPath = "../GaussianImg/";
    const int LowThre      = 30;
    const int HighThre     = 90;

    std::function<void(uint8_t*, const uint8_t*, int, int)>         Gauss  = AVX::A512::PerformGaussianBlur;
    std::function<void(float*, uint8_t*, const uint8_t*, int, int)> Grad   = Serial::ComputeGradients;
    std::function<void(float*, float*, uint8_t*, int, int)>         ReduNM = Serial::ReduceNonMaximum;
    std::function<void(uint8_t*, float*, int, int, int, int)>       DbThre = Serial::PerformDoubleThresholding;
    std::function<void(uint8_t*, uint8_t*, int, int)>               EdgeHy = Serial::PerformEdgeHysteresis;

    for (const auto& Entry : fs::directory_iterator(ImgPath))
    {
        if (Entry.path().extension() == ".jpg")
        {
            cv::Mat OriImg = cv::imread(Entry.path().string(), cv::IMREAD_COLOR);
            if (OriImg.empty())
            {
                std::cerr << "Error loading the image: " << Entry.path().string() << std::endl;
                continue;
            }

            cv::Mat GreyImg, EdgedImg;
            cv::cvtColor(OriImg, GreyImg, cv::COLOR_BGR2GRAY);

            const int Width  = GreyImg.cols;
            const int Height = GreyImg.rows;
            EdgedImg.create(Height, Width, CV_8UC1);

            std::vector<float>   GradientPixels(Width * Height);
            std::vector<float>   MatrixPixels(Width * Height);
            std::vector<uint8_t> SegmentPixels(Width * Height);
            std::vector<uint8_t> DoubleThrePixels(Width * Height);

            auto Start = hrClk::now();
            Gauss(EdgedImg.data, GreyImg.data, Width, Height);
            auto End = hrClk::now();
            cv::imwrite(GaussianPath + Entry.path().filename().string(), EdgedImg);
            Grad(GradientPixels.data(), SegmentPixels.data(), EdgedImg.data, Width, Height);
            ReduNM(MatrixPixels.data(), GradientPixels.data(), SegmentPixels.data(), Width, Height);
            DbThre(DoubleThrePixels.data(), MatrixPixels.data(), HighThre, LowThre, Width, Height);
            EdgeHy(EdgedImg.data, DoubleThrePixels.data(), Width, Height);

            ns Duration = std::chrono::duration_cast<ns>(End - Start);
            Cases.push_back(std::make_tuple(Entry.path().filename().string(), std::make_pair(Width, Height), Duration));

            if (!cv::imwrite(OutputPath + Entry.path().filename().string(), EdgedImg))
            {
                std::cerr << "Failed to save the image: " << OutputPath + Entry.path().filename().string() << std::endl;
            }
            else { std::cout << "Image saved to " << OutputPath + Entry.path().filename().string() << std::endl; }

            std::cout << "Processed " << Entry.path().filename().string() << " (" << Width << "x" << Height
                      << "): " << Duration.count() << "ns\n";
        }
    }
}