#include <bits/stdc++.h>
#include <filesystem>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>
#include "Solutions.h"

namespace fs = std::filesystem;
using ns     = std::chrono::nanoseconds;
using hrClk  = std::chrono::high_resolution_clock;
using Str    = std::string;
template <typename T>
using Vec = std::vector<T>;

int main()
{
    int n;
    int Choice;

    std::cout << "Max thread nums: " << std::thread::hardware_concurrency() << ".\n";
    Vec<std::tuple<Str, std::pair<int, int>, double>> ImageStatistics;

    Str           ImgPath      = "../Images/";
    Str           OutputPath   = "../Output/";
    Str           GaussianPath = "../GaussianImg/";
    const int     LowThre      = 30;
    const int     HighThre     = 90;
    std::ofstream CSV("Result.csv");
    CSV << "Image Name,Width x Height,Average Processing Time (ns)\n";

    std::function<void(uint8_t*, const uint8_t*, int, int)>         Gauss  = Serial::PerformGaussianBlur;
    std::function<void(float*, uint8_t*, const uint8_t*, int, int)> Grad   = Serial::ComputeGradients;
    std::function<void(float*, float*, uint8_t*, int, int)>         ReduNM = Serial::ReduceNonMaximum;
    std::function<void(uint8_t*, float*, int, int, int, int)>       DbThre = Serial::PerformDoubleThresholding;
    std::function<void(uint8_t*, uint8_t*, int, int)>               EdgeHy = Serial::PerformEdgeHysteresis;

    std::cout << "Enter the number to select the algorithm:\n"
                 "1. Serial\n"
                 "2. SSE\n"
                 "3. AVX256\n"
                 "4. AVX512\n";
    std::cin >> Choice;
    switch (Choice)
    {
    case 1:
        Gauss  = Serial::PerformGaussianBlur;
        break;
    case 2:
        Gauss  = SIMD::SSE::PerformGaussianBlur;
        break;
    case 3:
        Gauss  = SIMD::AVX::A256::PerformGaussianBlur;
        break;
    case 4:
        Gauss  = SIMD::AVX::A512::PerformGaussianBlur;
        break;
    default:
        break;
    }

    std::cout << "Please enter the number of iterations per image: ";
    std::cin >> n;

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

            cv::Mat GreyImg, EdgedImg, GaussianImg;
            cv::cvtColor(OriImg, GreyImg, cv::COLOR_BGR2GRAY);
            int OriginalWidth  = GreyImg.cols;
            int OriginalHeight = GreyImg.rows;
            int TmpWidth       = GreyImg.cols < 16 ? 16 : GreyImg.cols / 16 * 16;
            int TmpHeight      = GreyImg.rows < 16 ? 16 : GreyImg.rows / 16 * 16;
            cv::resize(GreyImg, GreyImg, cv::Size(TmpWidth, TmpHeight));
            EdgedImg.create(GreyImg.size(), CV_8UC1);
            GaussianImg.create(GreyImg.size(), CV_8UC1);

            std::vector<float>   GradientPixels(GreyImg.total());
            std::vector<float>   MatrixPixels(GreyImg.total());
            std::vector<uint8_t> SegmentPixels(GreyImg.total());
            std::vector<uint8_t> DoubleThrePixels(GreyImg.total());
            std::vector<ns>      Durations;
            for (int i = 0; i < n; ++i)
            {
                auto Start = hrClk::now();
                Gauss(GaussianImg.data, GreyImg.data, GreyImg.cols, GreyImg.rows);
                auto End = hrClk::now();
                // Grad(GradientPixels.data(), SegmentPixels.data(), GaussianImg.data, GreyImg.cols, GreyImg.rows);
                // ReduNM(MatrixPixels.data(), GradientPixels.data(), SegmentPixels.data(), GreyImg.cols, GreyImg.rows);
                // DbThre(DoubleThrePixels.data(), MatrixPixels.data(), HighThre, LowThre, GreyImg.cols, GreyImg.rows);
                // EdgeHy(EdgedImg.data, DoubleThrePixels.data(), GreyImg.cols, GreyImg.rows);

                Durations.push_back(std::chrono::duration_cast<ns>(End - Start));
            }

            ns AvgTime =
                std::accumulate(Durations.begin(), Durations.end(), ns(0), [](ns a, ns b) { return a + b; }) / n;

            cv::imwrite(GaussianPath + Entry.path().filename().string(), GaussianImg);
            /*
            cv.resize(EdgedImg, EdgedImg, cv::Size(OriginalWidth, OriginalHeight));
            if (!cv::imwrite(OutputPath + Entry.path().filename().string(), EdgedImg))
            {
                std::cerr << "Failed to save the image: " << OutputPath + Entry.path().filename().string() << std::endl;
            }
            else { std::cout << "Final image saved to " << OutputPath + Entry.path().filename().string() << std::endl; }
            */

            std::cout << "Processed " << Entry.path().filename().string() << " (" << GreyImg.cols << "x" << GreyImg.rows
                      << "): " << AvgTime.count() << "ns" << std::endl;
            CSV << Entry.path().filename().string() << "," << GreyImg.cols << "x" << GreyImg.rows << ","
                << AvgTime.count() << "\n";

            Durations.clear();
        }
    }
    CSV.close();
}