#include <bits/stdc++.h>
#include <filesystem>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>
#include <memory>
#include "Solutions.h"
#include <omp.h>
#include "CannyBase.h"

namespace fs = std::filesystem;
using ns     = std::chrono::nanoseconds;
using hrClk  = std::chrono::high_resolution_clock;
using Str    = std::string;
template <typename T>
using Vec = std::vector<T>;

int main()
{
    int n         = 1;
    int UseThread = 16;

    Str       ImgPath    = "../Images/";
    Str       OutputPath = "../Output/";
    const int LowThre    = 30;
    const int HighThre   = 90;

    Serial&          Serial_Ins   = Serial::GetInstance();
    SIMD::AVX::A512& SIMD_Ins     = SIMD::AVX::A512::GetInstance();
    PThread&         PThread_Ins  = PThread::GetInstance(UseThread);
    PThreadWithPool& PThreadP_Ins = PThreadWithPool::GetInstance(UseThread);
    OpenMP&          OMP_Ins      = OpenMP::GetInstance(UseThread);

    Canny* Gauss = &SIMD_Ins;
    Canny* Grad  = &SIMD_Ins;
    Canny* Redu  = &SIMD_Ins;
    Canny* DouTh = &SIMD_Ins;
    Canny* Edged = &SIMD_Ins;

    std::ofstream CSV("Record.csv");
    CSV << "Image Name,Width x Height,Average Processing Time (ns)\n";
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

            cv::Mat GreyImg, GaussianImg, EdgedImg;
            cv::cvtColor(OriImg, GreyImg, cv::COLOR_BGR2GRAY);
            int OriginalWidth  = GreyImg.cols;
            int OriginalHeight = GreyImg.rows;
            int TmpWidth       = GreyImg.cols < 16 ? 16 : GreyImg.cols / 16 * 16;
            int TmpHeight      = GreyImg.rows < 16 ? 16 : GreyImg.rows / 16 * 16;
            cv::resize(GreyImg, GreyImg, cv::Size(TmpWidth, TmpHeight));
            GaussianImg.create(GreyImg.size(), CV_8UC1);
            EdgedImg.create(GreyImg.size(), CV_8UC1);
            cv::resize(GaussianImg, GaussianImg, cv::Size(TmpWidth, TmpHeight));

            float*          GradientPixels   = (float*)_mm_malloc(TmpWidth * TmpHeight * sizeof(float), 64);
            float*          MatrixPixels     = (float*)_mm_malloc(TmpWidth * TmpHeight * sizeof(float), 64);
            uint8_t*        SegmentPixels    = (uint8_t*)_mm_malloc(TmpWidth * TmpHeight * sizeof(uint8_t), 64);
            uint8_t*        DoubleThrePixels = (uint8_t*)_mm_malloc(TmpWidth * TmpHeight * sizeof(uint8_t), 64);
            std::vector<ns> Durations;

            // 初始化omp线程池
            for (int i = 0; i < 10; ++i)
                OMP_Ins.PerformGaussianBlur(GaussianImg.data, GreyImg.data, GreyImg.cols, GreyImg.rows);

            // 以下部分为实际测试
            for (int i = 0; i < n; ++i)
            {

                Gauss->PerformGaussianBlur(GaussianImg.data, GreyImg.data, GreyImg.cols, GreyImg.rows);

                Grad->ComputeGradients(GradientPixels, SegmentPixels, GaussianImg.data, GreyImg.cols, GreyImg.rows);
                auto Start = hrClk::now();
                Redu->ReduceNonMaximum(MatrixPixels, GradientPixels, SegmentPixels, GreyImg.cols, GreyImg.rows);
                auto End = hrClk::now();
                DouTh->PerformDoubleThresholding(
                    DoubleThrePixels, MatrixPixels, HighThre, LowThre, GreyImg.cols, GreyImg.rows);

                Edged->PerformEdgeHysteresis(EdgedImg.data, DoubleThrePixels, GreyImg.cols, GreyImg.rows);

                Durations.push_back(std::chrono::duration_cast<ns>(End - Start));
            }

            _mm_free(GradientPixels);
            _mm_free(MatrixPixels);
            _mm_free(SegmentPixels);
            _mm_free(DoubleThrePixels);
            GradientPixels   = nullptr;
            MatrixPixels     = nullptr;
            SegmentPixels    = nullptr;
            DoubleThrePixels = nullptr;

            ns AvgTime =
                std::accumulate(Durations.begin(), Durations.end(), ns(0), [](ns a, ns b) { return a + b; }) / n;

            cv::resize(EdgedImg, EdgedImg, cv::Size(OriginalWidth, OriginalHeight));
            if (!cv::imwrite(OutputPath + Entry.path().filename().string(), EdgedImg))
            {
                std::cerr << "Failed to save the image: " << OutputPath + Entry.path().filename().string() << std::endl;
            }
            else { std::cout << "Final image saved to " << OutputPath + Entry.path().filename().string() << std::endl; }

            std::cout << "Processed " << Entry.path().filename().string() << " (" << GreyImg.cols << "x" << GreyImg.rows
                      << "): " << AvgTime.count() << "ns" << std::endl;
            CSV << Entry.path().filename().string() << "," << GreyImg.cols << "x" << GreyImg.rows << ","
                << AvgTime.count() << "\n";
            Durations.clear();
        }
    }
    std::cout << "Thread " << UseThread << " done.\n\n";
    CSV.close();
}