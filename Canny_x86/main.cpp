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
    Canny* Grad  = &PThreadP_Ins;
    Canny* Redu  = &SIMD_Ins;
    Canny* DouTh = &PThreadP_Ins;
    Canny* Edged = &PThreadP_Ins;

    std::ofstream CSV("Record.csv");
    CSV << "Image Name,Width x Height,Gaussian Time (ns),Gradient Time (ns),Reduce Time (ns),Double Threshold Time "
           "(ns),Edge Hysteresis Time (ns),Total Processing Time (ns)\n";
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
            int OriginalWidth  = GreyImg.cols;
            int OriginalHeight = GreyImg.rows;
            int TmpWidth       = GreyImg.cols < 16 ? 16 : GreyImg.cols / 16 * 16;
            int TmpHeight      = GreyImg.rows < 16 ? 16 : GreyImg.rows / 16 * 16;
            cv::resize(GreyImg, GreyImg, cv::Size(TmpWidth, TmpHeight));
            EdgedImg.create(GreyImg.size(), CV_8UC1);

            uint8_t* GreyImageArray     = (uint8_t*)_mm_malloc(TmpWidth * TmpHeight * sizeof(uint8_t), 64);
            uint8_t* GaussianImageArray = (uint8_t*)_mm_malloc(TmpWidth * TmpHeight * sizeof(uint8_t), 64);
            float*   GradientPixels     = (float*)_mm_malloc(TmpWidth * TmpHeight * sizeof(float), 64);
            float*   MatrixPixels       = (float*)_mm_malloc(TmpWidth * TmpHeight * sizeof(float), 64);
            uint8_t* SegmentPixels      = (uint8_t*)_mm_malloc(TmpWidth * TmpHeight * sizeof(uint8_t), 64);
            uint8_t* DoubleThrePixels   = (uint8_t*)_mm_malloc(TmpWidth * TmpHeight * sizeof(uint8_t), 64);
            uint8_t* EdgeArray          = (uint8_t*)_mm_malloc(TmpWidth * TmpHeight * sizeof(uint8_t), 64);

            memcpy(GreyImageArray, GreyImg.data, TmpWidth * TmpHeight * sizeof(uint8_t));

            for (int i = 0; i < 10; ++i)
                OMP_Ins.PerformGaussianBlur(GaussianImageArray, GreyImageArray, GreyImg.cols, GreyImg.rows);

            ns TotalGaussianTime        = ns(0);
            ns TotalGradientTime        = ns(0);
            ns TotalReduceTime          = ns(0);
            ns TotalDoubleThresholdTime = ns(0);
            ns TotalEdgeHysteresisTime  = ns(0);
            ns TotalProcessingTime      = ns(0);

            for (int i = 0; i < n; ++i)
            {
                auto start = hrClk::now();
                Gauss->PerformGaussianBlur(GaussianImageArray, GreyImageArray, GreyImg.cols, GreyImg.rows);
                auto end = hrClk::now();
                TotalGaussianTime += std::chrono::duration_cast<ns>(end - start);

                start = hrClk::now();
                Grad->ComputeGradients(GradientPixels, SegmentPixels, GaussianImageArray, GreyImg.cols, GreyImg.rows);
                end = hrClk::now();
                TotalGradientTime += std::chrono::duration_cast<ns>(end - start);

                start = hrClk::now();
                Redu->ReduceNonMaximum(MatrixPixels, GradientPixels, SegmentPixels, GreyImg.cols, GreyImg.rows);
                end = hrClk::now();
                TotalReduceTime += std::chrono::duration_cast<ns>(end - start);

                start = hrClk::now();
                DouTh->PerformDoubleThresholding(
                    DoubleThrePixels, MatrixPixels, HighThre, LowThre, GreyImg.cols, GreyImg.rows);
                end = hrClk::now();
                TotalDoubleThresholdTime += std::chrono::duration_cast<ns>(end - start);

                start = hrClk::now();
                Edged->PerformEdgeHysteresis(EdgeArray, DoubleThrePixels, GreyImg.cols, GreyImg.rows);
                end = hrClk::now();
                TotalEdgeHysteresisTime += std::chrono::duration_cast<ns>(end - start);

                TotalProcessingTime += (TotalGaussianTime + TotalGradientTime + TotalReduceTime +
                                        TotalDoubleThresholdTime + TotalEdgeHysteresisTime);
            }
            memcpy(EdgedImg.data, EdgeArray, TmpWidth * TmpHeight * sizeof(uint8_t));
            _mm_free(GreyImageArray);
            _mm_free(GaussianImageArray);
            _mm_free(GradientPixels);
            _mm_free(MatrixPixels);
            _mm_free(SegmentPixels);
            _mm_free(DoubleThrePixels);
            _mm_free(EdgeArray);
            GreyImageArray     = nullptr;
            GaussianImageArray = nullptr;
            GradientPixels     = nullptr;
            MatrixPixels       = nullptr;
            SegmentPixels      = nullptr;
            DoubleThrePixels   = nullptr;
            EdgeArray          = nullptr;

            ns avgGaussianTime        = TotalGaussianTime / n;
            ns avgGradientTime        = TotalGradientTime / n;
            ns avgReductionTime       = TotalReduceTime / n;
            ns avgDoubleThresholdTime = TotalDoubleThresholdTime / n;
            ns avgEdgeHysteresisTime  = TotalEdgeHysteresisTime / n;
            ns avgTotalProcessingTime = TotalProcessingTime / n;

            cv::resize(EdgedImg, EdgedImg, cv::Size(OriginalWidth, OriginalHeight));
            if (!cv::imwrite(OutputPath + Entry.path().filename().string(), EdgedImg))
            {
                std::cerr << "Failed to save the image: " << OutputPath + Entry.path().filename().string() << std::endl;
            }
            else { std::cout << "Final image saved to " << OutputPath + Entry.path().filename().string() << std::endl; }

            std::cout << "Processed " << Entry.path().filename().string() << " (" << GreyImg.cols << "x" << GreyImg.rows
                      << "): " << avgTotalProcessingTime.count() << "ns" << std::endl;

            CSV << Entry.path().filename().string() << "," << GreyImg.cols << "x" << GreyImg.rows << ","
                << avgGaussianTime.count() << "," << avgGradientTime.count() << "," << avgReductionTime.count() << ","
                << avgDoubleThresholdTime.count() << "," << avgEdgeHysteresisTime.count() << ","
                << avgTotalProcessingTime.count() << "\n";
        }
    }
    std::cout << "Thread " << UseThread << " done.\n\n";
    CSV.close();
}
