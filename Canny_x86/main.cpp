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
    int Choice    = 0;
    std::cout << "Choose the method to run:\n"
              << "1. Serial\n"
              << "2. SIMD\n"
              << "3. PThread\n"
              << "4. PThread with Pool\n"
              << "5. OpenMP\n";
    std::cin >> Choice;

    std::cout << "Repeat the process how many times? ";
    std::cin >> n;

    std::cout << "How many threads to use? ";
    std::cin >> UseThread;

    Str       ImgPath    = "../Images/";
    Str       OutputPath = "../Output/";
    const int LowThre    = 30;
    const int HighThre   = 90;

    Serial&          Serial_Ins   = Serial::GetInstance();
    SIMD::AVX::A512& SIMD_Ins     = SIMD::AVX::A512::GetInstance();
    PThread&         PThread_Ins  = PThread::GetInstance(UseThread);
    PThreadWithPool& PThreadP_Ins = PThreadWithPool::GetInstance(UseThread);
    OpenMP&          OMP_Ins      = OpenMP::GetInstance(UseThread);

    Canny* Gauss = nullptr;
    Canny* Grad  = nullptr;
    Canny* Redu  = nullptr;
    Canny* DouTh = nullptr;
    Canny* Edged = nullptr;

    switch (Choice)
    {
        case 1:
            Gauss = &Serial_Ins;
            Grad  = &Serial_Ins;
            Redu  = &Serial_Ins;
            DouTh = &Serial_Ins;
            Edged = &Serial_Ins;
            break;
        case 2:
            Gauss = &SIMD_Ins;
            Grad  = &SIMD_Ins;
            Redu  = &SIMD_Ins;
            DouTh = &SIMD_Ins;
            Edged = &SIMD_Ins;
            break;
        case 3:
            Gauss = &PThread_Ins;
            Grad  = &PThread_Ins;
            Redu  = &PThread_Ins;
            DouTh = &PThread_Ins;
            Edged = &PThread_Ins;
            break;
        case 4:
            Gauss = &PThreadP_Ins;
            Grad  = &PThreadP_Ins;
            Redu  = &PThreadP_Ins;
            DouTh = &PThreadP_Ins;
            Edged = &PThreadP_Ins;
            break;
        case 5:
            Gauss = &OMP_Ins;
            Grad  = &OMP_Ins;
            Redu  = &OMP_Ins;
            DouTh = &OMP_Ins;
            Edged = &OMP_Ins;
            break;
        default: std::cerr << "Invalid choice.\n"; return 1;
    }

    std::string filename = "";
    switch (Choice)
    {
        case 1: filename = "Serial"; break;
        case 2: filename = "SIMD"; break;
        case 3: filename = "PThread_" + std::to_string(UseThread); break;
        case 4: filename = "PThreadPool_" + std::to_string(UseThread); break;
        case 5: filename = "OpenMP_" + std::to_string(UseThread); break;
    }

    filename = filename + ".csv";

    std::ofstream CSV(filename);
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
            ns avgTotalProcessingTime =
                avgGaussianTime + avgGradientTime + avgReductionTime + avgDoubleThresholdTime + avgEdgeHysteresisTime;

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
