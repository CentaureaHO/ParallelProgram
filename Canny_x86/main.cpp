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

Str       ImgPath    = "../Images/";
Str       OutputPath = "../Output/";
const int LowThre    = 30;
const int HighThre   = 90;

void Excute(Canny* Op, uint8_t* GreyImageArray, uint8_t* GaussianImageArray, float* GradientPixels, float* MatrixPixels,
    uint8_t* SegmentPixels, uint8_t* DoubleThrePixels, uint8_t* EdgeArray, int Width, int Height, ns& TotalGaussianTime,
    ns& TotalGradientTime, ns& TotalReduceTime, ns& TotalDoubleThresholdTime, ns& TotalEdgeHysteresisTime)
{
    auto start = hrClk::now();
    Op->PerformGaussianBlur(GaussianImageArray, GreyImageArray, Width, Height);
    auto end = hrClk::now();
    TotalGaussianTime += std::chrono::duration_cast<ns>(end - start);

    start = hrClk::now();
    Op->ComputeGradients(GradientPixels, SegmentPixels, GaussianImageArray, Width, Height);
    end = hrClk::now();
    TotalGradientTime += std::chrono::duration_cast<ns>(end - start);

    start = hrClk::now();
    Op->ReduceNonMaximum(MatrixPixels, GradientPixels, SegmentPixels, Width, Height);
    end = hrClk::now();
    TotalReduceTime += std::chrono::duration_cast<ns>(end - start);

    start = hrClk::now();
    Op->PerformDoubleThresholding(DoubleThrePixels, MatrixPixels, HighThre, LowThre, Width, Height);
    end = hrClk::now();
    TotalDoubleThresholdTime += std::chrono::duration_cast<ns>(end - start);

    start = hrClk::now();
    Op->PerformEdgeHysteresis(EdgeArray, DoubleThrePixels, Width, Height);
    end = hrClk::now();
    TotalEdgeHysteresisTime += std::chrono::duration_cast<ns>(end - start);
}

cv::Mat PerformCanny(cv::Mat EdgedImg, const cv::Mat& GreyImg, Canny* Op, std::ofstream& CSV, Str Filename,
    unsigned int RepeatTimes = 1)
{
    std::string baseFilename   = Filename.substr(0, Filename.find_last_of("."));
    std::string suffix         = ".jpg";
    int         OriginalWidth  = GreyImg.cols;
    int         OriginalHeight = GreyImg.rows;
    int         TmpWidth       = GreyImg.cols < 16 ? 16 : GreyImg.cols / 16 * 16;
    int         TmpHeight      = GreyImg.rows < 16 ? 16 : GreyImg.rows / 16 * 16;
    cv::Mat     ResizedGreyImg;
    cv::resize(GreyImg, ResizedGreyImg, cv::Size(TmpWidth, TmpHeight));
    EdgedImg.create(ResizedGreyImg.size(), CV_8UC1);

    cv::Mat GradientImg(ResizedGreyImg.size(), CV_8UC1);
    cv::Mat MatrixImg(ResizedGreyImg.size(), CV_8UC1);
    cv::Mat DoubleThreImg(ResizedGreyImg.size(), CV_8UC1);

    uint8_t* GreyImageArray     = (uint8_t*)_mm_malloc(TmpWidth * TmpHeight * sizeof(uint8_t), 64);
    uint8_t* GaussianImageArray = (uint8_t*)_mm_malloc(TmpWidth * TmpHeight * sizeof(uint8_t), 64);
    float*   GradientPixels     = (float*)_mm_malloc(TmpWidth * TmpHeight * sizeof(float), 64);
    float*   MatrixPixels       = (float*)_mm_malloc(TmpWidth * TmpHeight * sizeof(float), 64);
    uint8_t* SegmentPixels      = (uint8_t*)_mm_malloc(TmpWidth * TmpHeight * sizeof(uint8_t), 64);
    uint8_t* DoubleThrePixels   = (uint8_t*)_mm_malloc(TmpWidth * TmpHeight * sizeof(uint8_t), 64);
    uint8_t* EdgeArray          = (uint8_t*)_mm_malloc(TmpWidth * TmpHeight * sizeof(uint8_t), 64);

    ns TotalGaussianTime        = ns(0);
    ns TotalGradientTime        = ns(0);
    ns TotalReduceTime          = ns(0);
    ns TotalDoubleThresholdTime = ns(0);
    ns TotalEdgeHysteresisTime  = ns(0);

    memcpy(GreyImageArray, ResizedGreyImg.data, TmpWidth * TmpHeight * sizeof(uint8_t));

    for (int i = 0; i < 1; ++i)
        Op->PerformGaussianBlur(GaussianImageArray, GreyImageArray, ResizedGreyImg.cols, ResizedGreyImg.rows);

    for (int i = 0; i < RepeatTimes; ++i)
    {
        Excute(Op,
            GreyImageArray,
            GaussianImageArray,
            GradientPixels,
            MatrixPixels,
            SegmentPixels,
            DoubleThrePixels,
            EdgeArray,
            ResizedGreyImg.cols,
            ResizedGreyImg.rows,
            TotalGaussianTime,
            TotalGradientTime,
            TotalReduceTime,
            TotalDoubleThresholdTime,
            TotalEdgeHysteresisTime);
    }
    cv::Mat GaussianImg(ResizedGreyImg.size(), CV_8UC1, GaussianImageArray);
    cv::imwrite(OutputPath + baseFilename + "_Gaussian" + suffix, GaussianImg);

    for (int y = 0; y < ResizedGreyImg.rows; ++y)
    {
        for (int x = 0; x < ResizedGreyImg.cols; ++x)
        {
            GradientImg.at<uchar>(y, x) = static_cast<uchar>(GradientPixels[y * ResizedGreyImg.cols + x]);
        }
    }

    cv::imwrite(OutputPath + baseFilename + "_Gradient" + suffix, GradientImg);

    for (int y = 0; y < ResizedGreyImg.rows; ++y)
    {
        for (int x = 0; x < ResizedGreyImg.cols; ++x)
        {
            MatrixImg.at<uchar>(y, x) = static_cast<uchar>(MatrixPixels[y * ResizedGreyImg.cols + x]);
        }
    }

    cv::imwrite(OutputPath + baseFilename + "_Matrix" + suffix, MatrixImg);

    for (int y = 0; y < ResizedGreyImg.rows; ++y)
    {
        for (int x = 0; x < ResizedGreyImg.cols; ++x)
        {
            DoubleThreImg.at<uchar>(y, x) = static_cast<uchar>(DoubleThrePixels[y * ResizedGreyImg.cols + x]);
        }
    }

    cv::imwrite(OutputPath + baseFilename + "_DoubleThre" + suffix, DoubleThreImg);

    ns avgGaussianTime        = TotalGaussianTime / RepeatTimes;
    ns avgGradientTime        = TotalGradientTime / RepeatTimes;
    ns avgReductionTime       = TotalReduceTime / RepeatTimes;
    ns avgDoubleThresholdTime = TotalDoubleThresholdTime / RepeatTimes;
    ns avgEdgeHysteresisTime  = TotalEdgeHysteresisTime / RepeatTimes;
    ns avgTotalProcessingTime =
        avgGaussianTime + avgGradientTime + avgReductionTime + avgDoubleThresholdTime + avgEdgeHysteresisTime;

    memcpy(EdgedImg.data, EdgeArray, TmpWidth * TmpHeight * sizeof(uint8_t));
    cv::resize(EdgedImg, EdgedImg, cv::Size(OriginalWidth, OriginalHeight));
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

    if (!cv::imwrite(OutputPath + Filename, EdgedImg))
    {
        std::cerr << "Failed to save the image: " << OutputPath + Filename << std::endl;
    }
    else { std::cout << "Final image saved to " << OutputPath + Filename << std::endl; }

    std::cout << "Processed " << Filename << " (" << ResizedGreyImg.cols << "x" << ResizedGreyImg.rows
              << "): " << avgTotalProcessingTime.count() << "ns" << std::endl;

    CSV << Filename << "," << ResizedGreyImg.cols << "x" << ResizedGreyImg.rows << "," << avgGaussianTime.count() << ","
        << avgGradientTime.count() << "," << avgReductionTime.count() << "," << avgDoubleThresholdTime.count() << ","
        << avgEdgeHysteresisTime.count() << "," << avgTotalProcessingTime.count() << "\n";
    return EdgedImg;
}

int main()
{
    int n         = 1;
    int MaxThread = 16;
    int Choice    = 0;
    std::cout << "Choose the method to run:\n"
              << "1. Serial\n"
              << "2. SIMD\n"
              << "3. PThread\n"
              << "4. PThread with Pool\n"
              << "5. OpenMP\n"
              << "6. OneAPI\n";
    std::cin >> Choice;

    std::cout << "Repeat the process how many times? ";
    std::cin >> n;

    std::cout << "How many threads to test? ";
    std::cin >> MaxThread;

    Serial&          Serial_Ins   = Serial::GetInstance();
    SIMD::AVX::A512& SIMD_Ins     = SIMD::AVX::A512::GetInstance();
    PThread&         PThread_Ins  = PThread::GetInstance(MaxThread);
    PThreadWithPool& PThreadP_Ins = PThreadWithPool::GetInstance(MaxThread);
    OpenMP&          OMP_Ins      = OpenMP::GetInstance(MaxThread);
    OneAPI&          OneAPI_Ins   = OneAPI::GetInstance(MaxThread);

    Canny* Op = nullptr;

    switch (Choice)
    {
        case 1: Op = &Serial_Ins; break;
        case 2: Op = &SIMD_Ins; break;
        case 3: Op = &PThread_Ins; break;
        case 4: Op = &PThreadP_Ins; break;
        case 5: Op = &OMP_Ins; break;
        case 6: Op = &OneAPI_Ins; break;
        default: std::cerr << "Invalid choice.\n"; return 1;
    }

    std::string filename = "";
    switch (Choice)
    {
        case 1: filename = "Serial"; break;
        case 2: filename = "SIMD"; break;
        case 3: filename = "PThread_" + std::to_string(MaxThread); break;
        case 4: filename = "PThreadPool_" + std::to_string(MaxThread); break;
        case 5: filename = "OpenMP_" + std::to_string(MaxThread); break;
        case 6: filename = "OneAPI_" + std::to_string(MaxThread); break;
    }

    filename = filename + ".csv";

    std::ofstream CSV(filename);
    CSV << "Image Name,Width x Height,Gaussian Time (ns),Gradient Time (ns),Reduce Time (ns),Double Threshold Time "
           "(ns),Edge Hysteresis Time (ns),Total Processing Time (ns)\n";
    for (const auto& Entry : fs::directory_iterator(ImgPath))
    {
        if (Entry.path().extension() == ".jpg" && Entry.path().filename().string() == "bakery.jpg")
        {
            cv::Mat OriImg = cv::imread(Entry.path().string(), cv::IMREAD_COLOR);
            if (OriImg.empty())
            {
                std::cerr << "Error loading the image: " << Entry.path().string() << std::endl;
                continue;
            }

            cv::Mat GreyImg, EdgedImg;
            cv::cvtColor(OriImg, GreyImg, cv::COLOR_BGR2GRAY);

            cv::Mat Serial = PerformCanny(EdgedImg, GreyImg, (Canny*)&Serial_Ins, CSV, Entry.path().filename().string(), n);
            // cv::Mat Target = PerformCanny(EdgedImg, GreyImg, Op, CSV, Entry.path().filename().string(), n);
            /*
            int ErrEdge = 0, ErrNoEdge = 0;

            for (int y = 0; y < Serial.rows; ++y)
            {
                for (int x = 0; x < Serial.cols; ++x)
                {
                    if (Serial.at<uchar>(y, x) > Target.at<uchar>(y, x)) ErrNoEdge++;
                    else if (Serial.at<uchar>(y, x) < Target.at<uchar>(y, x)) ErrEdge++;
                }
            }
            std::cout << ErrNoEdge << " " << ErrEdge << "\n";
            */
        }
    }
    std::cout << "Thread " << MaxThread << " done.\n\n";
    CSV.close();
}
