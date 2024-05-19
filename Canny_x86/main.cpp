#include <bits/stdc++.h>
#include <filesystem>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>
#include <memory>
#include "Solutions.h"
#include <omp.h>

namespace fs = std::filesystem;
using ns     = std::chrono::nanoseconds;
using hrClk  = std::chrono::high_resolution_clock;
using Str    = std::string;
template <typename T>
using Vec = std::vector<T>;

struct AlignedDeleter
{
    void operator()(void* ptr) const { std::free(ptr); }
};

template <typename T>
std::unique_ptr<T[], AlignedDeleter> MakeAlignedArray(size_t size, size_t alignment)
{
    void* ptr = std::aligned_alloc(alignment, size * sizeof(T));
    if (!ptr) throw std::bad_alloc();
    return std::unique_ptr<T[], AlignedDeleter>(static_cast<T*>(ptr));
}

void SaveGradientsToFile(const Vec<float>& Grads, const Vec<uint8_t>& Dirs, const Str& GradPath, const Str& DirPath)
{
    std::ofstream GradFile(GradPath, std::ios::binary);
    std::ofstream DirFile(DirPath, std::ios::binary);

    if (!GradFile.is_open() || !DirFile.is_open())
    {
        std::cerr << "Error opening gradient or direction files for writing.\n";
        return;
    }

    GradFile.write(reinterpret_cast<const char*>(Grads.data()), Grads.size() * sizeof(float));
    DirFile.write(reinterpret_cast<const char*>(Dirs.data()), Dirs.size() * sizeof(uint8_t));

    GradFile.close();
    DirFile.close();
}

int main()
{
    int n         = 0;
    int UseThread = 9;

    std::cout << "Max thread nums: " << std::thread::hardware_concurrency() << ".\n";
    Vec<std::tuple<Str, std::pair<int, int>, double>> ImageStatistics;

    Str       ImgPath      = "../Images/";
    Str       OutputPath   = "../Output/";
    Str       GaussianPath = "../GaussianImg/";
    Str       GradientPath = "../GradientData/";
    Str       ReduImgPath  = "../ReducedImg/";
    const int LowThre      = 30;
    const int HighThre     = 90;

    std::function<void(uint8_t*, const uint8_t*, int, int)>         Gauss  = OpenMP::PerformGaussianBlur;
    std::function<void(float*, uint8_t*, const uint8_t*, int, int)> Grad   = OpenMP::ComputeGradients;
    std::function<void(float*, float*, uint8_t*, int, int)>         ReduNM = Serial::ReduceNonMaximum;
    std::function<void(uint8_t*, float*, int, int, int, int)>       DbThre = Serial::PerformDoubleThresholding;
    std::function<void(uint8_t*, uint8_t*, int, int)>               EdgeHy = Serial::PerformEdgeHysteresis;

    omp_set_num_threads(16);
    n = 1;
    for (int th = 1; th >= 1; --th)
    {
        UseThread = th;
        std::ofstream CSV("PThread_Thread" + std::to_string(4) + ".csv");
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

                cv::Mat GreyImg, EdgedImg;
                cv::cvtColor(OriImg, GreyImg, cv::COLOR_BGR2GRAY);
                int OriginalWidth  = GreyImg.cols;
                int OriginalHeight = GreyImg.rows;
                int TmpWidth       = GreyImg.cols < 16 ? 16 : GreyImg.cols / 16 * 16;
                int TmpHeight      = GreyImg.rows < 16 ? 16 : GreyImg.rows / 16 * 16;
                cv::resize(GreyImg, GreyImg, cv::Size(TmpWidth, TmpHeight));
                EdgedImg.create(GreyImg.size(), CV_8UC1);
                cv::Mat GaussianImg = cv::imread(GaussianPath + Entry.path().filename().string(), CV_8UC1);
                cv::resize(GaussianImg, GaussianImg, cv::Size(TmpWidth, TmpHeight));

                auto            GradientPixels   = MakeAlignedArray<float>(GreyImg.total(), 64);
                auto            MatrixPixels     = MakeAlignedArray<float>(GreyImg.total(), 64);
                auto            SegmentPixels    = MakeAlignedArray<uint8_t>(GreyImg.total(), 64);
                auto            DoubleThrePixels = MakeAlignedArray<uint8_t>(GreyImg.total(), 64);
                std::vector<ns> Durations;

                // 以下部分为预运行，用于给OpenMP创建线程池来避免其初始化对测试数据的影响
                for (int i = 0; i < 10; ++i)
                    OpenMP::PerformGaussianBlur(GaussianImg.data, GreyImg.data, GreyImg.cols, GreyImg.rows);

                // 以下部分为实际测试
                for (int i = 0; i < n; ++i)
                {
                    auto Start = hrClk::now();
                    Gauss(GaussianImg.data, GreyImg.data, GreyImg.cols, GreyImg.rows);
                    auto End = hrClk::now();

                    Grad(GradientPixels.get(), SegmentPixels.get(), GaussianImg.data, GreyImg.cols, GreyImg.rows);

                    /*
                    SaveGradientsToFile(GradientPixels,
                        SegmentPixels,
                        GradientPath + "Gradient/" + Entry.path().stem().string() + ".bin",
                        GradientPath + "Direction/" + Entry.path().stem().string() + ".bin");
                    */

                    ReduNM(MatrixPixels.get(), GradientPixels.get(), SegmentPixels.get(), GreyImg.cols, GreyImg.rows);

                    DbThre(DoubleThrePixels.get(), MatrixPixels.get(), HighThre, LowThre, GreyImg.cols, GreyImg.rows);

                    EdgeHy(EdgedImg.data, DoubleThrePixels.get(), GreyImg.cols, GreyImg.rows);

                    Durations.push_back(std::chrono::duration_cast<ns>(End - Start));
                }
                ns AvgTime =
                    std::accumulate(Durations.begin(), Durations.end(), ns(0), [](ns a, ns b) { return a + b; }) / n;

                // cv::imwrite(GaussianPath + Entry.path().filename().string(), GaussianImg);
                cv::resize(EdgedImg, EdgedImg, cv::Size(OriginalWidth, OriginalHeight));
                if (!cv::imwrite(OutputPath + Entry.path().filename().string(), EdgedImg))
                {
                    std::cerr << "Failed to save the image: " << OutputPath + Entry.path().filename().string()
                              << std::endl;
                }
                else
                {
                    std::cout << "Final image saved to " << OutputPath + Entry.path().filename().string() << std::endl;
                }

                std::cout << "Processed " << Entry.path().filename().string() << " (" << GreyImg.cols << "x"
                          << GreyImg.rows << "): " << AvgTime.count() << "ns" << std::endl;
                CSV << Entry.path().filename().string() << "," << GreyImg.cols << "x" << GreyImg.rows << ","
                    << AvgTime.count() << "\n";
                Durations.clear();
            }
        }
        std::cout << "Thread " << UseThread << " done.\n\n";
        CSV.close();
    }
}