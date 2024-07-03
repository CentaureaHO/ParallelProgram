#include <iostream>
#include <filesystem>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>
#include <chrono>

namespace fs = std::filesystem;

void PerformCannyEdgeDetection(uint8_t* Output, const uint8_t* OriImg, int Width, int Height);

int main()
{
    std::string ImgPath    = "../Images/";
    std::string OutputPath = "../Output/";
    std::string filename   = "output.csv";
    {
        cv::Mat OriImg = cv::imread(ImgPath + "earth.jpg", cv::IMREAD_COLOR);

        cv::Mat GreyImg, EdgedImg;
        cv::cvtColor(OriImg, GreyImg, cv::COLOR_BGR2GRAY);
        int      OriginalWidth  = GreyImg.cols;
        int      OriginalHeight = GreyImg.rows;
        int      TmpWidth       = GreyImg.cols < 16 ? 16 : GreyImg.cols / 16 * 16;
        int      TmpHeight      = GreyImg.rows < 16 ? 16 : GreyImg.rows / 16 * 16;
        uint8_t* GreyImageArray = new uint8_t[TmpWidth * TmpHeight];
        uint8_t* EdgeArray      = new uint8_t[TmpWidth * TmpHeight];
        cv::resize(GreyImg, GreyImg, cv::Size(TmpWidth, TmpHeight));
        EdgedImg.create(GreyImg.size(), CV_8UC1);

        memcpy(GreyImageArray, GreyImg.data, TmpWidth * TmpHeight * sizeof(uint8_t));
        auto Start = std::chrono::high_resolution_clock::now();
        PerformCannyEdgeDetection(EdgeArray, GreyImageArray, TmpWidth, TmpHeight);
        auto End       = std::chrono::high_resolution_clock::now();
        auto TotalTime = std::chrono::duration_cast<std::chrono::nanoseconds>(End - Start).count();

        memcpy(EdgedImg.data, EdgeArray, TmpWidth * TmpHeight * sizeof(uint8_t));
        cv::resize(EdgedImg, EdgedImg, cv::Size(OriginalWidth, OriginalHeight));

        delete[] GreyImageArray;
        delete[] EdgeArray;
    }

    std::ofstream CSV(filename);
    CSV << "Image Name,Width x Height,Total Processing Time (ns)\n";
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
            int      OriginalWidth  = GreyImg.cols;
            int      OriginalHeight = GreyImg.rows;
            int      TmpWidth       = GreyImg.cols < 16 ? 16 : GreyImg.cols / 16 * 16;
            int      TmpHeight      = GreyImg.rows < 16 ? 16 : GreyImg.rows / 16 * 16;
            uint8_t* GreyImageArray = new uint8_t[TmpWidth * TmpHeight];
            uint8_t* EdgeArray      = new uint8_t[TmpWidth * TmpHeight];
            cv::resize(GreyImg, GreyImg, cv::Size(TmpWidth, TmpHeight));
            EdgedImg.create(GreyImg.size(), CV_8UC1);

            memcpy(GreyImageArray, GreyImg.data, TmpWidth * TmpHeight * sizeof(uint8_t));
            auto Start = std::chrono::high_resolution_clock::now();
            PerformCannyEdgeDetection(EdgeArray, GreyImageArray, TmpWidth, TmpHeight);
            auto End       = std::chrono::high_resolution_clock::now();
            auto TotalTime = std::chrono::duration_cast<std::chrono::nanoseconds>(End - Start).count();

            CSV << Entry.path().filename().string() << "," << OriginalWidth << "x" << OriginalHeight << "," << TotalTime
                << "\n";
            memcpy(EdgedImg.data, EdgeArray, TmpWidth * TmpHeight * sizeof(uint8_t));
            cv::resize(EdgedImg, EdgedImg, cv::Size(OriginalWidth, OriginalHeight));
            cv::imwrite(OutputPath + Entry.path().filename().string(), EdgedImg);
            delete[] GreyImageArray;
            delete[] EdgeArray;
        }
    }
    CSV.close();
}
