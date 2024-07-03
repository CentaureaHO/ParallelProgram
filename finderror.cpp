#include <bits/stdc++.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

int main()
{
    string  Path     = "Output/bakery_";
    string  SP       = Path + "serial.jpg";
    string  CP       = Path + "CUDA.jpg";
    string  MP       = Path + "MPI.jpg";
    cv::Mat Serial   = cv::imread(SP, CV_8UC1);
    cv::Mat CUDA     = cv::imread(CP, CV_8UC1);
    cv::Mat MPI      = cv::imread(MP, CV_8UC1);
    int     CUDAERR0 = 0, CUDAERR1 = 0;
    int     MPIERR0 = 0, MPIERR1 = 0;
    for (int y = 0; y < Serial.rows; ++y)
    {
        for (int x = 0; x < Serial.cols; ++x)
        {
            if (Serial.at<uchar>(y, x) > CUDA.at<uchar>(y, x))
                CUDAERR0++;
            else if (Serial.at<uchar>(y, x) < CUDA.at<uchar>(y, x))
                CUDAERR1++;
            if (Serial.at<uchar>(y, x) > MPI.at<uchar>(y, x))
                MPIERR0++;
            else if (Serial.at<uchar>(y, x) < MPI.at<uchar>(y, x))
                MPIERR1++;
        }
    }
    printf("%d %d\n%d %d", CUDAERR0, CUDAERR1, MPIERR0, MPIERR1);
}