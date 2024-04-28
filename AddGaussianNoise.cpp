#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
using namespace std;

int main()
{
    Mat img_ = imread("Images/bakery.jpg");
    Mat luffy;
    resize(img_, luffy, Size(img_.cols / 2, img_.rows / 2));
    Mat luffy_gray;
    cvtColor(luffy, luffy_gray, COLOR_BGR2GRAY);
    if (luffy.empty() || luffy_gray.empty())
    {
        cout << "请确认输入的路径是否正确" << endl;
        return -1;
    }
    // 生成与原图像相同尺寸、数据类型和通道类型的矩阵
    Mat luffy_noise      = Mat::zeros(luffy.rows, luffy.cols, luffy.type());
    Mat luffy_gray_noise = Mat::zeros(luffy.rows, luffy.cols, luffy_gray.type());
    imshow("luffy原图", luffy);
    imshow("luffy_gray原图", luffy_gray);
    RNG rng;                                          // 创建一个RNG类
    rng.fill(luffy_noise, RNG::NORMAL, 10, 20);       // 生成三通道的高斯分布随机数
    rng.fill(luffy_gray_noise, RNG::NORMAL, 15, 30);  // 生成三通道的高斯分布随机数
    imshow("三通道高斯噪声", luffy_noise);
    imshow("单通道高斯噪声", luffy_gray_noise);
    luffy      = luffy + luffy_noise;            // 在彩色图像中添加高斯噪声
    luffy_gray = luffy_gray + luffy_gray_noise;  // 在灰度图像中添加高斯噪声
    // 显示添加高斯噪声后的图像
    imshow("lufy添加噪声", luffy);
    imwrite("luffy_noise.jpg", luffy);
    imshow("lufy_gray添加噪声", luffy_gray);
    waitKey(0);
    return 0;
}