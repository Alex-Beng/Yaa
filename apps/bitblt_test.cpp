// 用于bitblt截屏测试
// 主要测试截屏频率

#include <opencv2/opencv.hpp>
#include <iostream>

#include <windows.h>

int main(int argc, char const *argv[]) {

    // 绘制一张包含Hello world的黑底白字图片并显示
    cv::Mat img = cv::Mat::zeros(100, 1000, CV_8UC3);
    cv::putText(img, "Hello World !", cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
    cv::imshow("Hello World", img);
    cv::waitKey(0);
    
    return 0;
}
