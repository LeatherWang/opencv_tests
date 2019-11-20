#ifndef READ_INTRINSIC
#define READ_INTRINSIC

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <string>

using namespace std;
using namespace cv;

struct IntrinsicAndUndistort
{
    cv::Mat K_Mat = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat DistCoef = cv::Mat(4, 1, CV_64F);
    cv::Size imageSize;
    cv::Mat mapx, mapy; //内部存的是重映射之后的坐标，不是像素值
} intrinsicAndUndistort;

bool readIntrinsic(const string &file_path, cv::Mat &K_Mat_, cv::Mat &DistCoef_, cv::Size &imageSize_)
{
    cv::FileStorage fs(file_path, cv::FileStorage::READ);
    if (!fs.isOpened())
        return false;

    DistCoef_.at<double>(0) = fs["Camera.k1"];
    DistCoef_.at<double>(1) = fs["Camera.k2"];
    DistCoef_.at<double>(2) = fs["Camera.p1"];
    DistCoef_.at<double>(3) = fs["Camera.p2"];

    K_Mat_.at<double>(0,0) = fs["Camera.fx"];
    K_Mat_.at<double>(1,1) = fs["Camera.fy"];
    K_Mat_.at<double>(0,2) = fs["Camera.cx"];
    K_Mat_.at<double>(1,2) = fs["Camera.cy"];

    imageSize_.height = fs["Camera.height"];
    imageSize_.width = fs["Camera.width"];

    cv::Mat R = cv::Mat::eye(3,3, CV_32F);
    cv::fisheye::initUndistortRectifyMap(K_Mat_, DistCoef_, R, K_Mat_, imageSize_, CV_32FC1,
                                         intrinsicAndUndistort.mapx, intrinsicAndUndistort.mapy);

    cout<<"________________________________________"<<endl;
    cout << endl << "Camera Parameters: " << endl;
    cout << "- fx: " << K_Mat_.at<double>(0,0) << endl;
    cout << "- fy: " << K_Mat_.at<double>(1,1) << endl;
    cout << "- cx: " << K_Mat_.at<double>(0,2) << endl;
    cout << "- cy: " << K_Mat_.at<double>(1,2) << endl;
    cout << "- k1: " << DistCoef_.at<double>(0) << endl;
    cout << "- k2: " << DistCoef_.at<double>(1) << endl;
    cout << "- p1: " << DistCoef_.at<double>(2) << endl;
    cout << "- p2: " << DistCoef_.at<double>(3) << endl;
    cout<<"________________________________________"<<endl;
    return true;
}

#endif
