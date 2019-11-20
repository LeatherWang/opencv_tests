#ifndef READ_INTRINSIC
#define READ_INTRINSIC

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/core/core.hpp>
#include <string>
#include <tic_toc.h>


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



struct FOVModelParam
{
    cv::Mat map_x;
    cv::Mat map_y;
    cv::Size imageSize = cv::Size(640, 480);
    float fx = 257.9419;
    float fy = 257.1647;
    float w = 0.9233381;
} fovUndistortParam;

void undistortFisheyeFOVModel()
{
    fovUndistortParam.map_x.create(fovUndistortParam.imageSize, CV_32FC1);
    fovUndistortParam.map_y.create(fovUndistortParam.imageSize, CV_32FC1);

    double Cx = fovUndistortParam.imageSize.width / 2.0;
    double Cy = fovUndistortParam.imageSize.height / 2.0;

    for (double x = -1.0; x < 1.0; x += 1.0/Cx) {
        for (double y = -1.0; y < 1.0; y += 1.0/Cy) {
            double ru = sqrt(x*x + y*y);
            double rd = (1.0 / fovUndistortParam.w)*atan(2.0*ru*tan(fovUndistortParam.w / 2.0));

            fovUndistortParam.map_x.at<float>(y*Cy + Cy, x*Cx + Cx) = rd/ru * x*fovUndistortParam.fx + Cx;
            fovUndistortParam.map_y.at<float>(y*Cy + Cy, x*Cx + Cx) = rd/ru * y*fovUndistortParam.fy + Cy;
        }
    }
}

#endif
