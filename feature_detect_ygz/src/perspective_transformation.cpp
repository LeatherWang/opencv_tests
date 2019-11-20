#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

bool readIntrinsic(const string &file_path, cv::Mat &K_Mat, cv::Mat &DistCoef, cv::Size &imageSize)
{
    cv::FileStorage fs(file_path, cv::FileStorage::READ);
    if (!fs.isOpened())
        return false;

    DistCoef.at<double>(0) = fs["Camera.k1"];
    DistCoef.at<double>(1) = fs["Camera.k2"];
    DistCoef.at<double>(2) = fs["Camera.p1"];
    DistCoef.at<double>(3) = fs["Camera.p2"];

    K_Mat.at<double>(0,0) = fs["Camera.fx"];
    K_Mat.at<double>(1,1) = fs["Camera.fy"];
    K_Mat.at<double>(0,2) = fs["Camera.cx"];
    K_Mat.at<double>(1,2) = fs["Camera.cy"];

    imageSize.height = fs["Camera.height"];
    imageSize.width = fs["Camera.width"];

    cout<<"________________________________________"<<endl;
    cout << endl << "Camera Parameters: " << endl;
    cout << "- fx: " << K_Mat.at<double>(0,0) << endl;
    cout << "- fy: " << K_Mat.at<double>(1,1) << endl;
    cout << "- cx: " << K_Mat.at<double>(0,2) << endl;
    cout << "- cy: " << K_Mat.at<double>(1,2) << endl;
    cout << "- k1: " << DistCoef.at<double>(0) << endl;
    cout << "- k2: " << DistCoef.at<double>(1) << endl;
    cout << "- p1: " << DistCoef.at<double>(2) << endl;
    cout << "- p2: " << DistCoef.at<double>(3) << endl;
    cout<<"________________________________________"<<endl;
    return true;
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        cout<<"usage: error! "<<endl;
        return 0;
    }

    Mat img = cv::imread(string(argv[1]), IMREAD_GRAYSCALE);
    Mat dst;

    Mat MoveImage(img.rows,img.cols,CV_8UC1,Scalar(0));
    double angle = 45;//角度
    Point2f center(img.cols/2,img.rows/2);//中心
    Mat M = getRotationMatrix2D(center, angle, 1);//计算旋转的仿射变换矩阵
    cout<<M<<endl;
    warpAffine(img, MoveImage, M, Size(img.cols, img.rows));//仿射变换

    cv::imshow("src", img);
    cv::imshow("dst", MoveImage);

    cv::imwrite("wrapped_image.jpg", MoveImage);
    waitKey();
    return 0;
}












