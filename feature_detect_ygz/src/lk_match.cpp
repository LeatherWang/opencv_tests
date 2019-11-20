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

int main(int argc, char **argv)
{
    string file_path = "./camera.yaml"; // camera_fisheye_ankobot
    cout<<"intrinsic file: "<<file_path<<endl;
    cv::Mat K_Mat = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat DistCoef = cv::Mat(4, 1, CV_64F);
    cv::Size imageSize;
    //cv::Mat DistCoef_Zero = cv::Mat::zeros(4, 1, CV_64F);
    if(!readIntrinsic(file_path, K_Mat, DistCoef, imageSize))
        return 1;

    cv::Mat mapx, mapy; //内部存的是重映射之后的坐标，不是像素值
    cv::Mat R = cv::Mat::eye(3,3, CV_32F);
    cv::fisheye::initUndistortRectifyMap(K_Mat, DistCoef, R, K_Mat,
                                         imageSize, CV_32FC1, mapx, mapy);


    Mat srcImg1, srcImg2;
    if(argc > 2)
    {
        srcImg1 = imread(string(argv[1]), IMREAD_GRAYSCALE);      //待配准图
        srcImg2 = imread(string(argv[2]), IMREAD_GRAYSCALE);      //基准图
    }
    else
    {
        srcImg1 = imread("1.jpg", IMREAD_GRAYSCALE);      //待配准图
        srcImg2 = imread("2.jpg", IMREAD_GRAYSCALE);      //基准图
    }

    cv::remap(srcImg1, srcImg1, mapx, mapy, cv::INTER_LINEAR);
    cv::remap(srcImg2, srcImg2, mapx, mapy, cv::INTER_LINEAR);

//    fastNlMeansDenoising(srcImg1, srcImg1);
//    fastNlMeansDenoising(srcImg2, srcImg2);

//    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
//    clahe->apply(srcImg1, srcImg1);
//    clahe->apply(srcImg2, srcImg2);

    if(srcImg1.empty() || srcImg2.empty())
    {
        cout<< "error to open image!" << endl;
        return 0;
    }


    vector<cv::Point2f> lKeyPoints;
    vector<cv::KeyPoint> keyPoints;
    cv::Ptr<cv::FastFeatureDetector> cornorDetector = cv::FastFeatureDetector::create(20);
    cornorDetector->detect(srcImg1, keyPoints);


    vector<cv::Point2f> lastPts, curPts;
    cv::Mat tmpImage = srcImg1.clone();
    if(tmpImage.channels() == 1)
        cv::cvtColor(tmpImage, tmpImage, CV_GRAY2BGR);
    for(cv::KeyPoint ele:keyPoints)
    {
        lKeyPoints.push_back(ele.pt);
        lastPts.push_back(ele.pt);
        cv::circle(tmpImage, ele.pt, 10, cv::Scalar(0, 240, 0), 1);
    }

    cv::imshow("tmpImage1", tmpImage);

    vector<unsigned char> status;
    vector<float> errors;
    cv::calcOpticalFlowPyrLK(srcImg1, srcImg2, lastPts, curPts, status, errors);

    tmpImage = srcImg2.clone();
    if(tmpImage.channels() == 1)
        cv::cvtColor(tmpImage, tmpImage, CV_GRAY2BGR);
    int size = curPts.size();
    for(int i=0; i< size; i++)
    {
        if(status[i])
        {
            cv::circle(tmpImage, curPts[i], 10, cv::Scalar(0, 240, 0), 1);
        }
    }

//    int i=0;
//    for ( auto iter=lKeyPoints.begin(); iter!=lKeyPoints.end(); i++)
//    {
//        if ( status[i] == 0 )
//        {
//            iter = lKeyPoints.erase(iter);
//            continue;
//        }
//        *iter = curPts[i];
//        iter++;
//    }

//    for(cv::Point2f ele:lKeyPoints)
//    {
//        cv::circle(tmpImage, ele, 10, cv::Scalar(0, 240, 0), 1);
//    }

    cv::imshow("tmpImage2", tmpImage);

    waitKey();

    return 0;
}
