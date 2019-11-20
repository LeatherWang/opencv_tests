#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
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

//        fastNlMeansDenoising(srcImg1, srcImg1);
//        fastNlMeansDenoising(srcImg2, srcImg2);

//        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
//        clahe->apply(srcImg1, srcImg1);
//        clahe->apply(srcImg2, srcImg2);


    //    imshow("srcImg2", srcImg2);
    //    waitKey();

    if(srcImg1.empty() || srcImg2.empty())
    {
        cout<< "error to open image!" << endl;
        return 0;
    }

    vector<KeyPoint> keypoints_1, keypoints_2;
    Mat descriptors_1, descriptors_2;
    Ptr<FeatureDetector> harrisDetector = FeatureDetector::create("FAST");
    auto descriptor = DescriptorExtractor::create("BRIEF");
    Ptr<cv::BRISK> briskDetector(new cv::BRISK(5, 1));

    harrisDetector->detect(srcImg1, keypoints_1);
    harrisDetector->detect(srcImg2, keypoints_2);
    cout<<"keypoints_1: "<<keypoints_1.size()<<"  keypoints_2: "<<keypoints_1.size()<<endl;

    {
        Mat feature_pic1, feature_pic2;
        drawKeypoints(srcImg1, keypoints_1, feature_pic1,
                      Scalar(0, 255, 0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        drawKeypoints(srcImg2, keypoints_2, feature_pic2,
                      Scalar(0, 255, 0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        imshow("feature1__", feature_pic1);
        imshow("feature2__", feature_pic2);
    }

    briskDetector->compute ( srcImg1, keypoints_1, descriptors_1 );
    briskDetector->compute ( srcImg2, keypoints_2, descriptors_2 );
    cout<<"keypoints_1: "<<keypoints_1.size()<<"  keypoints_2: "<<keypoints_1.size()<<endl;

    Mat feature_pic1, feature_pic2;
    drawKeypoints(srcImg1, keypoints_1, feature_pic1,
                  Scalar(0, 255, 0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    drawKeypoints(srcImg2, keypoints_2, feature_pic2,
                  Scalar(0, 255, 0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imwrite("feature_pic1.bmp", feature_pic1);
    imwrite("feature_pic2.bmp", feature_pic2);

    imshow("feature1", feature_pic1);
    imshow("feature2", feature_pic2);



    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<DMatch> match;
    //BFMatcher matcher ( NORM_HAMMING );
    Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce-Hamming" );
    matcher->match ( descriptors_1, descriptors_2, match );

    //-- 第四步:匹配点对筛选
    double min_dist=10000, max_dist=0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        double dist = match[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }

    printf ( "-- Max dist : %f \n", max_dist );
    printf ( "-- Min dist : %f \n", min_dist );

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    vector<DMatch> good_matches;
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        if ( match[i].distance <= max ( 2*min_dist, 200.0 ) )
        {
            good_matches.push_back ( match[i] );
        }
    }

    vector<KeyPoint> R_keypoint01, R_keypoint02;
    for (int i=0; i<good_matches.size(); i++)
    {
        R_keypoint01.push_back(keypoints_1[good_matches[i].queryIdx]);
        R_keypoint02.push_back(keypoints_2[good_matches[i].trainIdx]);
        good_matches[i].queryIdx = i;
        good_matches[i].trainIdx = i;
    }

    Mat img_R_matches;
    drawMatches(srcImg1, R_keypoint01,
                srcImg2, R_keypoint02, good_matches, img_R_matches,
                Scalar::all(-1), Scalar::all(-1));
    imshow("before ransac", img_R_matches);

    vector<Point2f> p01,p02;
    for (int i=0;i<good_matches.size();i++)
    {
        p01.push_back(R_keypoint01[i].pt);
        p02.push_back(R_keypoint02[i].pt);
    }

    //计算基础矩阵并剔除误匹配点
    vector<uchar> RansacStatus;
    Mat H = findHomography(p01, p02, RansacStatus, CV_RANSAC);
//    Mat dst;
//    warpPerspective(srcImg1, dst, H, Size(srcImg1.cols, srcImg1.rows));

    //剔除误匹配的点对
    vector<KeyPoint> RR_keypoint01, RR_keypoint02;
    vector<DMatch> RR_matches;            //重新定义RR_keypoint 和RR_matches来存储新的关键点和匹配矩阵
    int index=0;
    for (int i=0;i<good_matches.size();i++)
    {
        if (RansacStatus[i]!=0)
        {
            RR_keypoint01.push_back(R_keypoint01[i]);
            RR_keypoint02.push_back(R_keypoint02[i]);
            good_matches[i].queryIdx=index;
            good_matches[i].trainIdx=index;
            RR_matches.push_back(good_matches[i]);
            index++;
        }
    }
    cout<<"RR_matches.size(): "<<RR_matches.size()<<endl;

    //画出消除误匹配后的图
    Mat img_RR_matches;
    drawMatches(srcImg1, RR_keypoint01,
                srcImg2, RR_keypoint02, RR_matches,
                img_RR_matches, Scalar::all(-1), Scalar::all(-1));
    imshow("after ransac", img_RR_matches);


    waitKey();
    imwrite("before ransac.bmp", img_R_matches);
    imwrite("after ransac.bmp", img_RR_matches);

    return 0;
}
