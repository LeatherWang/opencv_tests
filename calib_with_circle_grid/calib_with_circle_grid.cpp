#include <Eigen/Core>
#include <Eigen/Dense> // 稠密矩阵的代数运算（逆，特征值等）
#include "stdio.h"
#include <iostream>
#include <fstream>
#include <sys/time.h>

#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;


void saveCameraParams(const cv::Size &img_size, const cv::Matx33d &kk, const cv::Vec4d &dd, const double &rms)
{
    const string outputYamlFile = "./camera_fisheye_ankobot.yaml";
    ofstream fs(outputYamlFile);
    fs << std::fixed;

    fs << "%YAML:1.0" << endl;
    fs << "    " << endl;
    fs << "# fisheye camera calibration and distortion parameters (OpenCV)" << endl;
    fs << "# Camera Number: " << outputYamlFile << endl;

    struct timeval tv;
    gettimeofday(&tv, NULL);
    struct tm *tmnow;
    tmnow = localtime(&tv.tv_sec);
    char formatTime[128] = {0};
    snprintf(formatTime, sizeof(formatTime), "%04d/%02d/%02d-%02d:%02d",
             tmnow->tm_year+1900,tmnow->tm_mon+1,tmnow->tm_mday, tmnow->tm_hour,tmnow->tm_min);
    fs << "# calibration time: " << formatTime << endl;

    fs << "Camera.width: " << img_size.width << endl;
    fs << "Camera.height: " << img_size.height << endl;
    fs.precision(12);
    fs << "Camera.fx: " << kk(0, 0) << endl;
    fs << "Camera.fy: " << kk(1, 1) << endl;
    fs << "Camera.cx: " << kk(0, 2) << endl;
    fs << "Camera.cy: " << kk(1, 2) << endl;

    fs << "Camera.k1: " << dd(0) << endl;
    fs << "Camera.k2: " << dd(1) << endl;
    fs << "Camera.k3: " << dd(2) << endl;
    fs << "Camera.k4: " << dd(3) << endl;
//    if (dd.total() > 4)
//        fs << "Camera.k3: " << dd(4) << endl;
    fs << endl;

    fs << "# rms = " << rms << endl;

    // fs << "camera_matrix = " << kk << endl;
    // fs << "distortion_coefficients = " << dd << endl;

    fs.close();
}


int main(int argc, char** argv)
{
    string filePath = "/home/leather/lxdata/ankobot_projects/datasets/images-11/";

    Mat image_raw;

    int widthCircleNum=4, heightCircleNum=11;
    float circleBaseSpacing = 3.0f; //cm
    float circleSpacing = circleBaseSpacing*2.0f;
    vector<Point3f> object;
    for(int i=0; i<heightCircleNum; i++)
    {
        for(int j=0; j<widthCircleNum; j++)
        {
            if(i&1)
                object.push_back(Point3f(i*circleBaseSpacing, j*circleSpacing+circleBaseSpacing, 0));
            else
                object.push_back(Point3f(i*circleBaseSpacing, j*circleSpacing, 0));
        }
    }

    cv::Matx33d intrinsic_matrix;
    cv::Vec4d distortion_coeff;

    vector<vector<Point3f> > vecObject3DPoints;
    vector<vector<Point2f> > vec2DCenters;

    ofstream intrinsicfile("./intrinsic_matrix_front1103.txt");

    cv::Size circleGridSize(widthCircleNum, heightCircleNum);
    std::vector<cv::Point2f> imageCenters;
    vector<Mat> vecImageRaw;
    for(int i=0; i < 10; i++)
    {
        image_raw = imread(filePath + to_string(i)+".jpg", -1);
        if (image_raw.empty())
        {
            std::cout<<"error to open: image "<<i<<std::endl;
            continue;
        }
        imshow("image_raw", image_raw);
        waitKey(10);

        Mat image_raw_bgr;
        image_raw.copyTo(image_raw_bgr);
        if(image_raw_bgr.channels() == 1)
            cvtColor(image_raw_bgr, image_raw_bgr, CV_GRAY2BGR);

        bool found = cv::findCirclesGrid(image_raw_bgr, circleGridSize, imageCenters, CALIB_CB_ASYMMETRIC_GRID);

        if (found)
        {
            cv::drawChessboardCorners(image_raw_bgr, circleGridSize, imageCenters, found);
            imshow("corner_image", image_raw_bgr);
            waitKey(500);

            vecObject3DPoints.push_back(object);
            vec2DCenters.push_back(imageCenters);
            vecImageRaw.push_back(image_raw);

            cout << "capture " << i << " pictures" << endl;
        }
        imageCenters.clear();
    }

    if(vecImageRaw.size() < 2)
    {
        cout<<"image number is less 2!!"<<endl;
        return 1;
    }

    cv::Size image_size= cv::Size(vecImageRaw[0].cols, vecImageRaw[0].rows);
    int flag = 0;
    flag |= cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
    flag |= cv::fisheye::CALIB_CHECK_COND;
    flag |= cv::fisheye::CALIB_FIX_SKEW;
    //flag |= cv::fisheye::CALIB_USE_INTRINSIC_GUESS;
    std::vector<cv::Vec3d> rotation_vectors;
    std::vector<cv::Vec3d> translation_vectors;
    double rms = cv::fisheye::calibrate(vecObject3DPoints, vec2DCenters, image_size,
                           intrinsic_matrix, distortion_coeff,
                           rotation_vectors, translation_vectors, flag, cv::TermCriteria(3, 20, 1e-6));


    cout<<intrinsic_matrix<<endl;
    cout<<distortion_coeff<<endl<<endl;

    // 对定标结果进行评价
    double total_err = 0.0;
    double err = 0.0;
    vector<Point2f>  vecReprojectPoint;

    for (int i = 0; i<vecImageRaw.size(); i++)
    {
        vector<Point3f> vecImagePointTemp3DPoint = vecObject3DPoints[i];
        fisheye::projectPoints(vecImagePointTemp3DPoint, vecReprojectPoint, rotation_vectors[i],
                               translation_vectors[i], intrinsic_matrix, distortion_coeff);
        vector<Point2f> vecImagePointTemp = vec2DCenters[i];
        Mat vecImagePointTempMat = Mat(1, vecImagePointTemp.size(), CV_32FC2);
        Mat vecReprojectPointMat = Mat(1, vecReprojectPoint.size(), CV_32FC2);
//! @todo
        for (size_t i = 0; i != vecImagePointTemp.size(); i++)
        {
            vecReprojectPointMat.at<Vec2f>(0, i) = Vec2f(vecReprojectPoint[i].x, vecReprojectPoint[i].y);
            vecImagePointTempMat.at<Vec2f>(0, i) = Vec2f(vecImagePointTemp[i].x, vecImagePointTemp[i].y);
        }
        err = norm(vecReprojectPointMat, vecImagePointTempMat, NORM_L2);

        total_err += err /= vecObject3DPoints[i].size();
        cout << "第" << i + 1 << "幅图像的平均误差：" << err << "像素" << endl;
    }
    cout << "总体平均误差：" << total_err / vecImageRaw.size() << "像素" << endl<<endl;

    Mat mapx = Mat(image_size, CV_32FC1);
    Mat mapy = Mat(image_size, CV_32FC1);
    Mat R = Mat::eye(3, 3, CV_32F);

    cout << "对图像去畸变..." << endl;
    for (int i = 0; i<vecImageRaw.size(); i++)
    {
//        fisheye::initUndistortRectifyMap(intrinsic_matrix,distortion_coeff,R,intrinsic_matrix,
//                                         image_size,CV_32FC1,mapx,mapy);
        fisheye::initUndistortRectifyMap(intrinsic_matrix, distortion_coeff, R,
                 getOptimalNewCameraMatrix(intrinsic_matrix, distortion_coeff, image_size, 1, image_size, 0),
                                         image_size, CV_32FC1, mapx, mapy);
        Mat image = vecImageRaw[i].clone();
        cv::remap(vecImageRaw[i], image, mapx, mapy, INTER_LINEAR);
        imshow("image raw", vecImageRaw[i]);
        imshow("undistort image", image);
        cout<<"translation: "<<translation_vectors[i].t()<<endl;
        waitKey();
    }

    // 测试一张图片
    if(1)
    {
        Mat distort_img = imread(filePath+"img45.jpg", -1);
        Mat undistort_img;
        Mat intrinsic_mat(intrinsic_matrix), new_intrinsic_mat;

        intrinsic_mat.copyTo(new_intrinsic_mat);
        //调节视场大小,乘的系数越小视场越大
        new_intrinsic_mat.at<double>(0, 0) *= 0.5;
        new_intrinsic_mat.at<double>(1, 1) *= 0.5;
        //调节校正图中心，建议置于校正图中心
        new_intrinsic_mat.at<double>(0, 2) = 0.5 * distort_img.cols;
        new_intrinsic_mat.at<double>(1, 2) = 0.5 * distort_img.rows;

        fisheye::undistortImage(distort_img, undistort_img, intrinsic_matrix, distortion_coeff, new_intrinsic_mat);
        imshow("output.jpg", undistort_img);
        waitKey();
    }

    cout<<endl;
    cout<<"save camera.yaml..."<<endl;
    saveCameraParams(image_size, intrinsic_matrix, distortion_coeff, rms);

    return 0;
}
















