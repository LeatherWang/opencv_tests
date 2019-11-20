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

#include <boost/filesystem.hpp>

using namespace cv;
using namespace std;


void saveCameraParams(const cv::Size &img_size, const cv::Matx33d &kk, const cv::Vec4d &dd, const double &rms, string savePath)
{
    if(savePath.back() != '/')
        savePath.push_back('/');
    const string outputYamlFile = savePath + "camera_fisheye.yaml";
    ofstream fs(outputYamlFile);
    fs << std::fixed;

    fs << "%YAML:1.0" << endl;
    fs << "    " << endl;
    fs << "# Camera calibration and distortion parameters (OpenCV)" << endl;
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
    fs << "Camera.p1: " << dd(2) << endl;
    fs << "Camera.p2: " << dd(3) << endl;
    //    if (dd.total() > 4)
    //        fs << "Camera.k3: " << dd(4) << endl;
    fs << endl;

    fs << "# rms = " << rms << endl;

    fs << "Camera.Model: fisheye" <<endl;
    fs << endl;

    fs.close();
}


int main(int argc, char** argv)
{
    if(argc < 2)
    {
        cout<<"Usage error!!"<<endl;
        return 0;
    }

    string inputDir = string(argv[1]);
    std::string fileExtension = ".png";
    std::vector<std::string> imageFilenames;
    for (boost::filesystem::directory_iterator itr(inputDir); itr != boost::filesystem::directory_iterator(); ++itr)
    {
        if (!boost::filesystem::is_regular_file(itr->status()))
            continue;

        std::string filename = itr->path().filename().string();

        if (filename.compare(filename.length() - fileExtension.length(), fileExtension.length(), fileExtension) != 0)
            continue;
        imageFilenames.push_back(itr->path().string());
    }
    if (imageFilenames.empty())
    {
        std::cerr << "# ERROR: No images found." << std::endl;
        return 1;
    }

    auto cmp = [](const std::string &a, const std::string &b){
        if(a.size() < b.size())
            return true;
        else if(a.size() == b.size())
            return a<b;
        return false;
    };
    sort(imageFilenames.begin(), imageFilenames.end(), cmp);


    Mat image_raw;
    cv::Size mBoardSize = cv::Size(6, 9); //    Size_(_Tp _width, _Tp _height);
    double mGridSize = 0.038;
    vector<cv::Point3f> object;
    //3D points
    for (int i=0; i<mBoardSize.height; i++) {  //8
        for (int j=0; j<mBoardSize.width; j++) //5
            object.push_back(cv::Point3f((float)i*mGridSize, (float)j*mGridSize, 0.0f));
    }


    cv::Matx33d intrinsic_matrix;
    cv::Vec4d distortion_coeff;

    vector<vector<Point3f> > vecObject3DPoints;
    vector<vector<Point2f> > vec2DCenters;

    cout<<"imageFilenames.size(): "<<endl;
    vector<Mat> vecImageRaw;
    int lastFindIndex=-10;
    for(int i=0; i < imageFilenames.size(); i++)
    {
        if(abs(i-lastFindIndex) > 0)
        {
            cout << imageFilenames[i] << "  ";
            image_raw = imread(imageFilenames[i], cv::IMREAD_GRAYSCALE);
            if (image_raw.empty())
            {
                std::cout<<"error to open: image "<<i<<std::endl;
                continue;
            }

            std::vector<cv::Point2f> imageCenters;
            bool found = cv::findChessboardCorners(image_raw, mBoardSize, imageCenters);

            Mat image_raw_bgr = image_raw.clone();
            if(image_raw_bgr.channels() == 1)
                cvtColor(image_raw_bgr, image_raw_bgr, CV_GRAY2BGR);

            if (found)
            {
                cv::cornerSubPix(image_raw, imageCenters,
                                 cv::Size(5,5),
                                 cv::Size(-1,-1),
                                 cv::TermCriteria(cv::TermCriteria::MAX_ITER +
                                                  cv::TermCriteria::EPS,
                                                  30,		// max number of iterations
                                                  0.1));     // min accuracy

                cv::drawChessboardCorners(image_raw_bgr, mBoardSize, imageCenters, found);
                cv::circle(image_raw_bgr, imageCenters[0], 20, cv::Scalar(0,255,0));
                vecObject3DPoints.push_back(object);
                vec2DCenters.push_back(imageCenters);
                vecImageRaw.push_back(image_raw);
                lastFindIndex = i;

                cout << "success" << endl;
            }
            else
                cout << "error__" << endl;

            imshow("corner_image", image_raw_bgr);
            waitKey(100);
        }
    }

    if(vecImageRaw.size() < 3)
    {
        cout<<"image number is less 3!!"<<endl;
        return 1;
    }

    cout<<"vec2DCenters.size(): "<<vec2DCenters.size()<<endl<<endl;
    cout<<"start calibrate..."<<endl;

    cv::Size image_size= cv::Size(vecImageRaw[0].cols, vecImageRaw[0].rows);
    int flag = 0;
    flag |= cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
    //flag |= cv::fisheye::CALIB_CHECK_COND;
    flag |= cv::fisheye::CALIB_FIX_SKEW;
    //flag |= cv::fisheye::CALIB_USE_INTRINSIC_GUESS;
    std::vector<cv::Vec3d> rotation_vectors;
    std::vector<cv::Vec3d> translation_vectors;
    double rms = cv::fisheye::calibrate(vecObject3DPoints, vec2DCenters, image_size,
                           intrinsic_matrix, distortion_coeff,
                           rotation_vectors, translation_vectors, flag, cv::TermCriteria(3, 20, 1e-6)); //cv::TermCriteria(3, 400, 1e-8)

    cout<<"intrinsic_matrix: "<<endl<<intrinsic_matrix<<endl;
    cout<<"distortion_coeff: "<<endl<<distortion_coeff<<endl<<endl;

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

    cout<<endl;
    cout<<"save camera.yaml..."<<endl<<endl<<endl;
    saveCameraParams(image_size, intrinsic_matrix, distortion_coeff, rms, inputDir);

    Mat mapx = Mat(image_size, CV_64FC1);
    Mat mapy = Mat(image_size, CV_64FC1);
    Mat R = Mat::eye(3, 3, CV_64F);

    cout << "对图像去畸变..." << endl;

    fisheye::initUndistortRectifyMap(intrinsic_matrix,distortion_coeff,R,intrinsic_matrix,
                                     image_size,CV_32FC1,mapx,mapy);
    for (int i = 0; i<vecImageRaw.size(); i++)
    {
        Mat image = vecImageRaw[i].clone();
        cv::remap(vecImageRaw[i], image, mapx, mapy, INTER_LINEAR);
        imshow("image raw", vecImageRaw[i]);
        imshow("undistort image", image);
        cout<<"translation: "<<translation_vectors[i].t()<<endl;
        waitKey();
    }

//    // 测试一张图片
//    if(1)
//    {
//        Mat distort_img = imread("./img45.jpg", -1);
//        Mat undistort_img;
//        Mat intrinsic_mat(intrinsic_matrix), new_intrinsic_mat;

//        intrinsic_mat.copyTo(new_intrinsic_mat);
//        //调节视场大小,乘的系数越小视场越大
//        new_intrinsic_mat.at<double>(0, 0) *= 0.5;
//        new_intrinsic_mat.at<double>(1, 1) *= 0.5;
//        //调节校正图中心，建议置于校正图中心
//        new_intrinsic_mat.at<double>(0, 2) = 0.5 * distort_img.cols;
//        new_intrinsic_mat.at<double>(1, 2) = 0.5 * distort_img.rows;

//        fisheye::undistortImage(distort_img, undistort_img, intrinsic_matrix, distortion_coeff, new_intrinsic_mat);
//        imshow("output.jpg", undistort_img);
//        waitKey();
//    }

    return 0;
}
















