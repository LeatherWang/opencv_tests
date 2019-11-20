/************************************************************************************************
* Undistort circle images using fisheye camera model.
************************************************************************************************/
#include <iostream>
#include <fstream>
#include <chrono>
#include <sys/time.h>
#include <opencv2/opencv.hpp>
// #include "src/CheckExtrinsics.h"       //对fisheye:calibrate函数优化时出现svd分解断言错误增加的检测函数
using namespace std;

vector<cv::Mat> gImgList; //保存每张原始的标定图片
cv::Mat imageMark;        //保存每张棋盘格角点的图片

// 检测图像上二维棋盘格角点的坐标(2D点)
bool detecChessboardCorners(const string &ImageDir, vector<vector<cv::Point2d>> &imagePoints,
                            const cv::Size &boardSize, const int &imageNum, cv::Size &imageSize)
{
    static bool firstFlags = true;
    imagePoints.resize(0);
    for (int i = 0; i < imageNum; ++i)
    {
        string imageFileName = ImageDir + "/" + to_string(i + 1) + ".jpg";
        cv::Mat img = cv::imread(imageFileName, cv::IMREAD_GRAYSCALE);   //读取并转化为灰度图
        if (img.empty())
        {
            // cout << "cannot read image : " << imageFileName << endl;
            continue;
        }

        cout << "Frame #" << i + 1 << "  ";
        gImgList.push_back(img); //保存每张原始的标定图片
        if (true == firstFlags)
        { //只初始化一次
            imageSize = img.size(); //获取图片大小
            imageMark = cv::Mat(imageSize.height, imageSize.width, CV_64FC3, cv::Scalar(0, 0, 0));
            firstFlags = false;
        }

        //开始提取角点，绘制检测到的角点并保存
        vector<cv::Point2d> tmpPoints;   //畸变过大的镜头不要用cv::CALIB_CB_FAST_CHECK参数
        //bool found = cv::findChessboardCorners(img, boardSize, tmpPoints, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE);
        bool found = cv::findCirclesGrid(img, boardSize, tmpPoints, cv::CALIB_CB_ASYMMETRIC_GRID);
        if (found && tmpPoints.size() == boardSize.width * boardSize.height)
        {
            imagePoints.push_back(tmpPoints);

            cv::Mat cornerImage = img.clone();
            cvtColor(cornerImage, cornerImage, CV_GRAY2RGB);
            for (int j = 0; j < tmpPoints.size(); ++j)
            {
                cv::Point2d tmpP = tmpPoints[j];
                circle(cornerImage, tmpP, 2, cv::Scalar(0, 0, 255), 2, 8, 0);                                              //绘制检测到的角点并保存(每张图片)
                line(imageMark, cvPoint(tmpP.x - 5, tmpP.y), cvPoint(tmpP.x + 5, tmpP.y), cv::Scalar(0, 0, 255), 2, 8, 0); //检测的所有角点都在一张图上显示
                line(imageMark, cvPoint(tmpP.x, tmpP.y - 5), cvPoint(tmpP.x, tmpP.y + 5), cv::Scalar(0, 0, 255), 2, 8, 0);
            }
            // 保存检测到的角点图像
            string cornerImageFile = ImageDir + "/" + to_string(i + 1) + "_corner.jpg";
            cv::imwrite(cornerImageFile, cornerImage);
        }
        cout << "circle corners " << (found ? "FOUND" : "NOT FOUND") << endl;
    }
    //保存所有提取的角点图片
    std::string mMarkImageFileName = ImageDir + "/imageMark.jpg";
    cv::imwrite(mMarkImageFileName, imageMark);

    cout << "Find corner images = " << imagePoints.size() << endl;
    if (imagePoints.size() < 3)
        return false;
    else
        return true;
}

// 生成三维圆点的坐标(3D点)(不同位姿棋盘格的3D坐标是一样的)
void calcChessboardCorners(const cv::Size &boardSize, const cv::Size2d &squareSize, vector<cv::Point3d> &corners)
{
    corners.clear();
    for (int y = 0; y < boardSize.height; ++y)
    {
        double tmpY = y * squareSize.width * 0.5;
        double tmpX = (y % 2 * 0.5) * squareSize.height;
        for (int x = 0; x < boardSize.width; ++x)
            corners.push_back(cv::Point3d(tmpX + x * squareSize.width, tmpY, 0.0));
    }
}

// 计算重投影误差
double ComputeReprojectionErrors(const vector<vector<cv::Point3d>> &objectPoints, const vector<vector<cv::Point2d>> &imagePoints, 
    const vector<cv::Vec3d> &rvecs, const vector<cv::Vec3d> &tvecs,
    const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs, vector<double> &perViewErrors)
{
    double totalErr = 0;
    size_t totalPoints = 0;    
    vector<cv::Point2d> imagePoints2;
    perViewErrors.resize(objectPoints.size());
    for (size_t i = 0; i < objectPoints.size(); ++i)
    {
        cv::fisheye::projectPoints(objectPoints[i], imagePoints2, rvecs[i], tvecs[i], cameraMatrix, distCoeffs);
        double err = norm(imagePoints[i], imagePoints2, cv::NORM_L2);

        size_t n = objectPoints[i].size();
        perViewErrors[i] = (double)std::sqrt(err * err / n);
        totalErr += err * err;
        totalPoints += n;
    }

    return sqrt(totalErr / totalPoints);
}

// 标定参数写到camera.yaml文件中
void saveCameraParams(const string &fileDir, const cv::Size &img_size, const cv::Mat &kk, const cv::Mat &dd, const double &rms)
{
    const string outputYamlFile = fileDir + "/camera.yaml";
    ofstream fs(outputYamlFile);

    fs << "%YAML:1.0" << endl;
    fs << "    " << endl;
    fs << "# fisheye camera calibration and distortion parameters (OpenCV)" << endl;
    fs << "# Camera Number: " << fileDir << endl;

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
    fs << "Camera.fx: " << kk.at<double>(0, 0) << endl;
    fs << "Camera.fy: " << kk.at<double>(1, 1) << endl;
    fs << "Camera.cx: " << kk.at<double>(0, 2) << endl;
    fs << "Camera.cy: " << kk.at<double>(1, 2) << endl;

    fs << "Camera.k1: " << dd.at<double>(0) << endl;
    fs << "Camera.k2: " << dd.at<double>(1) << endl;
    fs << "Camera.p1: " << dd.at<double>(2) << endl;
    fs << "Camera.p2: " << dd.at<double>(3) << endl;
    if (dd.total() > 4)
        fs << "Camera.k3: " << dd.at<double>(4) << endl;
    fs << endl;

    fs << "# rms = " << rms << endl;

    // fs << "camera_matrix = " << kk << endl;
    // fs << "distortion_coefficients = " << dd << endl;

    fs.close();
}

// 保存去畸变后的图片到本地文件夹
void saveUndistortImage(const string &fileDir, const cv::Mat &KK, const cv::Mat &DD, const cv::Size &image_Size)
{
    cv::Mat mapx = cv::Mat(image_Size, CV_32FC1);
    cv::Mat mapy = cv::Mat(image_Size, CV_32FC1);
    cv::Mat R = cv::Mat::eye(3, 3, CV_32F);
    for (int i = 0; i < gImgList.size(); ++i)
    {
        //cout << "Frame #" << i + 1 << "..." << endl;
        cv::fisheye::initUndistortRectifyMap(KK, DD, R, KK, image_Size, CV_32FC1, mapx, mapy);
        //fisheye::initUndistortRectifyMap(intrinsic_matrix, distortion_coeffs, R,
        //  getOptimalNewCameraMatrix(intrinsic_matrix, distortion_coeffs, image_size, 1, image_size, 0), image_size, CV_32FC1, mapx, mapy);
        cv::Mat mUndistortImage;
        cv::remap(gImgList[i], mUndistortImage, mapx, mapy, cv::INTER_LINEAR);
        string undistortImageFile = fileDir + "/" + std::to_string(i+1) + "_undistort.jpg";
        cv::imwrite(undistortImageFile, mUndistortImage);
    }
}


int main(int argc, char **argv)
{
    if (argc != 2)
    {
        cout << "Usage: ./fisheye_calibration ../images" << endl;
        return -1;
    }
    
    // 配置标定相关的参数-------------------------------------------------------------------------
    const cv::Size boardSize(4, 11);     //非对称圆点标定板
    const cv::Size2d squareSize(40, 40); //圆点水平或垂直方向 的距离
    const int imageNumber = 50;          //标定的图片数
    cv::Size imageSize;                  //在detecChessboardCorners里获取图片大小

    // 参数校验----------------------------------------------------------------------------------
    cout << "Check calibrate.yaml parameters..." << endl;
    cout << "   boardSize = " << boardSize << endl;
    cout << "   squareSize = " << squareSize << endl;
    cout << "   imageNumber = " << imageNumber << endl;

    // 检测图像上二维圆点坐标(2D点)-----------------------------------------------------------
    vector<vector<cv::Point2d>> imagePoints;   //二维圆点坐标(2D点)
    bool ret = detecChessboardCorners(argv[1], imagePoints, boardSize, imageNumber, imageSize);
    if (!ret)
    {
        cout << "[ERROR], Not enough corner detected images!" << endl;
        return -1;
    }

    // 生成三维棋盘格角点坐标(3D点)(不同位姿棋盘格的3D坐标是一样的)-------------------------------------
    vector<vector<cv::Point3d>> objectPoints(1); //三维棋盘格角点坐标(3D点)
    calcChessboardCorners(boardSize, squareSize, objectPoints[0]);
    objectPoints.resize(imagePoints.size(), objectPoints[0]);

    // run calibrate--------------------------------------------------------------------------
    // K:内参  D:畸变系数
    cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat D = cv::Mat::zeros(4, 1, CV_64F);
    vector<cv::Vec3d> rvecs, tvecs; //外参vector
    int flags = 0;                  //标定参数选项
    flags |= cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
    flags |= cv::fisheye::CALIB_FIX_SKEW;
    cout << endl << "fisheye::calibrate [Begin]" << endl;
    double Rms = cv::fisheye::calibrate(objectPoints, imagePoints, imageSize, K, D, rvecs, tvecs, flags, cv::TermCriteria(3, 400, 1e-8));
    cout << "fisheye::calibrate [End]" << endl;

    // 计算重投影误差-----------------------------------------------------------------------------
    cout << "compute reprojection error..." << endl;
    vector<double> reprojErrs;
    double totalAvgErr = ComputeReprojectionErrors(objectPoints, imagePoints, rvecs, tvecs, K, D, reprojErrs);

    // 输出标定结果-------------------------------------------------------------------------------
    cout << "calibrate rms = " << Rms << endl;
    cout << "calibrate reprojection error = " << totalAvgErr << endl;

    // 保存camera内参到camera.yaml文件中
    cout << "Saving camera params to camera.yaml" << endl;
    saveCameraParams(argv[1], imageSize, K, D, Rms);

    // 保存去畸变后的图片到本地文件夹----------------------------------------------------------------
    cout << "Saving undistort images..." << endl;
    saveUndistortImage(argv[1], K, D, imageSize);

    return 0;
}
