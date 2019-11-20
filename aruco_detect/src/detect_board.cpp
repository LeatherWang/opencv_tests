#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <Eigen/Core>
#include <Eigen/Dense> // 稠密矩阵的代数运算（逆，特征值等）

using namespace cv;
using namespace std;

/**
 */
static bool readCameraParameters(string filename, Mat &camMatrix, Mat &distCoeffs) {
    FileStorage fs(filename, FileStorage::READ);
    if(!fs.isOpened())
        return false;
    fs["camera_matrix"] >> camMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    return true;
}

int main(void)
{
    cv::VideoCapture inputVideo;
    inputVideo.open(1);
    cv::Mat cameraMatrix, distCoeffs;
    // camera parameters are read from somewhere
    string strImgPath = "../camera_Parameters.yaml";
    ifstream fileImg(strImgPath);
    if(fileImg)
        readCameraParameters(strImgPath,cameraMatrix, distCoeffs);
    else
        cout<<"error"<<endl;
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(10);//cv::aruco::DICT_6X6_250

    double markerLength = 0.037; //unit: m
    double markerSeparation = markerLength/10;
    cv::Ptr<cv::aruco::GridBoard> board = cv::aruco::GridBoard::create(5, 7, markerLength, markerSeparation, dictionary);

    cv::Mat image, imageCopy;
    while (inputVideo.grab())
    {

        inputVideo.retrieve(image);
        image.copyTo(imageCopy);
        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f> > corners;
        cv::aruco::detectMarkers(image, dictionary, corners, ids);
        // if at least one marker detected
        if (ids.size() > 0)
        {
            // 检测Markers
            cv::aruco::drawDetectedMarkers(imageCopy, corners, ids);

            cv::Vec3d rvec, tvec;

            // 估计标定板的坐标系到相机坐标系之间的转换矩阵
            int valid =  cv::aruco::estimatePoseBoard(corners, ids, board, cameraMatrix, distCoeffs, rvec, tvec);
            cout<<tvec[0]<<endl;

            // if at least one board marker detected
            if(valid > 0)
                cv::aruco::drawAxis(imageCopy, cameraMatrix, distCoeffs, rvec, tvec, 0.1);
        }
        cv::imshow("out", imageCopy);
        char key = (char) cv::waitKey(10);
        if (key == 27)
            break;
    }
    return 0;
}


