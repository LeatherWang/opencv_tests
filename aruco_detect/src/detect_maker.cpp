#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <Eigen/Core>
#include <Eigen/Dense> // 稠密矩阵的代数运算（逆，特征值等）
#include <./src/hello/libHelloSLAM.h>

using namespace cv;
using namespace std;

static bool readCameraParameters(string filename, Mat &camMatrix, Mat &distCoeffs)
{
    FileStorage fs(filename, FileStorage::READ);
    if(!fs.isOpened())
        return false;
    fs["camera_matrix"] >> camMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    return true;
}

int main(void)
{
    cv::Mat cameraMatrix, distCoeffs; // camera parameters are read from somewhere
    readCameraParameters("../camera_Parameters.yaml",cameraMatrix, distCoeffs);

    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(4);//cv::aruco::DICT_6X6_250
    cv::Mat image;
    cv::Mat imageCopy;

    //image = cv::imread("./frame0000.jpg");
    cv::VideoCapture inputVideo;
    inputVideo.open(1);

    while (inputVideo.grab())
    {
        inputVideo.retrieve(image);

        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f> > corners;
        aruco::detectMarkers(image, dictionary, corners, ids);

        image.copyTo(imageCopy);

        if (ids.size() > 0)
        {
            cv::aruco::drawDetectedMarkers(imageCopy, corners, ids);

            vector< Vec3d > rvecs, tvecs;

            double markerLength = 0.1; //unit: m

            cv::aruco::estimatePoseSingleMarkers(corners, markerLength, cameraMatrix, distCoeffs, rvecs, tvecs); // draw axis for each marker
            std::cout<<rvecs[0]<<tvecs[0]<<rvecs.size()<<endl;

            for(int i=0; i<ids.size(); i++)
                cv::aruco::drawAxis(imageCopy, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.1);
        }

        cv::imshow("image_out", imageCopy);
        waitKey(0);
    }

    return 0;
}


