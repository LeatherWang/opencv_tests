#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense> // 稠密矩阵的代数运算（逆，特征值等）

using namespace cv;
using namespace std;

char click_flag=false;
struct userdata{
    Mat im;
    Point2f points;
};

// 鼠标回调函数
void mouseHandler(int event, int x, int y, int flags, void* data_ptr)
{
    if  ( event == EVENT_LBUTTONDOWN )
    {
        userdata *data = ((userdata *) data_ptr);
        circle(data->im, Point(x,y),3,Scalar(0,0,255), 5, CV_AA);
        imshow("Image", data->im);
        data->points = Point2f(x,y);
        click_flag = true;
    }
}

Eigen::Matrix3d eulerAnglesToRotationMatrix(Eigen::Vector3d theta)
{
    // 计算旋转矩阵的X分量
    Eigen::Matrix3d R_x;
    R_x << 1,       0,               0,
           0,       cos(theta[0]),   sin(theta[0]),
           0,       -sin(theta[0]),   cos(theta[0]);

    // 计算旋转矩阵的Y分量
    Eigen::Matrix3d R_y;
    R_y << cos(theta[1]),    0,      -sin(theta[1]),
           0,                1,      0,
           sin(theta[1]),    0,      cos(theta[1]);

    // 计算旋转矩阵的Z分量
    Eigen::Matrix3d R_z;
    R_z << cos(theta[2]),    sin(theta[2]),     0,
           -sin(theta[2]),   cos(theta[2]),     0,
           0,                0,                 1;

    // 合并
    Eigen::Matrix3d R = R_z * R_y * R_x; /** @todo 顺序*/
    return R;
}

bool isRotationMatrix(cv::Mat &R)
{
    cv::Mat Rt;
    cv::transpose(R, Rt);
    cv::Mat shouldBeIdentity = Rt * R;
    cv::Mat I = cv::Mat::eye(3,3, shouldBeIdentity.type());
    return  cv::norm(I, shouldBeIdentity) < 1e-6;
}

// 这里对应的旋转矩阵的公式为: Rz*Ry*Rx
/// 旋转矩阵转换为欧拉角时，如果出现`Y`方向旋转为`+/-90`度，则出现一个角度的自由度，导致结算结果不唯一，一般的做法是令另外一个角度为`0`
cv::Vec3d rotationMatrixToEulerAngles(cv::Mat &R)
{
    assert(isRotationMatrix(R));
    float sy = sqrt(R.at<double>(0,0) * R.at<double>(0,0) +  R.at<double>(1,0) * R.at<double>(1,0) );
    bool singular = sy < 1e-6; // true: `Y`方向旋转为`+/-90`度
    double x, y, z;
    if (!singular)
    {
        x = atan2(R.at<double>(2,1) , R.at<double>(2,2));
        y = atan2(-R.at<double>(2,0), sy);
        z = atan2(R.at<double>(1,0), R.at<double>(0,0));
    }
    else
    {
        x = atan2(-R.at<double>(1,2), R.at<double>(1,1));
        y = atan2(-R.at<double>(2,0), sy);
        z = 0;
    }
    return cv::Vec3d(x, y, z);
}

// use: ./mono_extrinc_calib frame0000 true
int main( int argc, char** argv )
{
    //!@attention @attention @attention
    //!@attention @attention @attention
    //!@attention 注意根据使用图像的不同进行修改啊!!
    string file_path = "../camera_camera_calib_UI_1221.yaml";
    cv::FileStorage fs(file_path, cv::FileStorage::READ);
    if (!fs.isOpened())
        return 1;
    cv::Mat K_Mat = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat DistCoef = cv::Mat(4, 1, CV_64F);
    cv::Mat DistCoef_Zero = cv::Mat::zeros(4, 1, CV_64F);
    cv::FileNode n = fs["distortion_parameters"];
    DistCoef.at<double>(0) = static_cast<double>(n["k1"]);
    DistCoef.at<double>(1) = static_cast<double>(n["k2"]);
    DistCoef.at<double>(2) = static_cast<double>(n["p1"]);
    DistCoef.at<double>(3) = static_cast<double>(n["p2"]);

    //DistCoef_Zero = DistCoef;

    n = fs["projection_parameters"];
    K_Mat.at<double>(0,0) = static_cast<double>(n["fx"]);
    K_Mat.at<double>(1,1) = static_cast<double>(n["fy"]);
    K_Mat.at<double>(0,2) = static_cast<double>(n["cx"]);
    K_Mat.at<double>(1,2) = static_cast<double>(n["cy"]);

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

    string image_file="../frame0000.jpg";
    if(argc>1)
        image_file= argv[1];
    Mat image_raw = imread(image_file, IMREAD_GRAYSCALE);
    if(image_raw.empty())
    {
        std::cout<<"error to open: "<<image_file<<std::endl;
        exit(-1);
    }
    cv::Mat imageUndistort;
    //imageUndistort = image_raw;
    cv::undistort(image_raw, imageUndistort, K_Mat, DistCoef); //! @attention 先去畸变
    //cv::imshow("imageUndistort", imageUndistort);
    //waitKey();
    cv::Size boardSize(6,9);
    std::vector<cv::Point2f> imageCorners; //!@attention 不要试图修改成double类型，后面的算法不通过的
    std::vector<cv::Point3f> objectCorners; //3D points

    double grid_size=2.5; //unit: cm 5.4
    for (int i=0; i<boardSize.height; i++) {  //9
        for (int j=0; j<boardSize.width; j++) //6
            objectCorners.push_back(cv::Point3f((float)i*grid_size, (float)j*grid_size, 0.0f)); /** @todo */
    }

    bool found = cv::findChessboardCorners(imageUndistort, boardSize, imageCorners);

    if(!found)
    {
        cout<<"error, not find corners!!!"<<endl;
        return 1;
    }

    // Get subpixel accuracy on the corners
    cv::cornerSubPix(imageUndistort, imageCorners,
                     cv::Size(5,5),
                     cv::Size(-1,-1),
                     cv::TermCriteria(cv::TermCriteria::MAX_ITER +
                                      cv::TermCriteria::EPS,
                                      30,		// max number of iterations
                                      0.1));     // min accuracy

    Mat r_cw, t_cw; //Twc
    if(argc>2 && string(argv[2])=="true")
    {
        /** @todo 使用ransac会出现角度相反的问题*/
        cout<<"use solvePnPRansac !!!!!"<<endl;
        cv::solvePnPRansac(objectCorners,imageCorners,K_Mat, DistCoef_Zero,r_cw,t_cw);
    }
    else
        cv::solvePnP(objectCorners,imageCorners,K_Mat, DistCoef_Zero,r_cw,t_cw);

    Mat Rcw;
    cv::Rodrigues(r_cw, Rcw);
    Mat Rwc = Rcw.t();
    cv::Vec3f rulerAngleRand = rotationMatrixToEulerAngles(Rwc);
    cout<<"x-y-z:"<<rulerAngleRand*180/M_PI<<endl;
    cout<<"t_wc:"<<(-Rwc*t_cw).t()<<endl;

    Eigen::AngleAxisd rotation_vector ( rulerAngleRand[2], Eigen::Vector3d ( 0,0,1 ) ); //沿 Z 轴旋转
    Eigen::AngleAxisd rotation_vector_y ( 0, Eigen::Vector3d ( 0,1,0 ) );    //沿 Y 轴旋转
    Eigen::AngleAxisd rotation_vector_x ( M_PI, Eigen::Vector3d ( 1,0,0 ) ); //沿 X 轴旋转
    rotation_vector =                                            //1.对于机体坐标系，ZYX，右乘
            rotation_vector*rotation_vector_y*rotation_vector_x; //2.对于世界坐标系，XYZ，左乘


    Eigen::Matrix3d RwpEigen = rotation_vector.matrix();
    cv::Mat Rwp = (cv::Mat_<double>(3,3) <<
            RwpEigen(0,0), RwpEigen(0,1), RwpEigen(0,2),
            RwpEigen(1,0), RwpEigen(1,1), RwpEigen(1,2),
            RwpEigen(2,0), RwpEigen(2,1), RwpEigen(2,2));

    Mat Rpc = Rwp.t() * Rwc;

    Mat im_dst =  Mat::zeros(imageUndistort.size(), CV_8UC1);
    warpPerspective(imageUndistort, im_dst, K_Mat*Rpc*K_Mat.inv(), imageUndistort.size());
    cout<<"error, x-y-z:"<<rotationMatrixToEulerAngles(Rpc)*180/M_PI<<endl;

    cv::drawChessboardCorners(imageUndistort, boardSize, imageCorners, found);
    cv::imshow(image_file, imageUndistort);
    cv::waitKey();

    {
        cout<<"reproject......"<<endl;
        std::vector<cv::Point2f> imageCorners;
        bool found = cv::findChessboardCorners(im_dst, boardSize, imageCorners);
        if(found)
        {
            // Get subpixel accuracy on the corners
            cv::cornerSubPix(im_dst, imageCorners,
                             cv::Size(5,5),
                             cv::Size(-1,-1),
                             cv::TermCriteria(cv::TermCriteria::MAX_ITER +
                                              cv::TermCriteria::EPS,
                                              30,		// max number of iterations
                                              0.1));     // min accuracy
            Mat r_cw, t_cw; //Twc
            if(argc>2 && string(argv[2])=="true")
            {
                /** @todo 使用ransac会出现角度相反的问题*/
                cout<<"use solvePnPRansac !!!!!"<<endl;
                cv::solvePnPRansac(objectCorners,imageCorners,K_Mat,DistCoef_Zero,r_cw,t_cw);
            }
            else
                cv::solvePnP(objectCorners,imageCorners,K_Mat,DistCoef_Zero,r_cw,t_cw);

            Mat Rcw;
            cv::Rodrigues(r_cw, Rcw);
            Mat Rwc = Rcw.t();
            cv::Vec3f rulerAngleRand = rotationMatrixToEulerAngles(Rwc);
            cout<<"x-y-z:"<<rulerAngleRand*180/M_PI<<endl;
            cout<<"t_wc:"<<(-Rwc*t_cw).t()<<endl;

            cv::drawChessboardCorners(im_dst, boardSize, imageCorners, found);
            cv::imshow(image_file, im_dst);
            cv::waitKey();
        }
        else
        {
            cv::imshow(image_file, im_dst);
            cv::imwrite("../image_dst.jpg", im_dst);
            cv::waitKey();
        }
    }

	return 0;
}

















