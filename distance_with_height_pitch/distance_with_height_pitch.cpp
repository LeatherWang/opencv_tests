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

int main(void)
{
    //读取内参
    string file_path = "../camera_Parameters.yaml";
    cv::FileStorage camera_paras(file_path, cv::FileStorage::READ);
    float fx = camera_paras["Camera.fx"];
    float fy = camera_paras["Camera.fy"];
    float cx = camera_paras["Camera.cx"];
    float cy = camera_paras["Camera.cy"];
    Eigen::Matrix3d K, K_inv;
    K << fx,0,cx,0,fy,cy,0,0,1;
    K_inv = K.inverse();
    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = camera_paras["Camera.k1"];
    DistCoef.at<float>(1) = camera_paras["Camera.k2"];
    DistCoef.at<float>(2) = camera_paras["Camera.p1"];
    DistCoef.at<float>(3) = camera_paras["Camera.p2"];
    Mat K_Mat = (Mat_<double> (3,3) << K(0,0),K(0,1),K(0,2),
                 K(1,0),K(1,1),K(1,2),
                 K(2,0),K(2,1),K(2,2));

    cout << endl << "Camera Parameters: " << endl;
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- k1: " << DistCoef.at<float>(0) << endl;
    cout << "- k2: " << DistCoef.at<float>(1) << endl;
    cout << "- p1: " << DistCoef.at<float>(2) << endl;
    cout << "- p2: " << DistCoef.at<float>(3) << endl;

    //height(mm)和pitch(弧度)
    /** @todo set your own parameters*/
    float height = 620; //可由激光测距仪测得
    float pitch = 3.1415926/2; //可由陀螺仪测得

    Eigen::Vector3d eulerAngle(pitch,0,0); //pitch,roll,yaw
    Eigen::Vector3d Ow=Eigen::Vector3d(0,0,height); //X,Y,Z

    //计算旋转矩阵和平移矩阵
    //欧拉角转旋转矩阵
    Eigen::Matrix3d R_inv = eulerAnglesToRotationMatrix(eulerAngle);
    Eigen::Matrix3d R = R_inv.transpose();
    cout<<"R="<<endl<<R<<endl;

    Eigen::Vector3d t = -R*Ow;
    cout<<"t="<<endl<<t<<endl;

    Mat img_src = imread("../frame0000.jpg");

    imshow("Image", img_src);
    userdata data;
    data.im = img_src;
    setMouseCallback("Image", mouseHandler, &data);

    cout<<"click left button of mouse to appoint the point"<<endl;

    while(1)
    {
        if(click_flag == true)
        {
            Mat matInputPoint = Mat::zeros(1,1,CV_32FC2);
            matInputPoint.at<cv::Vec2f>(0,0)[0]=data.points.x;
            matInputPoint.at<cv::Vec2f>(0,0)[1]=data.points.y;
            cout<<"before undistort, pixel_x:"<<matInputPoint.at<cv::Vec2f>(0,0)[0]<<
                  "  pixel_y:"<<matInputPoint.at<cv::Vec2f>(0,0)[1]<<endl;
            cv::undistortPoints(matInputPoint,matInputPoint,K_Mat,DistCoef,Mat(),K_Mat);

            cout<<"after undistort, pixel_x:"<<matInputPoint.at<cv::Vec2f>(0,0)[0]<<
                  "  pixel_y:"<<matInputPoint.at<cv::Vec2f>(0,0)[1]<<endl;

            Eigen::Vector3d Q(matInputPoint.at<cv::Vec2f>(0,0)[0],
                    matInputPoint.at<cv::Vec2f>(0,0)[1], 1);//(u,v,1)
            Eigen::Vector3d Qc = K_inv*Q; //归一化坐标
            Eigen::Vector3d Qw = R_inv*Qc+Ow; //将归一化坐标转换到世界坐标系

            Eigen::Vector3d Pw;
            Pw[2]=0;
            double lamda = Ow[2]/(Ow[2]-Qw[2]);
            Pw[1] = Ow[1]+lamda*(Qw[1]-Ow[1]);
            Pw[0] = Ow[0]+lamda*(Qw[0]-Ow[0]);
            cout<<"Pw="<<endl<<Pw<<endl;
            click_flag = false;
        }
        cv::waitKey(1);
    }

	return 0;
}

















