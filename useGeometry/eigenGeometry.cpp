#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry> // Eigen 几何模块
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;

Eigen::Matrix3d eulerAnglesToRotationMatrix(Eigen::Vector3d theta)
{
    // 计算旋转矩阵的Z分量
    Eigen::Matrix3d R_z;
    R_z << cos(theta[0]),    sin(theta[0]),     0,
           -sin(theta[0]),   cos(theta[0]),     0,
           0,                0,                 1;

    // 计算旋转矩阵的Y分量
    Eigen::Matrix3d R_y;
    R_y << cos(theta[1]),    0,      -sin(theta[1]),
           0,                1,      0,
           sin(theta[1]),    0,      cos(theta[1]);

    // 计算旋转矩阵的X分量
    Eigen::Matrix3d R_x;
    R_x << 1,       0,               0,
           0,       cos(theta[2]),   sin(theta[2]),
           0,       -sin(theta[2]),   cos(theta[2]);

    // 合并
    Eigen::Matrix3d R = R_x * R_y * R_z ; /** @todo 顺序: Z-Y-X*/
    return R;
}

int main ( int argc, char** argv )
{
    // 顶视
    // -0.0121,   0.0362,    -1.6077  -- yaw pitch roll = 87.8628   179.158  178.231
    // 侧视
    // 0,         2.2214,    -2.2214  -- yaw pitch roll = 0.002376  179.998  90
    // 顶视
    // 0.01089,   0.06490,   -0.00386 -- yaw pitch roll = 179.799   176.28   -179.382

    cv::Mat rbc = (cv::Mat_<double>(3,1) << 0.01089,   0.06490,   -0.00386);
    cout<<"rotation vector =\n"<<rbc<<endl;

    //【1】旋转向量-->>旋转矩阵
    cv::Mat Rbc(3,3,CV_32F);
    cv::Rodrigues(rbc, Rbc);

    Eigen::Matrix3d rotation_matrix;
    rotation_matrix <<
            Rbc.at<double> ( 0,0 ), Rbc.at<double> ( 0,1 ), Rbc.at<double> ( 0,2 ),
            Rbc.at<double> ( 1,0 ), Rbc.at<double> ( 1,1 ), Rbc.at<double> ( 1,2 ),
            Rbc.at<double> ( 2,0 ), Rbc.at<double> ( 2,1 ), Rbc.at<double> ( 2,2 );

    cout<<"rotation matrix =\n"<<rotation_matrix<<endl;

    //【2】旋转矩阵-->>欧拉角
    Eigen::Vector3d euler_angles = rotation_matrix.eulerAngles( 2,1,0 );//ZYX顺序
    cout<<"Z--Y--X= "<<euler_angles.transpose()*180/M_PI<<endl;

    euler_angles << M_PI/2, -M_PI/2, 0; /** @attention 顺序: Z-Y-X*/

    //【3】欧拉角-->>旋转矩阵
    rotation_matrix = eulerAnglesToRotationMatrix(euler_angles);
    cout<<"rotation matrix =\n"<<rotation_matrix<<endl;

    return 0;
}



























