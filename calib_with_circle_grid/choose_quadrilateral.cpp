#include <iostream>  
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>    
#include <opencv2/core/core.hpp>
#include <string>

using namespace std;
static std::vector<std::vector<cv::Point>> vctvctPoint;
cv::Mat image_raw;
cv::Mat org;
static std::vector<cv::Point> vctPoint;
static cv::Point ptStart(-1, -1); //初始化起点
static cv::Point cur_pt(-1, -1);  //初始化临时节点
char temp[16];
int counter=0;

void mouseCallbackForPoly(int event, int x, int y, int flags, void *ustc)//event鼠标事件代号，x,y鼠标坐标，flags拖拽和键盘操作的代号
{
    if (event == CV_EVENT_LBUTTONDOWN)
    {
        cur_pt = cv::Point(x, y);
        cv::circle(org, cur_pt, 1, cv::Scalar(255, 0, 255), CV_FILLED, CV_AA, 0);
        if(!vctPoint.empty())
            cv::line(org, vctPoint.back(), cur_pt, cv::Scalar(0, 255, 0, 0), 1, 8, 0);
        cv::imshow("图片", org);
        vctPoint.push_back(cur_pt);
    }
    else if(event==CV_EVENT_RBUTTONDOWN)
    {
        vctPoint.clear();
        image_raw.copyTo(org);
        cv::imshow("图片", org);
        cv::waitKey();
    }
}

// usage:
// 左键单击设置边框，按空格重新开始绘制，右键完成(自动连接起点和终点，绘制成封闭的多边形)
int main(int argc, char** argv)
{
    std::string imageFileName = "../7.jpg";
    if(argc>1)
        imageFileName = argv[1];
    image_raw = cv::imread(imageFileName);

    if(image_raw.size().height > 900)
        cv::resize(image_raw, image_raw, cv::Size(0,0), 0.5, 0.5);

    cv::namedWindow("图片");
    cv::setMouseCallback("图片", mouseCallbackForPoly, 0);
    while(1)
    {
        vctPoint.clear();
        image_raw.copyTo(org);
        cv::imshow("图片", org);
        cv::waitKey();

        if(vctPoint.size() > 2)
        {
            cv::line(org, vctPoint.back(), vctPoint.front(), cv::Scalar(0, 255, 0, 0), 1, 8, 0);
            cv::imshow("图片", org);

            const cv::Point * ppt[1] = { &vctPoint[0] };//取数组的首地址
            int len = vctPoint.size();
            int npt[] = { len };
            //cv::polylines(org, ppt, npt, 1, 1, cv::Scalar(0,0, 0, 0), 1, 8, 0);
            cv::Mat dst, maskImage;
            org.copyTo(maskImage);
            maskImage.setTo(cv::Scalar(0,0, 0, 0));
            cv::fillPoly(maskImage, ppt, npt, 1, cv::Scalar(255,255, 255, 255));
            image_raw.copyTo(dst ,maskImage);
            cv::imshow("抠图",dst);

            cv::Size circleGridSize(4,13);
            std::vector<cv::Point2f> imageCenters;
            bool found = cv::findCirclesGrid(dst, circleGridSize, imageCenters, cv::CALIB_CB_ASYMMETRIC_GRID);

            if(!found)
                cout<<"error, not find corners!!!"<<endl;
            else
            {
                cv::imwrite(std::to_string(counter)+".jpg", dst);
                counter++;
                cv::drawChessboardCorners(dst, circleGridSize, imageCenters, found);
                cv::imshow("提取角点", dst);
            }

            cv::waitKey();
        }
    }
	return 0;
}

