#include <iostream>  
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>    
#include <opencv2/core/core.hpp>
#include <string>

using namespace std;
using namespace cv;


static std::vector<std::vector<cv::Point>> vctvctPoint;
cv::Mat image_raw, imageHSV;
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
        cout<<"HSV: "<<imageHSV.at<cv::Vec3b>(cur_pt)<<endl;
    }
}

// usage:
// 左键单击设置边框，按空格重新开始绘制，右键完成(自动连接起点和终点，绘制成封闭的多边形)
int main(int argc, char** argv)
{
    if(argc < 1)
    {
        cout<<"argc is less than 1"<<endl;
        return 0;
    }

    std::string imageFileName = string(argv[1]);
    image_raw = cv::imread(imageFileName, cv::IMREAD_COLOR);
    cv::cvtColor(image_raw, imageHSV, CV_BGR2HSV);


    cv::namedWindow("图片");
    cv::setMouseCallback("图片", mouseCallbackForPoly, 0);
    cv::imshow("图片", image_raw);
    waitKey();
	return 0;
}

