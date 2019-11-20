#include <iostream>  
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>    
#include <opencv2/core/core.hpp>
#include <string>

using namespace std;
static std::vector<std::vector<cv::Point>> vctvctPoint;
cv::Mat image_raw;
cv::Mat image_for_draw;
static std::vector<cv::Point> vctPoint;
static cv::Point ptStart(-1, -1);
static cv::Point start_pt(-1, -1), end_pt(-1, -1);
char temp[16];
int counter=0;
double radius=0;

void mouseCallbackForPoly(int event, int x, int y, int flags, void *ustc)//event鼠标事件代号，x,y鼠标坐标，flags拖拽和键盘操作的代号
{
    if (event == CV_EVENT_LBUTTONDOWN)
    {
        start_pt = cv::Point(x, y);
        cv::circle(image_for_draw, start_pt, 3, cv::Scalar(255, 0, 255), CV_FILLED, CV_AA, 0);
        cv::imshow("图片", image_for_draw);
    }
    else if(event==CV_EVENT_RBUTTONDOWN)
    {
        end_pt = cv::Point(x, y);
        radius = sqrt((end_pt- start_pt).dot(end_pt- start_pt));
        cv::circle(image_for_draw, start_pt, radius, cv::Scalar(255, 0, 255), 1, CV_AA, 0);
        cv::imshow("图片", image_for_draw);
//        cv::waitKey();
    }
}

// usage:
// 左键单击设置边框，按空格重新开始绘制，右键完成(自动连接起点和终点，绘制成封闭的多边形)
int main(int argc, char** argv)
{
    std::string imageFileName = "../7.jpg";
    if(argc>1)
        imageFileName = argv[1];
    image_raw = cv::imread(imageFileName, cv::IMREAD_GRAYSCALE);

    string maskName;
    if(argc > 2)
    {
        maskName = argv[2];
        cv::Mat mask = 255 - cv::imread(maskName, cv::IMREAD_GRAYSCALE);
        cv::Mat imageMasked;
        image_raw.copyTo(imageMasked, mask);
        cv::Mat imageTmp = cv::Mat(image_raw.rows, image_raw.cols*2, image_raw.type(), cv::Scalar::all(255));
        image_raw.copyTo(imageTmp.colRange(0, image_raw.cols));
        imageMasked.copyTo(imageTmp.colRange(image_raw.cols, image_raw.cols*2));
        cv::imshow("imageMasked", imageTmp);
        cv::waitKey();
        return 0;
    }


    if(image_raw.size().height > 900)
        cv::resize(image_raw, image_raw, cv::Size(0,0), 0.5, 0.5);

    cv::namedWindow("图片");
    cv::setMouseCallback("图片", mouseCallbackForPoly, 0);
    while(1)
    {
        vctPoint.clear();
        image_raw.copyTo(image_for_draw);
        cv::imshow("图片", image_for_draw);
        if(cv::waitKey() == 's')
        {
            cv::Mat mask = cv::Mat(image_raw.size(), CV_8UC1, cv::Scalar::all(0));
            cv::circle(mask, start_pt, radius, cv::Scalar(255), CV_FILLED);
            string imageName = "./t265_mask.png";
            cv::imwrite(imageName, mask);
            cout << "image is saved in: " << imageName << endl;
        }
    }



	return 0;
}

