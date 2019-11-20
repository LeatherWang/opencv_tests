
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <string>
#include <algorithm>

using namespace cv;
using namespace std;

const int floorHeight = 4790;
const int floorWidth = 2990;

//A4 size: 20.9*29.6
//A3 size: 29.6*41.91
int main(int argc, char** argv)
{
    //!@attention 宽度<高度
    int widthCircleNum=4, heightCircleNum=12;

    if(argc>1)
    {
        istringstream iss(string(argv[1])+" "+string(argv[2]));
        iss>>widthCircleNum>>heightCircleNum;
        cout<<widthCircleNum<<" "<<heightCircleNum<<endl;
    }

    // 物理尺寸
    float widthA4Physics = 29.6f;//29.6f; //cm
    float heightA4Physics = 41.91f;//20.9f; //1.416267943
    float ppi = 120.0f;

    float scale = 0.95;
    float circleDiameterPhysics = 2.25f * scale;
    float circleRadiusPhysics = circleDiameterPhysics/2.0f; //cm
    float circleBaseSpacing = 3.0f * scale;
    float circleSpacing = circleBaseSpacing*2.0f; //注意

    cout<< "- 棋盘格圆点尺寸: " << endl;
    cout<< "  每一个圆点直径: " << circleDiameterPhysics << "cm" <<endl;
    cout<< "  水平方向或者垂直方向相邻的圆点的圆心距离: " << circleSpacing << "cm" <<endl;


    // 转换成像素长度
    int widthA4Pixel = widthA4Physics*ppi;
    int heightA4Pixel = heightA4Physics*ppi;
    int circleRadiusPixel = circleRadiusPhysics*ppi;

    //初始化参数：边框的粗细
    float edgeSizePhysics = heightA4Physics * 0.03;
    int edgeInitSizePixel = edgeSizePhysics * ppi;
    int widthA4PixelMinusEdge = widthA4Pixel - 2*edgeInitSizePixel;
    int heightA4PixelMinusEdge = heightA4Pixel - 2*edgeInitSizePixel;
    

    // 计算需要的像素面积
    float widthAllCirclePhysics = (widthCircleNum-1)*circleSpacing + circleBaseSpacing + circleDiameterPhysics;
    float heightAllCirclePhysics = (heightCircleNum-1)*circleBaseSpacing + circleDiameterPhysics;
    int widthAllCirclePixel = widthAllCirclePhysics*ppi;
    int heightAllCirclePixel = heightAllCirclePhysics*ppi;

//    cout<<"widthAllCirclePhysics: "<<widthAllCirclePhysics + edgeSizePhysics*2.0f<<
//          "  heightAllCirclePhysics: "<<heightAllCirclePhysics + edgeSizePhysics*2.0f<<
//          "  radio: "<<heightAllCirclePhysics/widthAllCirclePhysics<<endl;

    // 判断是否会超过A3尺寸
    if(widthAllCirclePixel > widthA4PixelMinusEdge || heightAllCirclePixel>heightA4PixelMinusEdge)
    {
        cout<<"widthAllCirclePixel: "<<widthAllCirclePixel<<" widthA4PixelMinusEdge: "<<widthA4PixelMinusEdge<<endl<<
              " heightAllCirclePixel: "<<heightAllCirclePixel<<" heightA4PixelMinusEdge: "<<heightA4PixelMinusEdge<<endl;
        cout<<"error, out of range!!!!"<<endl<<endl;
        return 1;
    }

    int top = edgeInitSizePixel;
    //int bottom = heightA4Pixel - heightAllCirclePixel - edgeInitSizePixel;
    int bottom = edgeInitSizePixel;

    int right = edgeInitSizePixel;
    //int left = widthA4Pixel - widthAllCirclePixel - edgeInitSizePixel;
    int left = edgeInitSizePixel;

    // 边缘根据已制成的底板而定
    {
        bottom = floorHeight - top - heightAllCirclePixel;
        left = right = (floorWidth - widthAllCirclePixel)/2;
    }

    cout<< "- 边缘尺寸:" << endl;
    cout<< "  上边缘宽度: " << top/ppi << "cm" <<endl;
    cout<< "  下边缘宽度: " << bottom/ppi << "cm" <<endl;
    cout<< "  左边缘宽度: " << left/ppi << "cm" <<endl;
    cout<< "  右边缘宽度: " << right/ppi << "cm" <<endl;

    cout<< "- 整个棋盘格尺寸: " <<endl;
    cout<< "  高度: " << floorHeight/ppi << "cm" <<endl;
    cout<< "  宽度: " << floorWidth/ppi << "cm" <<endl;




    Mat image(heightAllCirclePixel,widthAllCirclePixel,CV_8UC1,Scalar::all(255));

    int blobRows = heightCircleNum/2;
    int blobCols = widthCircleNum;
    int startCenterRowPixel = (circleBaseSpacing + circleRadiusPhysics)*ppi;
    for(int i=0; i<blobRows; i++)
    {
        int startCenterColPixel = circleRadiusPhysics*ppi;
        for(int j=0; j<blobCols; j++)
        {
            circle(image, Point(startCenterColPixel, startCenterRowPixel), circleRadiusPixel,
                   Scalar::all(0), -1);
            startCenterColPixel += circleSpacing*ppi;
        }
        startCenterRowPixel += circleSpacing*ppi;
    }

    blobRows = heightCircleNum/2+1;
    blobCols = widthCircleNum;
    startCenterRowPixel = circleRadiusPhysics*ppi;
    for(int i=0; i<blobRows; i++)
    {
        int startCenterColPixel = (circleBaseSpacing + circleRadiusPhysics)*ppi;
        for(int j=0; j<blobCols; j++)
        {
            circle(image, Point(startCenterColPixel, startCenterRowPixel), circleRadiusPixel,
                   Scalar::all(0), -1);
            startCenterColPixel += circleSpacing*ppi;
        }
        startCenterRowPixel += circleSpacing*ppi;
    }


    Mat image_edge = image;
    cv::copyMakeBorder(image,image_edge,top,bottom,left,right,
                       cv::BORDER_CONSTANT,Scalar::all(255));

    //imshow("image_edge",image_edge);
    string imgFileName = "board-"+std::to_string(widthCircleNum)+"-"+std::to_string(heightCircleNum)+".jpg";
    imwrite(imgFileName,image_edge);
    cout << "- write image file name: " << imgFileName <<endl;
    //waitKey(0);

    return 0;
}
