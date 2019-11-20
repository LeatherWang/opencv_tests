
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <string>
#include <algorithm>

using namespace cv;
using namespace std;

//A4 size: 20.9*29.6
//A3 size: 29.6*41.91
int main(int argc, char** argv)
{
    int widthCubeNum=9, heightCubeNum=6;
    if(argc>1)
    {
        istringstream iss(string(argv[1])+" "+string(argv[2]));
        iss>>widthCubeNum>>heightCubeNum;
        cout<<widthCubeNum<<" "<<heightCubeNum<<endl;
    }

    float widthA4Physics = 41.91f;//29.6f; //cm
    float heightA4Physics = 29.6f;//20.9f; //1.416267943
    float ppi = 120.0f;
    float cubeSizePhysics = 4.1f; //cm

    int widthA4Pixel = widthA4Physics*ppi;
    int heightA4Pixel = heightA4Physics*ppi;
    int cubeSizePixel = cubeSizePhysics*ppi;

    //初始化参数：边框的粗细
    int edgeInitSizePixel = widthA4Pixel*0.05;
    int widthA4PixelMinusEdge = widthA4Pixel - 2*edgeInitSizePixel;
    int heightA4PixelMinusEdge = heightA4Pixel - 2*edgeInitSizePixel;
    


    int widthAllCubePixel = widthCubeNum * cubeSizePixel;
    int heightAllCubePixel = heightCubeNum * cubeSizePixel;

    if(widthAllCubePixel > widthA4PixelMinusEdge || heightAllCubePixel>heightA4PixelMinusEdge)
    {
        cout<<"widthAllCubePixel: "<<widthAllCubePixel<<" widthA4PixelMinusEdge: "<<widthA4PixelMinusEdge<<endl<<
              " heightAllCubePixel: "<<heightAllCubePixel<<" heightA4PixelMinusEdge: "<<heightA4PixelMinusEdge<<endl;
        cout<<"error, out of range!!!!"<<endl<<endl;
        return 1;
    }

    int top = edgeInitSizePixel;
    int bottom = heightA4Pixel - heightAllCubePixel - edgeInitSizePixel;

    int left = edgeInitSizePixel;
    int right = widthA4Pixel - widthAllCubePixel - edgeInitSizePixel;


    Mat image(heightAllCubePixel,widthAllCubePixel,CV_8UC1,Scalar::all(0));

    for(int j = 0;j<image.rows;j++)
    {
        uchar *data =image.ptr<uchar>(j);
        for(int i=0;i<image.cols;i+=1)
        {
            if((i/cubeSizePixel+j/cubeSizePixel)%2)//符合此规律的像素，置255
            {
                data[i] = 255;
            }
        }
    }

    Mat image_edge = image;
    cv::copyMakeBorder(image,image_edge,top,bottom,left,right,
                       cv::BORDER_CONSTANT,Scalar::all(255));

    //imshow("image_edge",image_edge);
    imwrite("board-"+std::to_string(widthCubeNum)+"-"+std::to_string(heightCubeNum)+".jpg",image_edge);
    //waitKey(0);

    return 0;
}
