
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
    int width = 9;
    int height = 6;
    int cube = 100; //PPI
    int edge = 10;
    
    if(argc>1)
    {
        istringstream iss(string(argv[1])+" "+string(argv[2]));
        iss>>width>>height;
        cout<<width<<" "<<height<<endl;
    }

    Mat image(height*cube,width*cube,CV_8UC1,Scalar::all(0));//630*900

    for(int j = 0;j<image.rows;j++)
    {
        uchar *data =image.ptr<uchar>(j);
        for(int i=0;i<image.cols;i+=1)
        {
            if((i/cube+j/cube)%2)//符合此规律的像素，置255
            {
                data[i] = 255;
            }
        }
    }

    //初始化参数：边框的粗细
    int top = (int) (0.02*image.cols);
    int bottom = (int) (0.02*image.cols);
    int left = (int) (0.02*image.cols);
    int right = (int) (0.02*image.cols);

    Mat image_edge = image;
    cv::copyMakeBorder(image,image_edge,top,bottom,left,right,
                       cv::BORDER_CONSTANT,Scalar::all(255));

    //imshow("image_edge",image_edge);
    imwrite("board-"+std::to_string(width)+"-"+std::to_string(height)+".jpg",image_edge);
    //waitKey(0);

    return 0;
}
