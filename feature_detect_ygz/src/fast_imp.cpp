

#include "read_intrinsic.h"

cv::Mat imageRawGray, imageRawBGR;
cv::Point prePointOfRect, curPointOfRect;

int main(int argc, char** argv)
{
    string file_path = "./camera.yaml"; // camera_fisheye_ankobot
    cout<<"intrinsic file: "<<file_path<<endl;

    //cv::Mat DistCoef_Zero = cv::Mat::zeros(4, 1, CV_64F);
    if(!readIntrinsic(file_path, intrinsicAndUndistort.K_Mat,
                      intrinsicAndUndistort.DistCoef, intrinsicAndUndistort.imageSize))
        return 1;

    std::string imageFileName = "./3.jpg";
    if(argc>1)
        imageFileName = argv[1];
    imageRawGray = cv::imread(imageFileName, cv::IMREAD_GRAYSCALE);

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    clahe->apply(imageRawGray, imageRawGray);

    cv::remap(imageRawGray, imageRawGray, intrinsicAndUndistort.mapx,
              intrinsicAndUndistort.mapy, cv::INTER_LINEAR);

    cv::GaussianBlur(imageRawGray, imageRawGray, Size(3,3), 0);
    cv::resize(imageRawGray, imageRawGray, imageRawGray.size()/2);

//    cv::GaussianBlur(imageRawGray, imageRawGray, Size(3,3), 0);
    // 相差180
    // 半径为3的圆
    // 非极大值抑制
    // 十字卷积试试
    Mat drawedImage;
    imageRawGray.copyTo(drawedImage);
    if(drawedImage.channels() == 1)
        cvtColor(drawedImage, drawedImage, CV_GRAY2BGR);
    Mat outImage = cv::Mat::zeros(imageRawGray.rows, imageRawGray.cols, CV_8U);

    int kernelSize = 7;
    int kernelSizeHalf = kernelSize/2;
    int thresh = 10;


    for(int y=0; y<imageRawGray.rows-kernelSize; y++)
    {
        uchar* rowPtr = imageRawGray.ptr<uchar>(y);
        for(int x=0; x<imageRawGray.cols-kernelSize; x++)
        {
            uchar pattern1[4], pattern2[4];
            pattern1[0] = rowPtr[x+3];
            pattern1[1] = rowPtr[x+   3*imageRawGray.cols];
            pattern1[2] = rowPtr[x+6+ 3*imageRawGray.cols];
            pattern1[3] = rowPtr[x+3+ 6*imageRawGray.cols];

            pattern2[0] = rowPtr[x+1+ 1*imageRawGray.cols];
            pattern2[1] = rowPtr[x+5+ 1*imageRawGray.cols];
            pattern2[2] = rowPtr[x+1+ 5*imageRawGray.cols];
            pattern2[3] = rowPtr[x+5+ 5*imageRawGray.cols];

            uchar centerPix = rowPtr[x+3+ 3*imageRawGray.cols];

            bool flag = false;
            if(abs(centerPix - pattern1[0]) < thresh)
                flag = true;
            if(     ((abs(centerPix-pattern1[1])<thresh) != flag) ||
                    ((abs(centerPix-pattern1[2])<thresh) != flag) ||
                    ((abs(centerPix-pattern1[3])<thresh) != flag))
                continue;
            flag = !flag;
            if(     ((abs(centerPix-pattern2[0])<thresh) != flag) ||
                    ((abs(centerPix-pattern2[1])<thresh) != flag) ||
                    ((abs(centerPix-pattern2[2])<thresh) != flag) ||
                    ((abs(centerPix-pattern2[3])<thresh) != flag))
                continue;

            outImage.at<uchar>(y+kernelSizeHalf, x+kernelSizeHalf) = 255;
            circle(drawedImage, Point(x+kernelSizeHalf, y+kernelSizeHalf), 3, cv::Scalar(0,255,0), 1);
        }
    }

    imshow("outImage", outImage);
    imshow("drawedImage", drawedImage);


    cv::waitKey();
    return 0;
}

