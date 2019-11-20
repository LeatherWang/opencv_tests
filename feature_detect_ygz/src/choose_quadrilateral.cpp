

#include "read_intrinsic.h"

cv::Mat imageRawGray, imageRawBGR;
cv::Mat orgImage;
cv::Point prePointOfRect, curPointOfRect;

void mouseCallbackForPoly(int event, int x, int y, int flags, void *ustc)//event鼠标事件代号，x,y鼠标坐标，flags拖拽和键盘操作的代号
{
    if (event == CV_EVENT_LBUTTONDOWN)
    {
        prePointOfRect.x = x;
        prePointOfRect.y = y;
        imageRawBGR.copyTo(orgImage);
        cv::imshow("image", orgImage);
    }
    else if( (event == CV_EVENT_MOUSEMOVE) && (flags & CV_EVENT_LBUTTONDOWN))
    {
        curPointOfRect.x = x;
        curPointOfRect.y = y;
        imageRawBGR.copyTo(orgImage);
        cv::rectangle(orgImage, prePointOfRect, curPointOfRect, cv::Scalar(0, 255, 0), 1);
        cv::imshow("image", orgImage);
    }
    else if(event==CV_EVENT_LBUTTONUP)
    {
        //        prePointOfRect = Point(116, 25);
        //        curPointOfRect = Point(135, 42);
        cout<<"prePointOfRect: "<<prePointOfRect<<
              "curPointOfRect: "<<curPointOfRect<<endl;

        cv::Rect roiRect(prePointOfRect, curPointOfRect);
        cv::Mat roiImage = imageRawGray(roiRect);
        cv::Mat outImage, tmpImage;
        roiImage.copyTo(outImage);
        roiImage.copyTo(tmpImage);

        double minValue, maxValue;
        cv::minMaxIdx(outImage, &minValue, &maxValue);

        double k = 255.0f/(maxValue - minValue);

        for(int y=0; y<outImage.rows; y++)
        {
            uchar* rowIndex = outImage.ptr(y);
            for(int x=0; x<outImage.cols; x++)
                rowIndex[x] =(uchar)(k*(rowIndex[x]-minValue));
        }

        namedWindow("roiImage", WINDOW_NORMAL);
        cv::resizeWindow("roiImage", ((outImage.cols*30>800)?800:outImage.cols*30),
                         ((outImage.rows*30>800)?800:outImage.rows*30));
        imshow("roiImage", outImage);



//        for(int y=0; y<roiImage.rows; y++)
//        {
//            uchar* rowPtr = roiImage.ptr<uchar>(y);
//            for(int x=0; x<roiImage.cols; x++)
//                cout<< (int)rowPtr[x] <<" ";
//            cout<<endl;
//        }
//        cout<<endl;


        Mat drawedImage;
        roiImage.copyTo(drawedImage);
        if(drawedImage.channels() == 1)
            cvtColor(drawedImage, drawedImage, CV_GRAY2BGR);

        if(0)
        {
            cv::Mat filterImg;
            cv::Mat kernel = (Mat_<char>(6,6)<<
                               1,  1, -1, -1,  1,  1,
                               1,  1, -1, -1,  1,  1,
                              -1, -1,  0,  0, -1, -1,
                              -1, -1,  0,  0, -1, -1,
                               1,  1, -1, -1,  1,  1,
                               1,  1, -1, -1,  1,  1);
            cv::filter2D(tmpImage, filterImg, CV_8UC1, kernel, Point(2, 2), 128);
            namedWindow("filterImg", WINDOW_NORMAL);
            imshow("filterImg", filterImg);
            cv::resizeWindow("filterImg", ((filterImg.cols*30>800)?800:filterImg.cols*30),
                             ((filterImg.rows*30>800)?800:filterImg.rows*30));

            for(int y=0; y<filterImg.rows; y++)
            {
                uchar* rowPtr = filterImg.ptr<uchar>(y);
                for(int x=0; x<filterImg.cols; x++)
                    cout<< (int)rowPtr[x] <<" ";
                cout<<endl;
            }
            cout<<endl;
        }
        else
        {
            int kernelSize = 7;
            int kernelSizeHalf = kernelSize/2;
            int thresh = 6, thresh2=5;
            for(int y=0; y<tmpImage.rows-kernelSize; y++)
            {
                uchar* rowPtr = tmpImage.ptr<uchar>(y);
                for(int x=0; x<tmpImage.cols-kernelSize; x++)
                {
                    uchar pattern1[4], pattern2[4];
                    pattern1[0] = rowPtr[x+3]; //up
                    pattern1[1] = rowPtr[x+   3*tmpImage.cols]; //left
                    pattern1[2] = rowPtr[x+6+ 3*tmpImage.cols]; //right
                    pattern1[3] = rowPtr[x+3+ 6*tmpImage.cols]; //down

                    pattern2[0] = rowPtr[x+1+ 1*tmpImage.cols];
                    pattern2[1] = rowPtr[x+5+ 1*tmpImage.cols];
                    pattern2[2] = rowPtr[x+1+ 5*tmpImage.cols];
                    pattern2[3] = rowPtr[x+5+ 5*tmpImage.cols];

                    uchar centerPix = rowPtr[x+3+ 3*tmpImage.cols];
                    //                for(int i=0; i<4; i++)
                    //                    cout<<(int)pattern1[i]<<" ";
                    //                for(int i=0; i<4; i++)
                    //                    cout<<(int)pattern2[i]<<" ";
                    //                cout<<endl;

                    bool flag = false;
                    if(abs(centerPix - pattern1[0]) < thresh)
                        flag = true;
                    if(     ((abs(centerPix-pattern1[1])<thresh) != flag) ||
                            ((abs(centerPix-pattern1[2])<thresh) != flag) ||
                            ((abs(centerPix-pattern1[3])<thresh) != flag))
                        continue;
                    flag = !flag;
                    if(     ((abs(centerPix-pattern2[0])<thresh2) != flag) ||
                            ((abs(centerPix-pattern2[1])<thresh2) != flag) ||
                            ((abs(centerPix-pattern2[2])<thresh2) != flag) ||
                            ((abs(centerPix-pattern2[3])<thresh2) != flag))
                        continue;

                    drawedImage.at<cv::Vec3b>(y+kernelSizeHalf, x+kernelSizeHalf) = cv::Vec3b(0,255,0);
                    //                circle(drawedImage, Point(x+kernelSizeHalf, y+kernelSizeHalf), 1, cv::Scalar(0,255,0), 1);
                }
            }

            //imshow("drawedImage", drawedImage);
            namedWindow("drawedImage", WINDOW_NORMAL);
            imshow("drawedImage", drawedImage);
            cv::resizeWindow("drawedImage", ((drawedImage.cols*30>800)?800:drawedImage.cols*30),
                             ((drawedImage.rows*30>800)?800:drawedImage.rows*30));
        }
    }
}

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

    //    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    //    clahe->apply(imageRawGray, imageRawGray);

    cv::remap(imageRawGray, imageRawGray, intrinsicAndUndistort.mapx,
              intrinsicAndUndistort.mapy, cv::INTER_LINEAR);

//    cv::GaussianBlur(imageRawGray, imageRawGray, Size(3,3), 0);
//    cv::resize(imageRawGray, imageRawGray, imageRawGray.size()/2);

    //    cv::Canny(imageRawGray, imageRawGray, 15, 30);

    if(imageRawGray.channels() == 1)
        cv::cvtColor(imageRawGray, imageRawBGR, CV_GRAY2BGR);


    cv::namedWindow("image");
    cv::setMouseCallback("image", mouseCallbackForPoly, 0);
    cv::imshow("image", imageRawBGR);


    cv::waitKey();
    return 0;
}

