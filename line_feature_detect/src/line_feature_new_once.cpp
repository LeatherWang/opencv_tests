#include <iostream>
#include <chrono>
#include <cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <fstream>
#include <iomanip>

// 将opencv contrib 中的line_descriptor模块拿出来自己编译
//#include <opencv2/line_descriptor/descriptor.hpp>
#include "line_descriptor/descriptor.hpp"

using namespace std;
using namespace cv;
double start, duration_ms;

int DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}

void ComputeFourMaxima(const vector<int>* rotHist, const int HISTO_LENGTH, int &ind1, int &ind2, int &ind3, int &ind4)
{
    int max1=0, max2=0, max3=0, max4=0;
    int num=0;
    for(int i=0; i<HISTO_LENGTH; i++)
    {
        num = rotHist[i].size();
        if(num>max1)
        {
            max4=max3;
            max3=max2;
            max2=max1;
            max1=num;
            ind4=ind3;
            ind3=ind2;
            ind2=ind1;
            ind1=i;
        }
        else if(num>max2)
        {
            max4=max3;
            max3=max2;
            max2=num;
            ind4=ind3;
            ind3=ind2;
            ind2=i;
        }
        else if(num>max3)
        {
            max4=max3;
            max3=num;
            ind4=ind3;
            ind3=i;
        }
        else if(num>max4)
        {
            max4=num;
            ind4=i;
        }
    }

    if(max2<0.1f*(float)max1)
    {
        ind2=-1;
        ind3=-1;
    }
    else if(max3<0.1f*(float)max1)
    {
        ind3=-1;
    }
    else if(max4<0.1f*(float)max1)
    {
        ind4=-1;
    }
}

void ComputeMaxima(const vector<int>* rotHist, const int HISTO_LENGTH, const int slidingWin, int &ind1)
{
    int max1=0, num=0;
    for(int i=1; i<slidingWin; i++)
        num += rotHist[HISTO_LENGTH-i].size();
    num += rotHist[0].size();
    max1 = num;
    ind1 = 0;
    for(int i=1; i<HISTO_LENGTH; i++)
    {
        int index = i-slidingWin;
        if(index<0)
            index +=HISTO_LENGTH;
        num -= rotHist[index].size();
        num += rotHist[i].size();

        if(num>max1)
        {
            max1=num;
            ind1=i;
        }
    }
}

Mat src, src_gray;
Mat dst, detected_edges;
int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 100;
int ratioTh = 3;
int kernel_size = 3;
char* window_name = "Edge Map";

void CannyThreshold(int, void*)
{
    blur( src_gray, detected_edges, Size(3,3) );
    Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratioTh, kernel_size );
    dst = Scalar::all(0);

    src.copyTo( dst, detected_edges);
    imshow( window_name, dst );
}

bool readIntrinsic(const string &file_path, cv::Mat &K_Mat, cv::Mat &DistCoef)
{
    cv::FileStorage fs(file_path, cv::FileStorage::READ);
    if (!fs.isOpened())
        return false;

    cv::FileNode n = fs["distortion_parameters"];
    DistCoef.at<double>(0) = static_cast<double>(n["k1"]);
    DistCoef.at<double>(1) = static_cast<double>(n["k2"]);
    DistCoef.at<double>(2) = static_cast<double>(n["p1"]);
    DistCoef.at<double>(3) = static_cast<double>(n["p2"]);

    n = fs["projection_parameters"];
    K_Mat.at<double>(0,0) = static_cast<double>(n["fx"]);
    K_Mat.at<double>(1,1) = static_cast<double>(n["fy"]);
    K_Mat.at<double>(0,2) = static_cast<double>(n["cx"]);
    K_Mat.at<double>(1,2) = static_cast<double>(n["cy"]);

    cout<<"________________________________________"<<endl;
    cout << endl << "Camera Parameters: " << endl;
    cout << "- fx: " << K_Mat.at<double>(0,0) << endl;
    cout << "- fy: " << K_Mat.at<double>(1,1) << endl;
    cout << "- cx: " << K_Mat.at<double>(0,2) << endl;
    cout << "- cy: " << K_Mat.at<double>(1,2) << endl;
    cout << "- k1: " << DistCoef.at<double>(0) << endl;
    cout << "- k2: " << DistCoef.at<double>(1) << endl;
    cout << "- p1: " << DistCoef.at<double>(2) << endl;
    cout << "- p2: " << DistCoef.at<double>(3) << endl;
    cout<<"________________________________________"<<endl;
    return true;
}

const char* keys =
{
    "{@input|../data/building.jpg | input image}"
    "{ds    | 2                   | dilate element size}"
    "{es    | 2                   | erode element size}"
    "{d     | true                | dilate or erode}"
    "{r     | false               | use roi or not}"
    "{o     | false               | use ostu or not}"
    "{c     | true                | use canny or adaptive threshold}"
    "{s     | true                | use sobel or robert}"
    "{l     | false               | use lsd or ed}"
    "{f     | false               | fuse lines or not}"
    "{ca    | false               | use manual adjustment}"
    "{help h|                     | show help message}"
};


int getMean(const Mat &image)
{
    int sum = 0;
    for (int i = 0; i < image.rows; i++)
    {
        const uchar* pData = image.ptr<uchar>(i);
        for (int j = 0; j < image.cols; j++)
        {
            sum += pData[j];
        }
    }
    return (int)(sum / (image.rows * image.cols));
}

int main(int argc, char** argv)
{
    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    string file_path = "../camera_camera_calib_ak.yaml";
    cout<<"intrinsic file: "<<file_path<<endl;
    cv::Mat K_Mat = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat DistCoef = cv::Mat(4, 1, CV_64F);
    //cv::Mat DistCoef_Zero = cv::Mat::zeros(4, 1, CV_64F);
    if(!readIntrinsic(file_path, K_Mat, DistCoef))
        return 1;

    string fileName = parser.get<string>("@input");
    Mat image_raw = imread(fileName, -1);

    double start = double(getTickCount());
    cv::Mat imageUndistort;
#if 0
    //cv::undistort(image_raw, imageUndistort, K_Mat, DistCoef); //! @attention 先去畸变
    cout<<">> undistort..."<<endl;
    cv::Mat mapx, mapy; //内部存的是重映射之后的坐标，不是像素值
    cv::fisheye::initUndistortRectifyMap(K_Mat, DistCoef, cv::Mat(), K_Mat, image_raw.size(), CV_32FC1, mapx, mapy);
    cv::remap(image_raw, imageUndistort, mapx, mapy, cv::INTER_LINEAR); //双线性插值
    duration_ms = (double(getTickCount()) - start) * 1000 / getTickFrequency();
    std::cout << "   It took " << duration_ms << " ms" << std::endl;
#else
    imageUndistort = image_raw;
#endif

    if(parser.get<bool>("r")) {
        cv::Rect roiRect(0,30, 640, 300);
        //imageUndistort = imageUndistort(roiRect); //错误的用法，下面`imageUndistort.ptr<uchar>(y)`将出问题
        Mat roiImage = imageUndistort(roiRect);
        roiImage.copyTo(imageUndistort);
    }
    Mat imageUndistortForSave;
    imageUndistort.copyTo(imageUndistortForSave);
    cv::imshow("imageUndistort", imageUndistort);
    waitKey();
    if(imageUndistort.channels() == 3)
        cv::cvtColor(imageUndistort, imageUndistort, cv::COLOR_RGB2GRAY);

    if(parser.get<bool>("c"))
    {
        if(!parser.get<bool>("ca"))
        {
            cout<<">> using canny detect edge..."<<endl;
//            cv::Mat blurImg;
//            cv::blur(imageUndistort, blurImg, Size(3,3)); //!@todo
//            imshow("blur", blurImg);
//            cv::Mat gaussMImg;
//            cv::GaussianBlur( imageUndistort, gaussMImg, Size(3,3), 0, 0, BORDER_DEFAULT );
//            imshow("GaussianBlur", gaussMImg);
            cv::Mat bilImg;
            cv::bilateralFilter(imageUndistort, bilImg,5, 30, 30);
//            imshow("bilateralFilter", bilImg);
//            waitKey();

            imageUndistort = bilImg;

            int avg = getMean(imageUndistort);
            int lowerThresh = (int)avg * 0.5;
            int upperThresh = (int)avg * 1.5;

            const int cannyTh = 10;
            std::cout << " - cannyTh: " << cannyTh << std::endl;
            start = double(getTickCount());
//            Canny(imageUndistort, imageUndistort, cannyTh, cannyTh*3, 3); // Apply canny edge

            Canny(imageUndistort, imageUndistort, lowerThresh, upperThresh, 3); // Apply canny edge



            duration_ms = (double(getTickCount()) - start) * 1000 / getTickFrequency();
            std::cout << "   It took " << duration_ms << " ms" << std::endl;

            cv::imshow("canny", imageUndistort);
        }
        else
        {
            src = imageUndistort;
            dst.create( src.size(), src.type() );
            src.copyTo(src_gray);
            if(src_gray.channels() == 3)
                cvtColor( src_gray, src_gray, CV_BGR2GRAY );
            namedWindow( window_name, CV_WINDOW_AUTOSIZE );
            createTrackbar( "Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold );
            CannyThreshold(0, 0);

            waitKey();
            imageUndistort = dst;
        }
    }
    else if(!parser.get<bool>("s")) {
        cout<<">> using robert detect edge..."<<endl;
        start = double(getTickCount());
        cout<<imageUndistort.size()<<endl;
        cv::Mat grandImg = Mat::zeros(imageUndistort.size(), CV_8UC1);
        for(int y = 0; y < grandImg.rows - 1; ++y)
        {
            unsigned char* dstData = grandImg.ptr<uchar>(y);
            unsigned char* srcData = imageUndistort.ptr<uchar>(y);
            for(int x = 0; x < grandImg.cols - 1; ++x)
            {
                int tmpData = abs(srcData[x] - srcData[grandImg.cols+x+1])
                        + abs(srcData[x+1] - srcData[grandImg.cols+x]);
                dstData[x] = tmpData > 255 ? 255 : tmpData;
            }
        }
        cv::imshow("robert_raw", grandImg);
        cv::threshold(grandImg, imageUndistort, 8, 255, THRESH_BINARY);
        duration_ms = (double(getTickCount()) - start) * 1000 / getTickFrequency();
        std::cout << "   It took " << duration_ms << " ms" << std::endl;
        cv::imshow("robert_raw_th", imageUndistort);
    }
//    else {
//        int scale = 1;//默认值
//        int delta = 0;//默认值
//        int ddepth = CV_16S;//防止输出图像深度溢出
//        GaussianBlur( imageUndistort, imageUndistort, Size(3,3), 0, 0, BORDER_DEFAULT );

//        Mat grad_x, grad_y, grad;
//        Mat abs_grad_x, abs_grad_y;
//        // Gradient X x方向梯度 1,0：x方向计算微分即导数
//        //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
//        Sobel( imageUndistort, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
//        convertScaleAbs( grad_x, abs_grad_x );
//        // Gradient Y y方向梯度 0，1：y方向计算微分即导数
//        //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
//        Sobel( imageUndistort, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
//        convertScaleAbs( grad_y, abs_grad_y );

//        //近似总的梯度
//        addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

//        imshow( "sobel_raw", grad );
//        cv::threshold(grad, imageUndistort, 10, 255, THRESH_BINARY);
//        imshow( "sobel_raw_th", imageUndistort );
//        waitKey();
//    }

    if(parser.get<bool>("d"))
    {
        int eleSizeForDilate = parser.get<int>("ds");
        if(eleSizeForDilate > 0)
        {
            Mat element = getStructuringElement(MORPH_RECT,
                                                Size(2*eleSizeForDilate+1, 2*eleSizeForDilate+1),
                                                Point( eleSizeForDilate, eleSizeForDilate ));
            cv::dilate(imageUndistort, imageUndistort, element);//进行膨胀操作
        }

        int eleSizeForErode = parser.get<int>("es");
        if(eleSizeForErode > 0)
        {
            Mat elementErode = getStructuringElement(MORPH_RECT,
                                                     Size(2*eleSizeForErode+1, 2*eleSizeForErode+1),
                                                     Point( eleSizeForErode, eleSizeForErode ));
            cv::erode(imageUndistort, imageUndistort, elementErode);
        }

        if(eleSizeForDilate>0 || eleSizeForErode > 0)
            imshow("erode", imageUndistort);
    }

    vector<line_descriptor::KeyLine> keylines;
    if(parser.get<bool>("l")) {
        cout<<">> LSD detect lines..."<<endl;
        start = double(getTickCount());
        Ptr<line_descriptor::LSDDetector> lsd = line_descriptor::LSDDetector::createLSDDetector();
        lsd->detect(imageUndistort, keylines, 1.2, 1);
        duration_ms = (double(getTickCount()) - start) * 1000 / getTickFrequency();
        std::cout << "   It took " << duration_ms << " ms" << std::endl;
    }
    else {
        cout<<">> ED detect lines..."<<endl;
        line_descriptor::BinaryDescriptor::Params tmParams;
        tmParams.numOfOctave_ = 1;
        tmParams.widthOfBand_ = 7;
        tmParams.reductionRatio = 2;
        tmParams.ksize_ = 99;
        Ptr<line_descriptor::BinaryDescriptor> bd = line_descriptor::BinaryDescriptor::createBinaryDescriptor(tmParams);
        bd->detect(imageUndistort, keylines);
        std::cout << "   It took " << duration_ms << " ms" << std::endl;
    }

    int lsdNFeatures = 0;
    {
        cout<<">> resort line features by their response..."<<endl;
        sort(keylines.begin(), keylines.end(), [](const line_descriptor::KeyLine& kl1,
             const line_descriptor::KeyLine &kl2){
            return kl1.response > kl2.response;
        });

        for(int i=0; i< keylines.size(); i++)
        {
            if(keylines[i].response > 0.001)
                lsdNFeatures++;
        }

        cout<<" - lsdNFeatures: "<<lsdNFeatures<<endl;

        keylines.resize(lsdNFeatures);
        for( int i=0; i<lsdNFeatures; i++)
            keylines[i].class_id = i;
    }

    Mat imageDrawedLineBeforeFilter = Mat::zeros(imageUndistort.rows, imageUndistort.cols, CV_8UC3);
    for(line_descriptor::KeyLine keyline:keylines)
        cv::line(imageDrawedLineBeforeFilter, keyline.getStartPoint(), keyline.getEndPoint(), cv::Scalar(255), 1, CV_AA);
    cv::namedWindow("imageDrawedLineBeforeFilter", CV_WINDOW_AUTOSIZE);
    imshow( "imageDrawedLineBeforeFilter", imageDrawedLineBeforeFilter );



    const int HISTO_LENGTH = 100;
    const double factor = (double)HISTO_LENGTH/(M_PI);
    vector<int> rotHist[HISTO_LENGTH];
    vector<line_descriptor::KeyLine> keylinesFilter;
    for(int i=0; i< keylines.size(); i++)
    {
        double rot = keylines[i].angle;

        // (-pi,pi] => (0,pi]
        if(rot <= 0)
            rot += M_PI;

        int bin = round(rot * factor);// 将rot分配到bin组
        if(bin==HISTO_LENGTH)
            bin=0;
        assert(bin>=0 && bin<HISTO_LENGTH);
        int id = keylinesFilter.size();
        rotHist[bin].push_back(id); //将ID号放进去
        keylines[i].class_id = id;
        keylinesFilter.push_back(keylines[i]);
    }
    cout<<" - keylinesFilter size: "<<keylinesFilter.size()<<endl;

    // save data to plot histogram
    std::ofstream f;
    f.open("./out_for_histogram.txt");
    f << std::fixed;
    for(int i=0; i<keylinesFilter.size(); i++)
    {
        double rot = keylinesFilter[i].angle;
        // (-pi,pi] => (0,pi]
        if(rot <= 0)
            rot += M_PI;
        f << std::setprecision(6)<<(rot*180.0f/M_PI)<<" ";
    }
    f.close();

    // 滑动窗口统计
    vector<line_descriptor::KeyLine> keylinesFilterAndHist;
    cout<<">> histogram filter..."<<endl;

    double countAngle = 8.0f;
    int slidingWin = round(countAngle*HISTO_LENGTH/180.0f);
    cout<<" - slidingWin: "<<slidingWin<<endl;
    int ind1 = -1;
    ComputeMaxima(rotHist, HISTO_LENGTH, slidingWin, ind1);
    int resAngle = ind1-(slidingWin-1)/2;
    if(resAngle < 0)
        resAngle += HISTO_LENGTH;
    cout<<" - ind1: "<<ind1<<endl;
    cout<<" - resAngle: "<<180.0f*resAngle/HISTO_LENGTH<<endl;
    for(int i=0; i<slidingWin; i++)
    {
        int index = ind1-i;
        if(index<0)
            index += HISTO_LENGTH;
        for(int j=0; j<rotHist[index].size(); j++)
        {
            keylinesFilterAndHist.push_back(keylinesFilter[rotHist[index][j]]);
        }
    }
    cout<<" - keylinesFilterAndHist size: "<<keylinesFilterAndHist.size()<<endl;


    Mat imageDrawedLineAfterFilterAndHist = Mat::zeros(imageUndistort.rows, imageUndistort.cols, CV_8UC3);
    if(imageDrawedLineAfterFilterAndHist.channels() == 1)
        cv::cvtColor(imageDrawedLineAfterFilterAndHist, imageDrawedLineAfterFilterAndHist, cv::COLOR_GRAY2BGR);
    for(line_descriptor::KeyLine keyline:keylinesFilterAndHist)
    {
        cv::line(imageDrawedLineAfterFilterAndHist, keyline.getStartPoint(), keyline.getEndPoint(), cv::Scalar(0,255,0), 1, CV_AA);
    }
    cv::namedWindow("imageDrawedLineAfterFilterAndHist", CV_WINDOW_AUTOSIZE);
    imshow( "imageDrawedLineAfterFilterAndHist", imageDrawedLineAfterFilterAndHist );

    waitKey();




    cv::imwrite("image_raw.bmp", imageUndistortForSave);
    cv::imwrite("image_canny.bmp", imageUndistort);
    cv::imwrite("imageDrawedLineBeforeFilter.bmp", imageDrawedLineBeforeFilter);
//    cv::imwrite("imageDrawedLineAfterFilter.bmp", imageDrawedLineAfterFilter);
    cv::imwrite("imageDrawedLineAfterFilterAndHist.bmp", imageDrawedLineAfterFilterAndHist);

    return 0;
}
