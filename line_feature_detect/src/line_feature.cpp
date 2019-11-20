#include <iostream>
#include <chrono>
#include <cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/line_descriptor/descriptor.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <fstream>
#include <iomanip>

// 将opencv contrib 中的line_descriptor模块拿出来自己编译
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

int main(int argc, char** argv)
{
    std::string in;
    cv::CommandLineParser parser(argc, argv, "{@input|../data/building.jpg|input image}{help h||show help message}");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    string file_path = "../camera_camera_calib_ak.yaml";
    cout<<"intrinsic file: "<<file_path<<endl;
    cv::FileStorage fs(file_path, cv::FileStorage::READ);
    if (!fs.isOpened())
        return 1;
    cv::Mat K_Mat = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat DistCoef = cv::Mat(4, 1, CV_64F);
    cv::Mat DistCoef_Zero = cv::Mat::zeros(4, 1, CV_64F);
    cv::FileNode n = fs["distortion_parameters"];
    DistCoef.at<double>(0) = static_cast<double>(n["k1"]);
    DistCoef.at<double>(1) = static_cast<double>(n["k2"]);
    DistCoef.at<double>(2) = static_cast<double>(n["p1"]);
    DistCoef.at<double>(3) = static_cast<double>(n["p2"]);

    //DistCoef_Zero = DistCoef;

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


    in = parser.get<string>("@input");
    Mat image_raw = imread(in, -1);

    cv::Mat imageUndistort;
#if 1
    //cv::undistort(image_raw, imageUndistort, K_Mat, DistCoef); //! @attention 先去畸变
    cv::Mat mapx, mapy; //内部存的是重映射之后的坐标，不是像素值
    cv::fisheye::initUndistortRectifyMap(K_Mat, DistCoef, cv::Mat(), K_Mat, image_raw.size(), CV_32FC1, mapx, mapy);
    cv::remap(image_raw, imageUndistort, mapx, mapy, cv::INTER_LINEAR); //双线性插值
#else
    imageUndistort = image_raw;
#endif
    Mat imageUndistortForSave;
    imageUndistort.copyTo(imageUndistortForSave);
    //cv::imshow("imageUndistort", imageUndistort);
    //waitKey();

#if 1
    cout<<">> using canny detect edge..."<<endl;
    double start = double(getTickCount());
    if(imageUndistort.channels() == 3)
        cv::cvtColor(imageUndistort, imageUndistort, cv::COLOR_RGB2GRAY);
    cv::blur(imageUndistort, imageUndistort, Size(3,3));
    const int cannyTh = 10;
    std::cout << " - cannyTh: " << cannyTh << std::endl;
    Canny(imageUndistort, imageUndistort, cannyTh, cannyTh*3, 3); // Apply canny edge

    duration_ms = (double(getTickCount()) - start) * 1000 / getTickFrequency();
    std::cout << "   It took " << duration_ms << " ms" << std::endl;

    cv::imshow("canny", imageUndistort);
    waitKey();
#else
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
#endif

/*
    // Create and LSD detector with standard or no refinement.
#if 1
    Ptr<LineSegmentDetector> ls = createLineSegmentDetector(LSD_REFINE_STD);
#else
    Ptr<LineSegmentDetector> ls = createLineSegmentDetector(LSD_REFINE_NONE);
#endif
    double start = double(getTickCount());
    vector<Vec4f> lines_std;
    // Detect the lines
    ls->detect(imageUndistortGray, lines_std);
    double duration_ms = (double(getTickCount()) - start) * 1000 / getTickFrequency();
    std::cout << "  It took " << duration_ms << " ms" << std::endl;
    // Show found lines
    Mat drawnLines(imageUndistortGray);
    ls->drawSegments(drawnLines, lines_std);
    imshow("Standard refinement", drawnLines);
    waitKey();*/


    cout<<">> LSD detect lines..."<<endl;
    start = double(getTickCount());
    vector<line_descriptor::KeyLine> keylines;
    Ptr<line_descriptor::LSDDetector> lsd = line_descriptor::LSDDetector::createLSDDetector();
    lsd->detect(imageUndistort, keylines, 1.2,1);
    duration_ms = (double(getTickCount()) - start) * 1000 / getTickFrequency();
        std::cout << "   It took " << duration_ms << " ms" << std::endl;


    int lsdNFeatures = keylines.size();
    cout<<">> resort line features by their response..."<<endl;
    cout<<" - lsdNFeatures: "<<lsdNFeatures<<endl;
    // 取前50个响应最大的线
    if(keylines.size()>lsdNFeatures)
    {
        sort(keylines.begin(), keylines.end(), [](const line_descriptor::KeyLine& kl1,
             const line_descriptor::KeyLine &kl2){
            return kl1.response > kl2.response;
        });
        keylines.resize(lsdNFeatures);
        for( int i=0; i<lsdNFeatures; i++)
            keylines[i].class_id = i;
    }

    // 计算描述子
    cout<<">> LBD compute descriptor..."<<endl;
    start = double(getTickCount());
    Mat mLdesc;
    Ptr<line_descriptor::BinaryDescriptor> lbd = line_descriptor::BinaryDescriptor::createBinaryDescriptor();
    lbd->compute(imageUndistort, keylines, mLdesc);
    duration_ms = (double(getTickCount()) - start) * 1000 / getTickFrequency();
        std::cout << "   It took " << duration_ms << " ms" << std::endl;

//    Mat imageAddedLine;
//    line_descriptor::drawKeylines(imageUndistort, keylines, imageAddedLine, Scalar::all( -1 ));
//    cv::namedWindow("imageAddedLine", CV_WINDOW_NORMAL);
//    imshow( "imageAddedLine", imageAddedLine );

    Mat imageDrawedLineBeforeFilter = Mat::zeros(imageUndistort.rows, imageUndistort.cols, CV_8UC3);
    for(line_descriptor::KeyLine keyline:keylines)
        cv::line(imageDrawedLineBeforeFilter, keyline.getStartPoint(), keyline.getEndPoint(), cv::Scalar(255), 1, CV_AA);
    cv::namedWindow("imageDrawedLineBeforeFilter", CV_WINDOW_NORMAL);
    imshow( "imageDrawedLineBeforeFilter", imageDrawedLineBeforeFilter );

     cout<<">> filter lines..."<<endl;
     start = double(getTickCount());
    int fuseCounter = 0;
    const double angleTh = 5.0/180.0*M_PI;
    const double sigma = 1.0;
    const double invSigmaSquare = 1.0/(sigma*sigma);
    const double p2pDistTh = 150.991;
    const double p2lineDistTh = 15.991;
    const int TH_LOW = 200;
    vector<bool> fused(keylines.size(), false);
    for(int i=0; i< keylines.size(); i++)
    {
        if(fused[i])
            continue;
        // a1*x+b1*y+c1=0
        double a1, b1, c1;
        float deltaX = keylines[i].getEndPoint().x - keylines[i].getStartPoint().x;
        if(fabs(deltaX) < 0.01)
        {
            a1 = 1;
            b1 = 0;
            c1 = -keylines[i].getStartPoint().x;
        }
        else
        {
            float deltaY = keylines[i].getEndPoint().y - keylines[i].getStartPoint().y;
            a1 = deltaY/deltaX;
            b1 = -1;
            c1 = keylines[i].getStartPoint().y - a1*keylines[i].getStartPoint().x;
        }
        for(int j=i+1; j<keylines.size(); j++)
        {
            if(fused[j])
                continue;

            double angleDiff=keylines[j].angle-keylines[i].angle;
            while(angleDiff>M_PI)
                angleDiff -= 2.0*M_PI;
            while(angleDiff<-M_PI)
                angleDiff += 2.0*M_PI;

            //【1】角度距离
            if(fabs(angleDiff) > angleTh)
                continue;

            //【2】点到线的距离
            Point2f midPoint = Point2f((keylines[j].getStartPoint().x + keylines[j].getEndPoint().x)/2.0,
                                       (keylines[j].getStartPoint().y + keylines[j].getEndPoint().y)/2.0);

            const float num1 = a1*midPoint.x + b1*midPoint.y + c1;
            const double squareDist = num1*num1/(a1*a1+b1*b1); // 点到线的几何距离 的平方
            const double chiSquare = squareDist*invSigmaSquare;
            if(chiSquare > p2lineDistTh)
                continue;

            //【3】端点到端点的距离
            double squareDist1 = (keylines[j].getStartPoint().x-keylines[i].getEndPoint().x)*(keylines[j].getStartPoint().x-keylines[i].getEndPoint().x)+
                    (keylines[j].getStartPoint().y-keylines[i].getEndPoint().y)*(keylines[j].getStartPoint().y-keylines[i].getEndPoint().y);
            const double chiSquare1 = squareDist1*invSigmaSquare;
            double squareDist2 = (keylines[j].getEndPoint().x-keylines[i].getStartPoint().x)*(keylines[j].getEndPoint().x-keylines[i].getStartPoint().x)+
                    (keylines[j].getEndPoint().y-keylines[i].getStartPoint().y)*(keylines[j].getEndPoint().y-keylines[i].getStartPoint().y);
            const double chiSquare2 = squareDist2*invSigmaSquare;
            if(chiSquare1 >= p2pDistTh && chiSquare2 >= p2pDistTh)
                continue;

            //【4】描述子距离
            int distOfDescriptor = DescriptorDistance(mLdesc.row(i), mLdesc.row(j)); //256维
            if(distOfDescriptor < TH_LOW)
            {
                //! @todo
                if(chiSquare1 < p2pDistTh)
                {
                    keylines[i].endPointX = keylines[j].endPointX;
                    keylines[i].endPointY = keylines[j].endPointY;
                }
                else
                {
                    keylines[i].startPointX = keylines[j].startPointX;
                    keylines[i].startPointY = keylines[j].startPointY;
                }
                keylines[j].class_id = i;
                fused[j] = true;
                //cout<<"angle: "<<keylines[j].angle<<" "<<keylines[i].angle<<endl;
                fuseCounter++;
            }
        }
    }

    cout<<" - fused line number by filter: "<<fuseCounter<<endl;

    const int HISTO_LENGTH = 100;
    const double factor = (double)HISTO_LENGTH/(2.0f*M_PI);
    vector<int> rotHist[HISTO_LENGTH];
    vector<line_descriptor::KeyLine> keylinesFilter;
    for(int i=0; i< fused.size(); i++)
    {
        if(!fused[i])
        {
            double rot = keylines[i].angle;
            if(rot<0)
                rot += 2*M_PI;
            int bin = round(rot * factor);// 将rot分配到bin组
            if(bin==HISTO_LENGTH)
                bin=0;
            assert(bin>=0 && bin<HISTO_LENGTH);
            int id = keylinesFilter.size();
            rotHist[bin].push_back(id); //将ID号放进去
            keylines[i].class_id = id;
            keylinesFilter.push_back(keylines[i]);
        }
    }
    cout<<" - keylinesFilter size: "<<keylinesFilter.size()<<endl;

    // save data to plot histogram
    std::ofstream f;
    f.open("./out_for_histogram.txt");
    f << std::fixed;
    for(int i=0; i<keylinesFilter.size(); i++)
    {
        double rot = keylinesFilter[i].angle;
        if(rot<0)
            rot += 2*M_PI;
        f << std::setprecision(6)<<(rot*180.0f/M_PI)<<" "; //
    }
    f.close();

    // 滑动窗口统计
    vector<line_descriptor::KeyLine> keylinesFilterAndHist;
    cout<<">> histogram filter..."<<endl;
#if 1
    double countAngle = 15.0f;
    int slidingWin = round(countAngle*HISTO_LENGTH/360.0f);
    cout<<" - slidingWin: "<<slidingWin<<endl;
    int ind1 = -1;
    ComputeMaxima(rotHist, HISTO_LENGTH, slidingWin, ind1);
    int resAngle = ind1-(slidingWin-1)/2;
    if(resAngle < 0)
        resAngle += HISTO_LENGTH;
    cout<<" - ind1: "<<ind1<<endl;
    cout<<" - resAngle: "<<360.0f*resAngle/HISTO_LENGTH<<endl;
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

#else
    int ind1=-1, ind2=-1, ind3=-1, ind4=-1;
    ComputeFourMaxima(rotHist,HISTO_LENGTH, ind1,ind2,ind3,ind4);
    cout<<" - rotation histogram: "<<360.0f*ind1/HISTO_LENGTH<<" "<<360.0f*ind2/HISTO_LENGTH<<" "<<
          360.0f*ind3/HISTO_LENGTH<<" "<<360.0f*ind4/HISTO_LENGTH<<endl;
    cout<<" - rotation histogram: "<<rotHist[ind1].size()<<" "<<rotHist[ind2].size()<<" "<<
          rotHist[ind3].size()<<" "<<rotHist[ind4].size()<<endl;
    duration_ms = (double(getTickCount()) - start) * 1000 / getTickFrequency();
    std::cout << "   It took " << duration_ms << " ms" << std::endl;

    for(int i=0; i<HISTO_LENGTH; i++)
    {
        if(i == ind1 || i == ind2 || i == ind3 || i == ind4)
        {
            for(int j=0; j<rotHist[i].size(); j++)
            {
                keylinesFilterAndHist.push_back(keylinesFilter[rotHist[i][j]]);
            }
        }
    }
#endif
    cout<<" - keylinesFilterAndHist size: "<<keylinesFilterAndHist.size()<<endl;



    //Mat imageDrawedLineAfterFilter = imageUndistort;
    Mat imageDrawedLineAfterFilter = Mat::zeros(imageUndistort.rows, imageUndistort.cols, CV_8UC3);
    if(imageDrawedLineAfterFilter.channels() == 1)
        cv::cvtColor(imageDrawedLineAfterFilter, imageDrawedLineAfterFilter, cv::COLOR_GRAY2BGR);
    for(line_descriptor::KeyLine keyline:keylinesFilter)
    {
        cv::line(imageDrawedLineAfterFilter, keyline.getStartPoint(), keyline.getEndPoint(), cv::Scalar(0,0,255), 1, CV_AA);
    }
    cv::namedWindow("imageDrawedLineAfterFilter", CV_WINDOW_NORMAL);
    imshow( "imageDrawedLineAfterFilter", imageDrawedLineAfterFilter );


    Mat imageDrawedLineAfterFilterAndHist = Mat::zeros(imageUndistort.rows, imageUndistort.cols, CV_8UC3);
    if(imageDrawedLineAfterFilterAndHist.channels() == 1)
        cv::cvtColor(imageDrawedLineAfterFilterAndHist, imageDrawedLineAfterFilterAndHist, cv::COLOR_GRAY2BGR);
    for(line_descriptor::KeyLine keyline:keylinesFilterAndHist)
    {
        cv::line(imageDrawedLineAfterFilterAndHist, keyline.getStartPoint(), keyline.getEndPoint(), cv::Scalar(0,255,0), 1, CV_AA);
    }
    cv::namedWindow("imageDrawedLineAfterFilterAndHist", CV_WINDOW_NORMAL);
    imshow( "imageDrawedLineAfterFilterAndHist", imageDrawedLineAfterFilterAndHist );

    waitKey();

    cv::imwrite("image_raw.jpg", imageUndistortForSave);
    cv::imwrite("image_canny.jpg", imageUndistort);
    cv::imwrite("imageDrawedLineBeforeFilter.jpg", imageDrawedLineBeforeFilter);
    cv::imwrite("imageDrawedLineAfterFilter.jpg", imageDrawedLineAfterFilter);
    cv::imwrite("imageDrawedLineAfterFilterAndHist.jpg", imageDrawedLineAfterFilterAndHist);

    return 0;
}
