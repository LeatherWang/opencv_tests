#include <stdio.h>
#include <stdlib.h>

#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>

#include <opencv2/line_descriptor.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/flann.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/ximgproc.hpp>

#include "ORBextractor.h"
#include "RIBrief.h"

#define RESET    "\033[0m"
#define BRED     "\033[1;31m"
#define BGREEN   "\033[1;32m"
#define BYELLOW  "\033[1;33m"
#define BWHITE   "\033[1;37m"

#define INFO(...)  (printf(BGREEN __VA_ARGS__), printf(RESET), printf("\n"))

using namespace cv;
using namespace std;
using namespace cv::ximgproc;
using namespace cv::xfeatures2d;
using namespace cv::line_descriptor;

vector<string> gImgPath;
bool LoadImages(const string &imgFolder, const string &strPathTimes, vector<string>& vstrImg) 
{
    ifstream fTimes;
    fTimes.open(strPathTimes.c_str());
    vstrImg.reserve(5000);
   
    while (!fTimes.eof()) 
    {
        string s;
        getline(fTimes, s);
        if (!s.empty())
        {
            stringstream ss;
            ss << s;
            //vstrImg.push_back(imgFolder + "/" + ss.str() + ".png");
            vstrImg.push_back(imgFolder + "/" + ss.str());
            gImgPath.push_back(imgFolder + "/" + ss.str() + ".jpg");
        }
    }
    fTimes.close();

    return true;
}

struct sort_lines_by_response
{
    inline bool operator()(const KeyLine& a, const KeyLine& b){
        return ( a.response > b.response );
    }
};

bool ascendSort(vector<Point> a, vector<Point> b) 
{
	return a.size() < b.size();
 
}
 
//轮廓按照面积大小降序排序
bool descendSort(vector<Point> a, vector<Point> b) 
{
	return a.size() > b.size();
}

int OtsuAlgThreshold(const Mat image)
{
    int threValue = 15;          //Otsu算法阈值
    double varValue = 0;         //类间方差中间值保存
    double w0 = 0;               //前景像素点数所占比例
    double w1 = 0;               //背景像素点数所占比例
    double u0 = 0;               //前景平均灰度
    double u1 = 0;               //背景平均灰度
    double Histogram[256] = {0};
    double totalNum = image.rows * image.cols; 
    for (size_t y = 0; y < image.rows; ++y)
    {
        const uchar* pData = image.ptr<uchar>(y);
        for (size_t x = 0; x < image.cols; ++x)
        {
            Histogram[pData[x]]++;
        }
    }

    for (size_t i = 0; i < 255; ++i)
    {
        w0 = w1 = 0;
        u0 = u1 = 0;
        for (size_t j = 0; j <= i; ++j)
        {
            w1 += Histogram[j];
            u1 += j * Histogram[j];
        }

        if (w1 == 0)
        {
            continue;
        }
        u1 = u1 / w1;
        w1 = w1 / totalNum;
        for (size_t k = i + 1; k < 255; ++k)
        {
            w0 += Histogram[k];
            u0 += k * Histogram[k];
        }
        
        if (w0 == 0) 
        {
            break;
        }

        u0 = u0 / w0;
        w0 = w0 / totalNum;
        double varValueI = w0 * w1 * (u1 - u0) * (u1 - u0);
        if (varValue < varValueI)
        {
            varValue = varValueI;
            threValue = i;
        }
    }
    return threValue;
}

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

int main()
{
    vector<string> vstrImg;
    //std::string imgFolder = "/home/henry/Dataset/MH_01_easy/mav0/cam0/data";
    //LoadImages(imgFolder, "./data/MH01.txt", vstrImg);
    //std::string imgFolder = "/home/henry/Dataset/akshanghai-office/001_dataSample";
    //LoadImages(imgFolder, "./data/rgb.txt", vstrImg);
    // std::string imgFolder = "/home/henry/Dataset/11F-D";
    // LoadImages(imgFolder, "./data/11d.txt", vstrImg);
    LoadImages("/home/henry/workspace/a35", "./data/a38.txt", vstrImg);
    //string imgList[] = { "./img/img45.jpg", "./img/img46.jpg", "./img/img47.jpg", "./img/img48.jpg", "./img/img49.jpg", "./img/img50.jpg"};
    string imgList[] = { "./img/church1.jpg", "./img/church2.jpg", "./img/01.png", 
        "./img/02.png", "./img/church13.jpg", "./img/church23.jpg", 
        "./img/church111.jpg", "./img/church222.jpg"};
    ORBextractor exFeatures(500, 1.2f, 3, 15, 20);
    ORBextractor exFeaturesRight(500, 1.2f, 3, 15, 20);

    ORBextractor exFeaturesORB(500, 1.2f, 3, 15, 20);
    ORBextractor exFeaturesRightORB(500, 1.2f, 3, 15, 20);

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    Ptr<xfeatures2d::BriefDescriptorExtractor> brief = xfeatures2d::BriefDescriptorExtractor::create();
    
    // /* create a random binary mask */
    // cv::Mat mask = Mat::ones( imgLeft.size(), CV_8UC1 );

    /* create a pointer to a BinaryDescriptor object with deafult parameters */
    BinaryDescriptor::Params tmParams;
    tmParams.numOfOctave_ = 1;
    tmParams.widthOfBand_ = 5;
    tmParams.reductionRatio = 2;
    tmParams.ksize_ = 3;


    Ptr<BinaryDescriptor> bd = BinaryDescriptor::createBinaryDescriptor(tmParams);
    Ptr<LSDDetector> lsd = LSDDetector::createLSDDetector();
    
    for(size_t idx = 0; idx < vstrImg.size(); idx += 1)
    {
        cout << "file path = " << vstrImg[idx] << endl;
        cv::Mat imgLeft = cv::imread(vstrImg[idx], 1);
        cv::Mat imgRight = cv::imread(vstrImg[idx], 1);

        cv::Mat greyLeft, greyRight;
        cv::cvtColor(imgLeft, greyLeft, cv::COLOR_BGR2GRAY);
        cv::cvtColor(imgRight, greyRight, cv::COLOR_BGR2GRAY);

        int avg = getMean(greyLeft);
        int lowerThresh = (int)avg * 0.5;
        int upperThresh = (int)avg * 1.25;


        cv::Mat cannyImg, blurImage, scaleImg;
        int tmpScale = 1;
        cv::resize(greyLeft, scaleImg, cv::Size(greyLeft.cols / tmpScale, greyLeft.rows / tmpScale));

        cv::Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
        cv::Mat imgErode;
        erode(scaleImg, imgErode, element);
        Mat internalGradientImg;
	    subtract(scaleImg, imgErode, internalGradientImg, Mat());
        cv::imshow("Gradient", internalGradientImg);

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        
        blur(internalGradientImg, blurImage, Size(3,3));
        int threVal = OtsuAlgThreshold(blurImage);
        cv::Canny(blurImage, cannyImg, lowerThresh, upperThresh, 5, true);
        vector<KeyLine> lines;
        bd->detect(cannyImg, lines);
        //bd->detect(internalGradientImg, lines);
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        double timeCost1 = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
        INFO("bd cost time: %lf", timeCost1);

        if(lines.size() > 0)
        {
            sort(lines.begin(), lines.end(), sort_lines_by_response());

            // KeyLine kl = lines[0];
            // Point pt1 = Point2f( kl.startPointX * tmpScale, kl.startPointY * tmpScale);
            // Point pt2 = Point2f( kl.endPointX * tmpScale, kl.endPointY * tmpScale);
            // line( imgLeft, pt1, pt2, Scalar( 0, 255, 0 ), 3 );

            size_t tmpSize = lines.size() > 30 ? 30 : lines.size();
            lines.resize(tmpSize);
            for(size_t i=0; i< tmpSize; i++)
                lines[i].class_id = i;

            for ( size_t i = 0; i < lines.size(); i++ )
            {
                KeyLine kl = lines[i];
                if( kl.octave == 0)
                {
                    /* get a random color */
                    int R = ( rand() % (int) ( 255 + 1 ) );
                    int G = ( rand() % (int) ( 255 + 1 ) );
                    int B = ( rand() % (int) ( 255 + 1 ) );

                    /* get extremes of line */
                    Point pt1 = Point2f( kl.startPointX * tmpScale, kl.startPointY * tmpScale);
                    Point pt2 = Point2f( kl.endPointX * tmpScale, kl.endPointY * tmpScale);

                    /* draw line */
                    line( imgLeft, pt1, pt2, Scalar( B, G, R ), 2 );
                }
            }
            char angle[32];
            sprintf(angle, "angle:%.3f", lines[0].angle);
            cv::putText(imgLeft, angle, Point(imgLeft.cols - 250, 50), 2, 1, Scalar(0, 255, 255), 1);
        }

        // if(lineVect.size() > 0)
        // {
        //     sort(lineVect.begin(), lineVect.end(), sort_lines_by_response());
        //     KeyLine kl = lineVect[0];
        //     Point pt1 = Point2f( kl.startPointX, kl.startPointY);
        //     Point pt2 = Point2f( kl.endPointX, kl.endPointY);
        //     line(imgLeft, pt1, pt2, Scalar( 255, 0, 0 ), 3 );
        // }

        cv::imshow("line", imgLeft);
        cv::imshow("canny", cannyImg);
        cv::imwrite(gImgPath[idx], imgLeft);

        // std::vector<cv::KeyPoint> orbKeyPointsLeft;
        // std::vector<cv::KeyPoint> orbKeyPointsRight;
        // {
        //     cv::Mat orbDescLeft, orbDescRight;
            
        //     exFeaturesORB(greyLeft, cv::noArray(), orbKeyPointsLeft, orbDescLeft);
        //     exFeaturesRightORB(greyRight, cv::noArray(), orbKeyPointsRight, orbDescRight);
            
        //     std::vector<DMatch> orbMatchesAll, orbMatchesGMS;
        //     matcher->match(orbDescRight, orbDescLeft, orbMatchesAll);

        //     matchGMS(greyRight.size(), greyLeft.size(), orbKeyPointsRight, orbKeyPointsLeft, orbMatchesAll, orbMatchesGMS, true, true);
        //     std::cout << "orb matchesGMS: " << orbMatchesGMS.size() << std::endl;
        //     if(orbMatchesGMS.size() < 5)
        //         cv::waitKey(0);

        //     Mat orbMatches;
        //     {    
        //         drawMatches(greyRight, orbKeyPointsRight, greyLeft, orbKeyPointsLeft, orbMatchesGMS, orbMatches
        //             , Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
                
        //         char fCount[32];
        //         sprintf(fCount, "num:%d", orbMatchesGMS.size());
        //         putText(orbMatches, fCount, Point(orbMatches.cols - 200, 50), 2, 1.0, Scalar(0, 255, 0), 2);
        //     }

        //     imshow("Matches GMS ORB", orbMatches);
        // }

        // {
        //     std::vector<cv::KeyPoint> keypoints;
        //     std::vector<cv::KeyPoint> keypointsRight;
        //     cv::Mat desc, descRight;

        //     std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        //     //exFeatures(greyLeft, cv::noArray(), keypoints, desc);
        //     //exFeaturesRight(greyRight, cv::noArray(), keypointsRight, descRight);

        //     brief->compute(greyLeft, orbKeyPointsLeft, desc);
        //     brief->compute(greyRight, orbKeyPointsRight, descRight);
        //     //RIBrief::ComputerDesc(greyLeft, orbKeyPointsLeft, desc);
        //     //RIBrief::ComputerDesc(greyRight, orbKeyPointsRight, descRight);

        //     std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        //     double timeCost = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
        //     INFO("lsd cost time: %lf", timeCost);

        //     std::vector<DMatch> matchesAll, matchesGMS;
        //     matcher->match(descRight, desc, matchesAll);

        //     matchGMS(greyRight.size(), greyLeft.size(), orbKeyPointsRight, orbKeyPointsLeft, matchesAll, matchesGMS, true, true);
        //     std::cout << "matchesGMS: " << matchesGMS.size() << std::endl;
        //     if(matchesGMS.size() < 5)
        //         cv::waitKey(0);

        //     Mat riMatches;
        //     {    
        //         drawMatches(greyRight, orbKeyPointsRight, greyLeft, orbKeyPointsLeft, matchesGMS, riMatches
        //             , Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

        //         char fCount[32];
        //         sprintf(fCount, "num:%d", matchesGMS.size());
        //         putText(riMatches, fCount, Point(riMatches.cols - 200, 50), 2, 1.0, Scalar(0, 255, 0), 2);
        //     }

        //     imshow("Matches GMS RI", riMatches);
        // }
        
        waitKey(0);
    }

    return 0;
}