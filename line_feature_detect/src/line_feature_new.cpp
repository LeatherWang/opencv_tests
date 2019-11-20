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
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

// 将opencv contrib 中的line_descriptor模块拿出来自己编译
#include "line_descriptor/descriptor.hpp"

using namespace std;
using namespace cv;
double start, duration_ms;
double totalTime=0.0f;
const int HISTO_LENGTH = 100;
vector<line_descriptor::KeyLine> lastKeylines;
cv::Mat lastImageUndistort;
cv::Mat mLdesc2;
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

void ComputeMaxima(const vector<int>* rotHist, const int slidingWin,
                   int &ind1)
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

        if(num > max1) {
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
string window_name = "Edge Map";

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

bool readIntrinsic_2(const string &file_path, cv::Mat &K_Mat, cv::Mat &DistCoef)
{
    cv::FileStorage fs(file_path, cv::FileStorage::READ);
    if (!fs.isOpened())
        return false;

    DistCoef.at<double>(0) = fs["Camera.k1"];
    DistCoef.at<double>(1) = fs["Camera.k2"];
    DistCoef.at<double>(2) = fs["Camera.p1"];
    DistCoef.at<double>(3) = fs["Camera.p2"];

    K_Mat.at<double>(0,0) = fs["Camera.fx"];
    K_Mat.at<double>(1,1) = fs["Camera.fy"];
    K_Mat.at<double>(0,2) = fs["Camera.cx"];
    K_Mat.at<double>(1,2) = fs["Camera.cy"];

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

double computeMeanOutAngle(const vector<int> *rotHist, const vector<line_descriptor::KeyLine> &keylinesFilter,
                           vector<line_descriptor::KeyLine> &keylinesFilterAndHist,
                           const double &countAngleRad, const int &slidingWin, const int &ind1)
{
    double numAngel = 0.0f;
    double weightNum = 0.0f;
    bool bZeroRight = false, bZeroLeft=false;
    for(int i=0; i<slidingWin; i++)
    {
        int index = ind1-i;
        if(index<0)
            index += HISTO_LENGTH;
        for(uint j=0; j<rotHist[index].size(); j++)
        {
            keylinesFilterAndHist.push_back(keylinesFilter[rotHist[index][j]]);
            double rot = keylinesFilterAndHist.back().angle;
            if(rot < 0)
                rot += M_PI;
            numAngel += rot * keylinesFilterAndHist.back().response;
            weightNum += keylinesFilterAndHist.back().response;
            if(rot < countAngleRad)
                bZeroRight = true;
            if(rot > M_PI - countAngleRad)
                bZeroLeft = true;
        }
    }

    // 角度互补时，导致输出不连贯，重新进行计算
    if(bZeroRight && bZeroLeft)
    {
        numAngel = 0.0f;
        weightNum = 0.0f;
        for(line_descriptor::KeyLine ele:keylinesFilterAndHist)
        {
            double rot = ele.angle;
            if(rot < 0)
                rot += M_PI;
            if(rot > M_PI - countAngleRad)
                rot -= M_PI;
            numAngel += rot * ele.response;
            weightNum += ele.response;
        }
    }
    double curAngle = 180.0f*numAngel/weightNum/M_PI;
    return curAngle;
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
    "{v     | true                | verbose or not}"
    "{help h|                     | show help message}"
};

// ./line_feature_new  '' --ds=-1 --es=-2 -r=true -c=false -s=false -l=false -v=true
int main(int argc, char** argv)
{
    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    string inputDir = parser.get<string>("@input");

    string file_path = inputDir+"/camera.yaml";//"../camera_camera_calib_ak.yaml"; // camera_fisheye_ankobot
    cout<<"intrinsic file: "<<file_path<<endl;
    cv::Mat K_Mat = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat DistCoef = cv::Mat(4, 1, CV_64F);
    //cv::Mat DistCoef_Zero = cv::Mat::zeros(4, 1, CV_64F);
    if(!readIntrinsic_2(file_path, K_Mat, DistCoef))
        return 1;


    bool bRoi = parser.get<bool>("r");
    bool bUsingCanny = parser.get<bool>("c");
    bool bManualAdjustCanny = parser.get<bool>("ca");
    bool bUsingRobert = parser.get<bool>("s");
    bool bUsingMorphology = parser.get<bool>("d");
    bool bUsingLSD = parser.get<bool>("l");
    bool verbose = parser.get<bool>("v");
    int eleSizeForDilate = parser.get<int>("ds");
    int eleSizeForErode = parser.get<int>("es");

    string imageFilePath = inputDir+"/cam0";
    if (!boost::filesystem::exists(imageFilePath) && !boost::filesystem::is_directory(imageFilePath))
    {
        std::cerr << "# ERROR: Cannot find input directory " << imageFilePath << "." << std::endl;
        return 1;
    }

    std::string prefix = "Main";
    std::string fileExtension = ".jpg";
    std::vector<std::string> imageFilenames;
    for (boost::filesystem::directory_iterator itr(imageFilePath); itr != boost::filesystem::directory_iterator(); ++itr)
    {
        if (!boost::filesystem::is_regular_file(itr->status()))
            continue;

        std::string filename = itr->path().filename().string();
        // check if prefix matches
        //        if (!prefix.empty())
        //            if (filename.compare(0, prefix.length(), prefix) != 0)
        //                continue;
        // check if file extension matches
        if (filename.compare(filename.length() - fileExtension.length(), fileExtension.length(), fileExtension) != 0)
            continue;
        imageFilenames.push_back(itr->path().string());
    }
    if (imageFilenames.empty())
    {
        std::cerr << "# ERROR: No chessboard images found." << std::endl;
        return 1;
    }

    auto cmp = [](const std::string &a, const std::string &b){
        if(a.size() < b.size())
            return true;
        else if(a.size() == b.size())
            return a<b;
        return false;
    };
    sort(imageFilenames.begin(), imageFilenames.end(), cmp);

    Mat image = imread(imageFilenames[0], -1);
    cv::Mat mapx, mapy; //内部存的是重映射之后的坐标，不是像素值
    cv::Mat R = Mat::eye(3,3, CV_64F);
    cv::fisheye::initUndistortRectifyMap(K_Mat, DistCoef, R, K_Mat,
                                         image.size(), CV_32FC1, mapx, mapy);


    for(uint i=0; i<imageFilenames.size(); i++)
    {
        if(verbose)
            cout<<"filename: "<<imageFilenames[i]<<endl;
        Mat image_raw = imread(imageFilenames[i], -1);

        double start = double(getTickCount());
        double startTotal = double(getTickCount());
        cv::Mat imageUndistort;
#if 1
        //cv::undistort(image_raw, imageUndistort, K_Mat, DistCoef); //! @attention 先去畸变
        if(verbose)
            cout<<">> undistort..."<<endl;
        cv::remap(image_raw, imageUndistort, mapx, mapy, cv::INTER_LINEAR); //双线性插值
        duration_ms = (double(getTickCount()) - start) * 1000 / getTickFrequency();
        if(verbose)
            std::cout << "   It took " << duration_ms << " ms" << std::endl;
#else
        imageUndistort = image_raw;
#endif

        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(imageUndistort, imageUndistort);

        if(bRoi) {
            cv::Rect roiRect(0,30, 640, 300);
            //imageUndistort = imageUndistort(roiRect); //错误的用法，下面`imageUndistort.ptr<uchar>(y)`将出问题
            Mat roiImage = imageUndistort(roiRect);
            roiImage.copyTo(imageUndistort);
        }
        Mat imageUndistortForSave;
        imageUndistort.copyTo(imageUndistortForSave);
        if(verbose)
            cv::imshow("imageUndistort", imageUndistort);
        //        waitKey();
        if(imageUndistort.channels() == 3)
            cv::cvtColor(imageUndistort, imageUndistort, cv::COLOR_RGB2GRAY);

        if(bUsingCanny)
        {
            if(!bManualAdjustCanny)
            {
                if(verbose)
                    cout<<">> using canny detect edge..."<<endl;
                start = double(getTickCount());
                // cv::Mat blurImg;
                // cv::blur(imageUndistort, blurImg, Size(3,3)); //!@todo
                // imshow("blur", blurImg);
                cv::Mat gaussMImg;
                cv::GaussianBlur( imageUndistort, gaussMImg, Size(3,3), 0, 0, BORDER_DEFAULT );
                // imshow("GaussianBlur", gaussMImg);

                // cv::Mat bilImg;
                // cv::bilateralFilter(imageUndistort, bilImg,5, 30, 30);
                // imshow("bilateralFilter", bilImg);
                // waitKey();

                // int avg = getMean(imageUndistort);
                // int lowerThresh = (int)avg * 0.5;
                // int upperThresh = (int)avg * 1.25;

                //imageUndistort = gaussMImg;
                if(verbose)
                {
                    duration_ms = (double(getTickCount()) - start) * 1000 / getTickFrequency();
                    std::cout << "   It took " << duration_ms << " ms" << std::endl;
                }

                const int cannyTh = 22;
                if(verbose)
                    std::cout << " - cannyTh: " << cannyTh << std::endl;
                start = double(getTickCount());
                Canny(imageUndistort, imageUndistort, cannyTh, cannyTh*3, 3); // Apply canny edge
                //Canny(imageUndistort, imageUndistort, lowerThresh, upperThresh, 4);

                if(verbose)
                {
                    duration_ms = (double(getTickCount()) - start) * 1000 / getTickFrequency();
                    std::cout << "   It took " << duration_ms << " ms" << std::endl;
                    cv::imshow("canny", imageUndistort);
                }
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
        else if(bUsingRobert) {
            if(verbose)
                cout<<">> using robert detect edge..."<<endl;
            start = double(getTickCount());
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
            if(verbose)
            {
                duration_ms = (double(getTickCount()) - start) * 1000 / getTickFrequency();
                std::cout << "   It took " << duration_ms << " ms" << std::endl;
                cv::imshow("robert_raw_th", imageUndistort);
            }
        }

        /*【形态学】*/
        if(bUsingMorphology)
        {
            if(eleSizeForDilate > 0)
            {
                Mat element = getStructuringElement(MORPH_RECT,
                                                    Size(2*eleSizeForDilate+1, 2*eleSizeForDilate+1),
                                                    Point( eleSizeForDilate, eleSizeForDilate ));
                cv::dilate(imageUndistort, imageUndistort, element);//进行膨胀操作
            }

            if(eleSizeForErode > 0)
            {
                Mat elementErode = getStructuringElement(MORPH_RECT,
                                                         Size(2*eleSizeForErode+1, 2*eleSizeForErode+1),
                                                         Point( eleSizeForErode, eleSizeForErode ));
                cv::erode(imageUndistort, imageUndistort, elementErode);
            }

            if(verbose)
                if(eleSizeForDilate>0 || eleSizeForErode > 0)
                    imshow("erode", imageUndistort);
        }

        /*【检测线特征】*/
        vector<line_descriptor::KeyLine> keylines;
        if(bUsingLSD) {
            if(verbose) {
                cout<<">> LSD detect lines..."<<endl;
                start = double(getTickCount());
            }
            Ptr<line_descriptor::LSDDetector> lsd = line_descriptor::LSDDetector::createLSDDetector();
            lsd->detect(imageUndistort, keylines, 1.2, 1);
            if(verbose) {
                duration_ms = (double(getTickCount()) - start) * 1000 / getTickFrequency();
                std::cout << "   It took " << duration_ms << " ms" << std::endl;
            }
        }
        else {
            if(verbose) {
                cout<<">> ED detect lines..."<<endl;
                start = double(getTickCount());
            }
            line_descriptor::BinaryDescriptor::Params tmParams;
            tmParams.numOfOctave_ = 1;
            tmParams.widthOfBand_ = 6;
            tmParams.reductionRatio = 2;
            tmParams.ksize_ = 3;
            Ptr<line_descriptor::BinaryDescriptor> bd = line_descriptor::BinaryDescriptor::createBinaryDescriptor(tmParams);
            bd->detect(imageUndistort, keylines);
            if(verbose) {
                duration_ms = (double(getTickCount()) - start) * 1000 / getTickFrequency();
                std::cout << "   It took " << duration_ms << " ms" << std::endl;
            }
        }

        int lsdNFeatures = 0;
        {
            if(verbose)
                cout<<">> resort line features by their response..."<<endl;

            sort(keylines.begin(), keylines.end(), [](const line_descriptor::KeyLine& kl1,
                 const line_descriptor::KeyLine &kl2){
                return kl1.response > kl2.response;
            });

            for(uint i=0; i< keylines.size(); i++)
            {
                if(keylines[i].response > 0.05)
                    lsdNFeatures++;
            }

            if(verbose)
                cout<<" - lsdNFeatures: "<<lsdNFeatures<<endl;

            keylines.resize(lsdNFeatures);
            for( int i=0; i<lsdNFeatures; i++)
                keylines[i].class_id = i;
        }

        /*【LBD描述子】*/
        if(verbose) {
            cout<<">> Compute descriptor..."<<endl;
            start = double(getTickCount());
        }
        Mat mLdesc;
        Ptr<line_descriptor::BinaryDescriptor> lbd = line_descriptor::BinaryDescriptor::createBinaryDescriptor();
        lbd->compute(imageUndistort, keylines, mLdesc);
        if(verbose) {
            duration_ms = (double(getTickCount()) - start) * 1000 / getTickFrequency();
            std::cout << "   It took " << duration_ms << " ms" << std::endl;
        }
        if(!lastImageUndistort.empty())
        {
            vector<vector<DMatch>> lmatches;
            BFMatcher* bfm = new BFMatcher(NORM_HAMMING, false);
            bfm->knnMatch(mLdesc, mLdesc2, lmatches, 2);
            vector<DMatch> matches;
            for(size_t i=0;i<lmatches.size();i++)
            {
                const DMatch& bestMatch = lmatches[i][0];
                const DMatch& betterMatch = lmatches[i][1];
                float  distanceRatio = bestMatch.distance / betterMatch.distance;
                if (distanceRatio < 0.75)
                    matches.push_back(bestMatch);
            }
            cout<<"matches size: "<<matches.size()<<endl;
            cv::Mat outImg;
            std::vector<char> mask( lmatches.size(), 1 );
            cv::Mat image1, image2;
            if(imageUndistort.channels() == 1)
                cvtColor(imageUndistort, image1, COLOR_GRAY2BGR);
            if(lastImageUndistort.channels() == 1)
                cvtColor(lastImageUndistort, image2, COLOR_GRAY2BGR);
            cv::line_descriptor::drawLineMatches( image1, keylines,
                                                  image2, lastKeylines, matches, outImg,
                                                  Scalar::all( -1 ), Scalar::all( -1 ), mask,
                                                  cv::line_descriptor::DrawLinesMatchesFlags::DEFAULT );
            //imshow( "Matches", outImg );
            //waitKey();
        }

        imageUndistort.copyTo(lastImageUndistort);
        mLdesc.copyTo(mLdesc2);
        lastKeylines = keylines;


        if(verbose)
        {
            Mat imageDrawedLineBeforeFilter = Mat::zeros(imageUndistort.rows, imageUndistort.cols, CV_8UC3);
            for(line_descriptor::KeyLine keyline:keylines)
                cv::line(imageDrawedLineBeforeFilter, keyline.getStartPoint(), keyline.getEndPoint(), cv::Scalar(255), 1, CV_AA);
            cv::namedWindow("imageDrawedLineBeforeFilter", CV_WINDOW_AUTOSIZE);
            imshow( "imageDrawedLineBeforeFilter", imageDrawedLineBeforeFilter );
        }


        // 构造直方图
        const double factor = (double)HISTO_LENGTH/(M_PI);
        vector<int> rotHist[HISTO_LENGTH];
        vector<line_descriptor::KeyLine> keylinesFilter;
        for(uint i=0; i< keylines.size(); i++)
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
        if(verbose)
            cout<<" - keylinesFilter size: "<<keylinesFilter.size()<<endl;

        // save data to plot histogram
        if(verbose)
        {
            std::ofstream f;
            f.open("./out_for_histogram.txt");
            f << std::fixed;
            for(uint i=0; i<keylinesFilter.size(); i++)
            {
                double rot = keylinesFilter[i].angle;
                // (-pi,pi] => (0,pi]
                if(rot <= 0)
                    rot += M_PI;
                f << std::setprecision(6)<<(rot*180.0f/M_PI)<<" ";
            }
            f.close();
        }

        // 滑动窗口统计
        vector<line_descriptor::KeyLine> keylinesFilterAndHist;

        if(verbose)
            cout<<">> histogram filter..."<<endl;

        double countAngle = 8.0f;
        double countAngleRad = countAngle*M_PI/180.0f;
        int slidingWin = round(countAngle*HISTO_LENGTH/180.0f);
        if(verbose)
            cout<<" - slidingWin: "<<slidingWin<<endl;
        int ind1 = -1, ind2 = -1;
        // 统计主方向
        ComputeMaxima(rotHist, slidingWin, ind1);


        int resAngle = ind1-(slidingWin-1)/2;
        if(resAngle < 0)
            resAngle += HISTO_LENGTH;

        // 根据主方向范围内所有角度，计算一个均值作为当前的主方向的角度
        double curAngle = computeMeanOutAngle(rotHist, keylinesFilter, keylinesFilterAndHist,
                                              countAngleRad, slidingWin,  ind1);

        if(verbose)
            cout<<" - ind1: "<<ind1<<"  size: "<<keylinesFilterAndHist.size()<<
                  "  resAngle: "<<180.0f*resAngle/HISTO_LENGTH<<endl;

        // 出现两个主方向，相差90度时
        static double lastAngle = curAngle;
        static int healthCounter = 0;
        double error = fabs(curAngle-lastAngle);
        if(85.0f<error && error<95.0f)
        {
            // 连续多次出现此种情况，就判别为异常
            healthCounter++;
            if(healthCounter > 10) {
                healthCounter = 0;
                cout<<"_______healthCouter > 10_________"<<endl;
            }
            else {
                for(int i=0; i<slidingWin; i++)
                {
                    int index = ind1-i;
                    if(index<0)
                        index += HISTO_LENGTH;
                    rotHist[index].clear();
                }
                ComputeMaxima(rotHist, slidingWin, ind2);
                resAngle = ind2-(slidingWin-1)/2;
                if(resAngle < 0)
                    resAngle += HISTO_LENGTH;

                int sizeInd1 = keylinesFilterAndHist.size();
                keylinesFilterAndHist.clear();
                double tmpAngle= computeMeanOutAngle(rotHist, keylinesFilter, keylinesFilterAndHist,
                                                     countAngleRad, slidingWin, ind2);
                if(keylinesFilterAndHist.size() > sizeInd1*0.6) {
                    curAngle = tmpAngle;
                    cout<<" - ind2: "<<ind2<<"  size: "<<keylinesFilterAndHist.size()<<
                          "  resAngle: "<<180.0f*resAngle/HISTO_LENGTH<<endl;
                    error = fabs(curAngle-lastAngle);
                }
                else {
                    cout<<"|||||||||||=> this thing should not appear, check!"<<endl;
                    waitKey();
                }
            }
        }
        else
        {
            if(healthCounter > 0)
                healthCounter--;
        }
        cout<<" - curAngle: "<<curAngle<<"  lastAngle: "<<lastAngle<<"  error: "<<error<<endl;
        //        if(error > 1.0f && error<5.0f)
        //            waitKey();
        lastAngle = curAngle;


        // 映射到-90到90度进行输出
        double finalAngle;
        // (0,pi] => (-pi/4, pi/4]
        if(curAngle < 45.0f)
            finalAngle = curAngle;
        else if(curAngle < 135.0f)
            finalAngle = curAngle - 90.0f;
        else
            finalAngle = curAngle - 180.0f;

        cout<<" - finalAngle: "<<finalAngle<<endl;

        //        if(verbose)
        //            if(ind2 != -1)
        //                waitKey();


        if(verbose)
        {
            Mat imageDrawedLineAfterFilterAndHist = Mat::zeros(imageUndistort.rows, imageUndistort.cols, CV_8UC3);
            if(imageDrawedLineAfterFilterAndHist.channels() == 1)
                cv::cvtColor(imageDrawedLineAfterFilterAndHist, imageDrawedLineAfterFilterAndHist, cv::COLOR_GRAY2BGR);
            for(line_descriptor::KeyLine keyline:keylinesFilterAndHist)
            {
                cv::line(imageDrawedLineAfterFilterAndHist, keyline.getStartPoint(), keyline.getEndPoint(), cv::Scalar(0,255,0), 1, CV_AA);
            }
            cv::namedWindow("imageDrawedLineAfterFilterAndHist", CV_WINDOW_AUTOSIZE);
            imshow( "imageDrawedLineAfterFilterAndHist", imageDrawedLineAfterFilterAndHist );
        }

        duration_ms = (double(getTickCount()) - startTotal) * 1000 / getTickFrequency();
        std::cout << "Total time token " << duration_ms << " ms" << std::endl;
        totalTime += duration_ms;
        std::cout << "mean time spent: " << totalTime/(i+1) << "ms" << endl;
        waitKey( );
        cout<<"----------------------------------------------"<<endl<<endl;
    }



    //    cv::imwrite("image_raw.bmp", imageUndistortForSave);
    //    cv::imwrite("image_canny.bmp", imageUndistort);
    //    cv::imwrite("imageDrawedLineBeforeFilter.bmp", imageDrawedLineBeforeFilter);
    //    cv::imwrite("imageDrawedLineAfterFilter.bmp", imageDrawedLineAfterFilter);
    //    cv::imwrite("imageDrawedLineAfterFilterAndHist.bmp", imageDrawedLineAfterFilterAndHist);

    return 0;
}
