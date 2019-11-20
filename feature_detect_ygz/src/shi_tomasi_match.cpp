
#include "read_intrinsic.h"
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include "orb_slam_extractor/ORBextractor.h"
#include "gms_matcher.h"
#include "oclGoodFeatureTracker.h"

using namespace ORB_SLAM2;

ORBextractor *mpORBextractorLeft;
int main(int argc, char **argv)
{
    mpORBextractorLeft = new ORBextractor(1000, 1.2, 7, 20, 7);

    string file_path = "./camera.yaml"; // camera_fisheye_ankobot

    if(!readIntrinsic(file_path, intrinsicAndUndistort.K_Mat,
                      intrinsicAndUndistort.DistCoef, intrinsicAndUndistort.imageSize))
        return 0;

    Mat srcImg1, srcImg2;
    if(argc > 2) {
        srcImg1 = imread(string(argv[1]), IMREAD_GRAYSCALE);
        srcImg2 = imread(string(argv[2]), IMREAD_GRAYSCALE);
    }
    else {
        srcImg1 = imread("1.jpg", IMREAD_GRAYSCALE);
        srcImg2 = imread("2.jpg", IMREAD_GRAYSCALE);
    }

    cv::remap(srcImg1, srcImg1, intrinsicAndUndistort.mapx, intrinsicAndUndistort.mapy, cv::INTER_LINEAR);
    cv::remap(srcImg2, srcImg2, intrinsicAndUndistort.mapx, intrinsicAndUndistort.mapy, cv::INTER_LINEAR);

//            fastNlMeansDenoising(srcImg1, srcImg1);
//            fastNlMeansDenoising(srcImg2, srcImg2);

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    clahe->apply(srcImg1, srcImg1);
    clahe->apply(srcImg2, srcImg2);

    //    vector<Point2f> corners1, corners2;
    vector<KeyPoint> keyPoints1, keyPoints2;
    Mat srcImageForDrawKp1, srcImageForDrawKp2;
    TicToc t;
    computeShiTomasiCornor(srcImg1, keyPoints1, "srcImg1");
    cout<<"shi_tomasi corner time: "<<t.toc()<<" ms"<<endl;
    TicToc t4;
    mpORBextractorLeft->ComputeShi_TomasiOrientation(srcImg1, keyPoints1); //计算方向
    cout<<"  - ComputeShi_TomasiOrientation time: "<<t4.toc()<<" ms"<<endl;
    drawKeypoints(srcImg1, keyPoints1, srcImageForDrawKp1, Scalar(0,255,0),
                  DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    computeShiTomasiCornor(srcImg2, keyPoints2, "srcImg2");
    mpORBextractorLeft->ComputeShi_TomasiOrientation(srcImg2, keyPoints2);
    drawKeypoints(srcImg2, keyPoints2, srcImageForDrawKp2, Scalar(0,255,0),
                  DrawMatchesFlags::DRAW_RICH_KEYPOINTS);//DEFAULT


    cout<<"  - keyPoints1.size(): "<<keyPoints1.size()<<"  keyPoints2.size(): "<<keyPoints2.size()<<endl;

    //    vector<KeyPoint> keyPoints1(corners1.size()), keyPoints2(corners2.size());
    //    for(int i=0; i<corners1.size(); i++)
    //        keyPoints1[i] = KeyPoint(corners1[i], 3, -1, 0, 0, i);
    //    for(int i=0; i<corners2.size(); i++)
    //        keyPoints2[i] = KeyPoint(corners2[i], 3, -1, 0, 0, i);


    bool usingSift = false;
    if(argc > 3)
       usingSift = atoi(argv[3])?true:false;
    cv::Ptr<xfeatures2d::SiftDescriptorExtractor> siftDescriptor =
            xfeatures2d::SiftDescriptorExtractor::create(); //!@todo
    cv::Ptr<xfeatures2d::SurfDescriptorExtractor> suftDescriptor = xfeatures2d::SurfDescriptorExtractor::create();
    Ptr<DescriptorExtractor> orbDescriptor = ORB::create();

    /*【计算描述子】*/
    TicToc t1;
    Mat description1, description2;
    if(usingSift) {
        siftDescriptor->compute(srcImg1, keyPoints1, description1);
        cout<<">> compute descriptor time: "<<t1.toc()<<endl;
        siftDescriptor->compute(srcImg2, keyPoints2, description2);
    }
    else {
        orbDescriptor->compute(srcImg1, keyPoints1, description1);
        cout<<">> compute descriptor time: "<<t1.toc()<<endl;
        orbDescriptor->compute(srcImg2, keyPoints2, description2);
    }


    /*【匹配】*/
    vector<DMatch> matches;
    if(1)
    {
        if(usingSift) {
            cout<<">> using FLANN match"<<endl;
            FlannBasedMatcher matcher;
            matcher.match(description1, description2, matches);
        }
        else {
            cout<<">> using BFM match"<<endl;
            BFMatcher bfm( NORM_HAMMING );
            bfm.match(description1, description2, matches);
        }
    }
    else
    {
        const int TH_HIGH = 100;
        float vertialTh = 20, horizontal = 100;
        for(int i=0; i<description1.rows; i++)
        {
            int bestDist = 256;
            int bestIdx2 = -1;
            const cv::Mat &dp1 = description1.row(i);
            for(int j=0; j<description2.rows; j++)
            {
                Point2f errorPoint = keyPoints1[i].pt - keyPoints2[j].pt;
                if(fabs(errorPoint.x)>horizontal || fabs(errorPoint.y)>vertialTh)
                    continue;
                const int dist = DescriptorDistance(dp1, description2.row(j));
                if(dist < bestDist)
                {
                    bestDist = dist;
                    bestIdx2 = j;
                }
            }
            if(bestDist<=TH_HIGH)
            {
                matches.push_back(DMatch(i, bestIdx2, bestDist));
            }
        }
    }

    cout<<"  - matches.size(): "<<matches.size()<<endl;
    double max_dist = 0;
    double min_dist = 100;
    for(int i=0; i<matches.size(); i++) {
        double dist = matches[i].distance;
        if(dist < min_dist) min_dist = dist;
        if(dist > max_dist) max_dist = dist;
    }
    cout<<"  - max distance: "<<max_dist<< "  min distance: "<<min_dist<<endl;

    /*【初次筛选】*/
    vector<DMatch> good_matches;
    double dThreshold = 0.75;    //!@attention 匹配的阈值，越大匹配的点数越多
    for(int i=0; i<matches.size(); i++) {
        if(matches[i].distance < dThreshold * max_dist)
            good_matches.push_back(matches[i]);
    }

    Mat img_R_matches;
    drawMatches(srcImg1, keyPoints1,
                srcImg2, keyPoints2, good_matches, img_R_matches,
                Scalar::all(-1), Scalar::all(-1));

    putText(img_R_matches, string("kpt1: ")+to_string(keyPoints1.size()),
            Point(img_R_matches.size().width/2, 20), cv::FONT_HERSHEY_PLAIN, 1.2, Scalar(255,0,255));
    putText(img_R_matches, string("kpt2: ")+to_string(keyPoints2.size()),
            Point(img_R_matches.size().width/2, 40), cv::FONT_HERSHEY_PLAIN, 1.2, Scalar(255,0,255));
    putText(img_R_matches, string("match: ")+to_string(good_matches.size()),
            Point(img_R_matches.size().width/2, 60), cv::FONT_HERSHEY_PLAIN, 1.2, Scalar(255,0,255));

    imshow("before ransac", img_R_matches);
    cout<<"  - good_matches.size(): "<<good_matches.size()<<endl;

    // 将过滤得到的存起来备用
    vector<KeyPoint> R_keypoint01, R_keypoint02;
    for (int i=0; i<good_matches.size(); i++) {
        R_keypoint01.push_back(keyPoints1[good_matches[i].queryIdx]);
        R_keypoint02.push_back(keyPoints2[good_matches[i].trainIdx]);
        good_matches[i].queryIdx = i;
        good_matches[i].trainIdx = i;
    }

    if(0)
    {
        cout<<">> Using GMS: "<<endl;
        vector<DMatch> matches_gms;
        cv::xfeatures2d::matchGMS(srcImg1.size(), srcImg2.size(), R_keypoint01, R_keypoint02,
                                  good_matches, matches_gms, false, false, 4.0);

        cout<<"  -matches_gms.size(): "<<matches_gms.size()<<endl;
        good_matches.clear();
        good_matches.resize(0);
        good_matches = matches_gms;

        Mat imageGMSMatches;
        drawMatches(srcImg1, R_keypoint01,
                    srcImg2, R_keypoint02, good_matches, imageGMSMatches,
                    Scalar::all(-1), Scalar::all(-1));
        imshow("GMS_matches", imageGMSMatches);
        waitKey();
    }

    //坐标转换
    vector<Point2f> p01,p02;
    for (int i=0;i<good_matches.size();i++)
    {
        p01.push_back(R_keypoint01[i].pt);
        p02.push_back(R_keypoint02[i].pt);
    }

    //计算基础矩阵并剔除误匹配点
    vector<uchar> RansacStatus;
    Mat H = findHomography(p01, p02, RansacStatus, CV_RANSAC);
    //cv::findEssentialMat(p01, p02, intrinsicAndUndistort.K_Mat, CV_RANSAC, 0.5, 5.99, RansacStatus);

    //    Mat dst;
    //    warpPerspective(srcImg1, dst, H, Size(srcImg1.cols, srcImg1.rows));

    //剔除误匹配的点对
    vector<DMatch> RR_matches;            //重新定义RR_keypoint 和RR_matches来存储新的关键点和匹配矩阵
    for (int i=0;i<good_matches.size();i++)
    {
        if (RansacStatus[i]!=0)
            RR_matches.push_back(good_matches[i]);
    }
    cout<<"RR_matches.size(): "<<RR_matches.size()<<endl;

    //画出消除误匹配后的图
    Mat img_RR_matches;
    drawMatches(srcImg1, R_keypoint01,
                srcImg2, R_keypoint02, RR_matches,
                img_RR_matches, Scalar::all(-1), Scalar::all(-1));

    putText(img_RR_matches, string("kpt1: ")+to_string(R_keypoint01.size()),
            Point(img_RR_matches.size().width/2, 20), cv::FONT_HERSHEY_PLAIN, 1.2, Scalar(255,0,255));
    putText(img_RR_matches, string("kpt2: ")+to_string(R_keypoint02.size()),
            Point(img_RR_matches.size().width/2, 40), cv::FONT_HERSHEY_PLAIN, 1.2, Scalar(255,0,255));
    putText(img_RR_matches, string("match: ")+to_string(RR_matches.size()),
            Point(img_RR_matches.size().width/2, 60), cv::FONT_HERSHEY_PLAIN, 1.2, Scalar(255,0,255));


    imshow("after ransac", img_RR_matches);

    waitKey();
    imwrite( string("./res/") + "srcImage1.bmp", srcImageForDrawKp1 );
    imwrite( string("./res/") + "srcImage2.bmp", srcImageForDrawKp2 );
    imwrite(string("./res/before_ransac_")+(usingSift?"sift":"orb")+".bmp", img_R_matches);
    imwrite(string("./res/after_ransac_")+(usingSift?"sift":"orb")+".bmp", img_RR_matches);
    cout<<">> write images into './res'"<<endl;

    return 0;
}



