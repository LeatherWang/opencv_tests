
#include "read_intrinsic.h"
//#include "gms_matcher.h"

int main(int argc, char **argv)
{
    string file_path = "./camera.yaml"; // camera_fisheye_ankobot
    cout<<"intrinsic file: "<<file_path<<endl;

    //cv::Mat DistCoef_Zero = cv::Mat::zeros(4, 1, CV_64F);
    if(!readIntrinsic(file_path, intrinsicAndUndistort.K_Mat,
                      intrinsicAndUndistort.DistCoef, intrinsicAndUndistort.imageSize))
        return 1;


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

    fastNlMeansDenoising(srcImg1, srcImg1);
    fastNlMeansDenoising(srcImg2, srcImg2);

//        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
//        clahe->apply(srcImg1, srcImg1);
//        clahe->apply(srcImg2, srcImg2);


    //    imshow("srcImg2", srcImg2);
    //    waitKey();

    if(srcImg1.empty() || srcImg2.empty()) {
        cout<< "error to open image!" << endl;
        return 0;
    }


    cv::Ptr<xfeatures2d::SiftFeatureDetector> siftDetector =
            xfeatures2d::SiftFeatureDetector::create(2000);
    Ptr<cv::FastFeatureDetector> fastDetect =
            FastFeatureDetector::create(6, true, FastFeatureDetector::TYPE_9_16);
    Ptr<FeatureDetector> orbDetector = ORB::create(1000, 2, 2, 31, 0, 2, ORB::HARRIS_SCORE, 31, 6);


/*【角点检测】*/
    TicToc t;
    vector<KeyPoint> keyPoints1, keyPoints2;
    orbDetector->detect(srcImg1, keyPoints1);
    cout<<"> compute keypoint time: "<<t.toc()<<endl;
    orbDetector->detect(srcImg2, keyPoints2);
    cout << " - keyPoints1.size(): "<<keyPoints1.size() << "  keyPoints2.size(): "<<keyPoints2.size() << endl;

    //!@attention 绘制的时候，考虑关键点的大小和方向
    Mat feature_pic1, feature_pic2;
    drawKeypoints(srcImg1, keyPoints1, feature_pic1, Scalar(0, 255, 0),
                  DrawMatchesFlags::DEFAULT);
    drawKeypoints(srcImg2, keyPoints2, feature_pic2, Scalar(0, 255, 0),
                  DrawMatchesFlags::DEFAULT);
    imshow("feature1", feature_pic1); imshow("feature2", feature_pic2);

//    for(auto &ele:keyPoints1)
//        ele.angle = 0;
//    for(auto &ele:keyPoints2)
//        ele.angle = 0;
//    cout<<endl;

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
        cout<<"> compute descriptor time: "<<t1.toc()<<endl;
        siftDescriptor->compute(srcImg2, keyPoints2, description2);
    }
    else {
        orbDescriptor->compute(srcImg1, keyPoints1, description1);
        cout<<"> compute descriptor time: "<<t1.toc()<<endl;
        orbDescriptor->compute(srcImg2, keyPoints2, description2);
    }

/*【匹配】*/
    vector<DMatch> matches;
    if(usingSift) {
        cout<<"> using FLANN match"<<endl;
        FlannBasedMatcher matcher;
        matcher.match(description1, description2, matches);
    }
    else {
        cout<<"> using BFM match"<<endl;
        BFMatcher bfm( NORM_HAMMING );
        bfm.match(description1, description2, matches);
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
    double dThreshold = 0.6;    //!@attention 匹配的阈值，越大匹配的点数越多
    for(int i=0; i<matches.size(); i++) {
        if(matches[i].distance < dThreshold * max_dist)
            good_matches.push_back(matches[i]);
    }

    //根据matches将特征点对齐,将坐标转换为float类型
    vector<KeyPoint> R_keypoint01, R_keypoint02;
    for (int i=0; i<good_matches.size(); i++) {
        R_keypoint01.push_back(keyPoints1[good_matches[i].queryIdx]);
        R_keypoint02.push_back(keyPoints2[good_matches[i].trainIdx]);
        good_matches[i].queryIdx = i;
        good_matches[i].trainIdx = i;
    }

    Mat img_R_matches;
    drawMatches(srcImg1, R_keypoint01,
                srcImg2, R_keypoint02, good_matches, img_R_matches,
                Scalar::all(-1), Scalar::all(-1));
    imshow("before ransac", img_R_matches);
    cout<<"  - good_matches.size(): "<<good_matches.size()<<endl;

    if(1)
    {
        cout<<"> Using GMS: "<<endl;
        vector<DMatch> matches_gms;
        cv::xfeatures2d::matchGMS(srcImg1.size(), srcImg2.size(), R_keypoint01, R_keypoint02,
                                  good_matches, matches_gms, false, false, 4.0);

//        std::vector<bool> vbInliers;
//        gms_matcher gms(R_keypoint01, srcImg1.size(), R_keypoint02, srcImg2.size(), good_matches);
//        int num_inliers = gms.GetInlierMask(vbInliers, false, false);
//        cout << "  - Get total " << num_inliers << " matches." << endl;

//        for(int i=0; i<vbInliers.size(); i++)
//        {
//            if(vbInliers[i])
//                matches_gms.push_back(good_matches[i]);
//        }
        good_matches.clear();
        good_matches.resize(0);
        good_matches = matches_gms;

        Mat imageGMSMatches;
        drawMatches(srcImg1, R_keypoint01,
                    srcImg2, R_keypoint02, good_matches, imageGMSMatches,
                    Scalar::all(-1), Scalar::all(-1));
        imshow("GMS_matches", imageGMSMatches);
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
    imshow("after ransac", img_RR_matches);

    waitKey(0);
    imwrite(string("./res/before_ransac_")+(usingSift?"sift":"orb")+".bmp", img_R_matches);
    imwrite(string("./res/after_ransac_")+(usingSift?"sift":"orb")+".bmp", img_RR_matches);
    cout<<"> write images into './res'"<<endl;
    return 0;
}
