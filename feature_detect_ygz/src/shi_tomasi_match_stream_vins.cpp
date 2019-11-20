
#include "read_intrinsic.h"
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include "orb_slam_extractor/ORBextractor.h"
#include "gms_matcher.h"
#include "vins_feature_tracker.h"
#include "shi_tomasi_api.h"

using namespace ORB_SLAM2;

ORBextractor *mpORBextractorLeft;

int main(int argc, char **argv)
{
    mpORBextractorLeft = new ORBextractor(1000, 1.2, 1, 20, 7); //!@todo 调整

    if(argc < 2)
    {
        cout<<"Usage error!!"<<endl;
        return 0;
    }

    std::vector<std::string> imageFilenames;
    string inputDir;
    std::string fileExtension;

    bool bUsingAnkobotDataSet = true;
    if(bUsingAnkobotDataSet)
    {
        string file_path = string(argv[1]) + "/camera.yaml"; // camera_fisheye_ankobot
        cout<<"intrinsic file: "<<file_path<<endl;
        //cv::Mat DistCoef_Zero = cv::Mat::zeros(4, 1, CV_64F);
        if(!readIntrinsic(file_path, intrinsicAndUndistort.K_Mat,
                          intrinsicAndUndistort.DistCoef, intrinsicAndUndistort.imageSize))
            return 1;


        inputDir = string(argv[1])+"/cam0";
        fileExtension = ".jpg";
    }
    else
    {
        undistortFisheyeFOVModel();
        inputDir = string("/media/leather/娱乐/data_set/ds1/subImages/cam0");
        fileExtension = ".pgm";
    }

    for (boost::filesystem::directory_iterator itr(inputDir); itr != boost::filesystem::directory_iterator(); ++itr)
    {
        if (!boost::filesystem::is_regular_file(itr->status()))
            continue;

        std::string filename = itr->path().filename().string();

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

    ofstream log;
    log.open("./shi_tomasi_log.csv");

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    Mat srcImg1, srcImg2;

    bool usingSift = false;
    if(argc >= 3)
        usingSift = (strcmp(argv[2], "true")==0)?true:false;
    cv::Ptr<xfeatures2d::SiftDescriptorExtractor> siftDescriptor =
            xfeatures2d::SiftDescriptorExtractor::create(); //!@todo
    cv::Ptr<xfeatures2d::SurfDescriptorExtractor> suftDescriptor = xfeatures2d::SurfDescriptorExtractor::create();
    Ptr<DescriptorExtractor> orbDescriptor = ORB::create();
    Ptr<cv::BRISK> briskDetector = cv::BRISK::create(5, 1);

    int startIndex = 0;
    if(argc >= 4)
        startIndex = atoi(argv[3]);

    int size = imageFilenames.size();
    for(int i=startIndex; i<size; i++) //300
    {
        cout<<imageFilenames[i].substr(imageFilenames[i].size()-30, 30)<<endl;
        log<<imageFilenames[i].substr(imageFilenames[i].size()-30, 30)<<endl;
        srcImg2 = imread(imageFilenames[i], IMREAD_GRAYSCALE);

        if(bUsingAnkobotDataSet)
        {
            cv::remap(srcImg2, srcImg2, intrinsicAndUndistort.mapx, intrinsicAndUndistort.mapy, cv::INTER_LINEAR);
            //clahe->apply(srcImg2, srcImg2);
        }
        else
            cv::remap(srcImg2, srcImg2, fovUndistortParam.map_x, fovUndistortParam.map_y, cv::INTER_LINEAR);


        vector<KeyPoint> keyPoints2;
        Mat description1, description2;


//        computeShiTomasiCornor(srcImg2, keyPoints2, "srcImg2");
        //computeShiTomasiCornorORBSLAM(srcImg2, keyPoints2, "srcImg2");

        (*mpORBextractorLeft)(srcImg2, cv::Mat(), keyPoints2, description2);

        // 去除伪角点，上面的shi_tomasi已经根据梯度这么做了一次，但是受到噪声影响



//        Mat workingMat = srcImg2.clone();
//        GaussianBlur(workingMat, workingMat, Size(7, 7), 2, 2, BORDER_REFLECT_101);
//        mpORBextractorLeft->ComputeShi_TomasiOrientation(workingMat, keyPoints2);

//        orbDescriptor->detect(srcImg2, keyPoints2);
//        drawKeypoints(srcImg2, keyPoints2, srcImageForDrawKp2, Scalar(0,255,0),
//                      DrawMatchesFlags::DRAW_RICH_KEYPOINTS);//DEFAULT
        Mat srcImageForDrawKp2 = srcImg2.clone(), srcImageForDrawKpOctave;
        if(srcImageForDrawKp2.channels() == 1)
            cvtColor(srcImageForDrawKp2, srcImageForDrawKp2, CV_GRAY2BGR);
        srcImageForDrawKpOctave = srcImageForDrawKp2.clone();
        for(KeyPoint ele:keyPoints2)
        {
            if(ele.octave < 1)
                cv::circle(srcImageForDrawKp2, ele.pt, 2, cv::Scalar(0, 255, 0));
        }

        for(KeyPoint ele:keyPoints2)
        {
            if(ele.octave >= 1)
                cv::circle(srcImageForDrawKpOctave, ele.pt, 2, cv::Scalar(0, 0, 255));
        }
        imshow("srcImageForDrawKpOctave", srcImageForDrawKpOctave);
        imshow("srcImageForDrawKp2", srcImageForDrawKp2);




//        orbDescriptor->compute(srcImg2, keyPoints2, description2);


        if(!cur_kps.empty())
        {
            vector<DMatch> matches;
            orbDescriptor->compute(srcImg1, cur_kps, description1);
            if(0)
            {
                if(usingSift) {
                    cout<<">> using FLANN match"<<endl;
                    log <<">> using FLANN match"<<endl;
                    FlannBasedMatcher matcher;
                    matcher.match(description1, description2, matches);
                }
                else {
                    cout<<">> using BFM match"<<endl;
                    log <<">> using BFM match"<<endl;
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
                        Point2f errorPoint = cur_kps[i].pt - keyPoints2[j].pt;
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
            log<<"  - matches.size(): "<<matches.size()<<endl;
            double max_dist = 0;
            double min_dist = 100;
            for(int i=0; i<matches.size(); i++) {
                double dist = matches[i].distance;
                if(dist < min_dist) min_dist = dist;
                if(dist > max_dist) max_dist = dist;
            }
            cout<<"  - max distance: "<<max_dist<< "  min distance: "<<min_dist<<endl;
            log<<"  - max distance: "<<max_dist<< "  min distance: "<<min_dist<<endl;

            /*【初次筛选】*/
            vector<DMatch> good_matches;
            double dThreshold = 0.75;    //!@attention 匹配的阈值，越大匹配的点数越多
            for(int i=0; i<matches.size(); i++) {
                if(matches[i].distance < dThreshold * max_dist)
                    good_matches.push_back(matches[i]);
            }

            Mat img_R_matches;

            drawMatches(srcImg1, cur_kps,
                        srcImg2, keyPoints2, good_matches, img_R_matches,
                        Scalar::all(-1), Scalar::all(-1));

            putText(img_R_matches, string("kpt1: ")+to_string(cur_kps.size()),
                    Point(img_R_matches.size().width/2, 20), cv::FONT_HERSHEY_PLAIN, 1.2, Scalar(255,0,255));
            putText(img_R_matches, string("kpt2: ")+to_string(keyPoints2.size()),
                    Point(img_R_matches.size().width/2, 40), cv::FONT_HERSHEY_PLAIN, 1.2, Scalar(255,0,255));
            putText(img_R_matches, string("match: ")+to_string(good_matches.size()),
                    Point(img_R_matches.size().width/2, 60), cv::FONT_HERSHEY_PLAIN, 1.2, Scalar(255,0,255));

            imshow("before ransac", img_R_matches);
            cout<<"  - good_matches.size(): "<<good_matches.size()<<endl;
            log<<"  - good_matches.size(): "<<good_matches.size()<<endl;



            //坐标转换
            vector<Point2f> p01,p02;
            for (int i=0;i<good_matches.size();i++)
            {
                p01.push_back(cur_kps[good_matches[i].queryIdx].pt);
                p02.push_back(keyPoints2[good_matches[i].trainIdx].pt);
            }

            //计算基础矩阵并剔除误匹配点
            vector<uchar> RansacStatus;
            Mat H = findHomography(p01, p02, RansacStatus, CV_RANSAC);

            vector<uchar> status(cur_kps.size(), 0);
            forw_kps.clear();
            forw_kps.resize(cur_kps.size());

            vector<DMatch> RR_matches;
            set<int> setTrackedKps;
            for (int i=0;i<good_matches.size();i++)
            {
                if (RansacStatus[i]!=0)
                {
                    RR_matches.push_back(good_matches[i]);
                    status[good_matches[i].queryIdx] = 255;
                    forw_kps[good_matches[i].queryIdx] =
                            keyPoints2[good_matches[i].trainIdx]; //跟踪成功，记录下新位置
                    setTrackedKps.insert(good_matches[i].trainIdx);
                }
            }

            //画出消除误匹配后的图
            Mat img_RR_matches;
            drawMatches(srcImg1, cur_kps,
                        srcImg2, keyPoints2, RR_matches,
                        img_RR_matches, Scalar::all(-1), Scalar::all(-1));

            putText(img_RR_matches, string("kpt1: ")+to_string(cur_kps.size()),
                    Point(img_RR_matches.size().width/2, 20), cv::FONT_HERSHEY_PLAIN, 1.2, Scalar(255,0,255));
            putText(img_RR_matches, string("kpt2: ")+to_string(keyPoints2.size()),
                    Point(img_RR_matches.size().width/2, 40), cv::FONT_HERSHEY_PLAIN, 1.2, Scalar(255,0,255));
            putText(img_RR_matches, string("match: ")+to_string(RR_matches.size()),
                    Point(img_RR_matches.size().width/2, 60), cv::FONT_HERSHEY_PLAIN, 1.2, Scalar(255,0,255));
            imshow("after ransac", img_RR_matches);


            reduceVector(forw_kps, status); //当前跟踪结果，记录keyPoint的位置
            reduceVector(cur_kps, status); //上一次跟踪结果，cur_kps与keyPoints2进行暴力匹配
            reduceVector(ids, status);
            reduceVector(track_cnt, status);

            for (auto &n : track_cnt) //对跟踪成功的进行，增加计数
                n++;

            n_kps.clear();
            for(int i=0; i<keyPoints2.size(); i++)
            {
                if(setTrackedKps.count(i) < 1)
                    n_kps.push_back(keyPoints2[i]);
            }

            addPoints(); //将没有跟踪上的特征点放入vector的后部分

            cur_kps = forw_kps;

            for (unsigned int i = 0;; i++)
                if (!updateID(i))
                    break;

            Mat tmp_img = srcImg2.clone();
            if(tmp_img.channels() == 1)
                cvtColor(tmp_img, tmp_img, CV_GRAY2BGR);
            for (int j = cur_kps.size()-1; j >= 0; j--)
            {
                double len = std::min(1.0, 1.0 * track_cnt[j] / WINDOW_SIZE);
                cv::circle(tmp_img, cur_kps[j].pt, 2,
                           cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
            }
            cv::imshow("tmp_image", tmp_img);
        }
        else
        {
            n_kps.clear();
            for(int i=0; i<keyPoints2.size(); i++)
                n_kps.push_back(keyPoints2[i]);
            addPoints();
            cur_kps = forw_kps;
        }

        srcImg1 = srcImg2;
        waitKey();
    }

    return 0;
}



