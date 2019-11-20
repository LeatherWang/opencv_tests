#ifndef OCLGOODFEATURETRACKER
#define OCLGOODFEATURETRACKER

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <string>
#include "tic_toc.h"

using namespace cv;
using namespace std;

int eleSizeForDilate = 3;
Mat element = getStructuringElement(MORPH_RECT,
                                    Size(2*eleSizeForDilate+1, 2*eleSizeForDilate+1),
                                    Point( eleSizeForDilate, eleSizeForDilate ));

struct FeatureTrackParamUsingMask
{
    double qualityLevel = 0.001;
    int minDistance = 2;
    int blockSize = 5;
    bool useHarrisDetector = false;
    double k = 0.04;
    int maxCorners=1000;
} featureTrackParamUsingMask;

struct FeatureTrackParamUsingGrid
{
    // blocksize调小时，qualityLevel也要跟着调小，才能保证角点数目不变
    double qualityLevel = 0.075; //!@attention 修改为固定值，0.05~0.075之间变化
    int minDistance = 3;
    int blockSize = 5; //6 9 blocksize越小，角点的定位能力越好，瞎猫碰到死耗子
    bool useHarrisDetector = false;
    double k = 0.04;
    int maxCorners = 10; //500
} featureTrackParamUsingGrid;

struct greaterThanPtr : public std::binary_function<const float *, const float *, bool>
{
    bool operator () (const float * a, const float * b) const
    // Ensure a fully deterministic result of the sort
    { return (*a > *b) ? true : (*a < *b) ? false : (a > b); }
};

void ocvGoodFeaturesToTrack( InputArray _image, OutputArray _corners,
                             int maxCorners, double qualityLevel, double minDistance,
                             InputArray _mask, int blockSize, int gradientSize,
                             bool useHarrisDetector, double harrisK, vector<float> &vecCornersResponse);

void computeShiTomasiCornorUsingMask(const Mat &_srcGray, vector<KeyPoint> &vecKeyPoints,
                            const string &imageName);


void computeShiTomasiCornorUsingGrid(const Mat &_srcGray, const Mat &mask, vector<KeyPoint> &vecKeyPoints,
                            const string &imageName);

void computeShiTomasiCornorORBSLAM(const Mat &_srcGray, vector<KeyPoint> &vecKeyPoints,
                                     const string &imageName);

int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);
float shiTomasiScore(const cv::Mat& img, int u, int v);

#endif
