#ifndef VINS_FEATURE_TRACKER
#define VINS_FEATURE_TRACKER

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <string>
#include <tic_toc.h>

using namespace std;
using namespace cv;

const int WINDOW_SIZE = 10;
int n_id = 0;
cv::Mat prev_img, cur_img, forw_img;

vector<cv::KeyPoint> n_kps;//每一帧中新提取的特征点

// prev_img是上一次发布的帧的特征点数据
// cur_img是光流跟踪的前一帧的特征点数据
// forw_img是光流跟踪的后一帧的特征点数据
vector<cv::KeyPoint> cur_kps, forw_kps;

vector<int> ids;//能够被跟踪到的特征点的id

vector<int> track_cnt; //当前帧forw_img中每个特征点被追踪的时间次数

//去除无法追踪的特征
void reduceVector(vector<cv::KeyPoint> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

//去除无法追踪到的特征点
void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void addPoints()
{
    for (auto &p : n_kps)
    {
        forw_kps.push_back(p);//将新提取的特征点push到forw_kps中
        ids.push_back(-1);//新提取的特征点id初始化为-1
        track_cnt.push_back(1);//新提取的特征点被跟踪的次数初始化为1
    }
}

//更新特征点id
bool updateID(unsigned int i)
{
    if (i < ids.size())
    {
        if (ids[i] == -1)
            ids[i] = n_id++;
        return true;
    }
    else
        return false;
}

#endif //VINS_FEATURE_TRACKER
