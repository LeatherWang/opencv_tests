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

struct FeatureTrackParam
{
    double qualityLevel = 0.05;
    int minDistance = 20;
    int blockSize = 9;
    bool useHarrisDetector = false;
    double k = 0.04;
    int maxCorners=400;
} featureTrackParam;

void oclGoodFeaturesToTrack( InputArray _image, OutputArray _corners,
                             int maxCorners, double qualityLevel, double minDistance,
                             InputArray _mask, int blockSize, int gradientSize,
                             bool useHarrisDetector, double harrisK, vector<float> &vecCornersResponse);

void computeShiTomasiCornor(const Mat &_srcGray, vector<KeyPoint> &vecKeyPoints,
                            const string &imageName)
{
    vector<Point2f> corners, filterCorners, meanCornors, singleCorners;

    Mat srcGray = _srcGray.clone(), mask;

    double cannyTh = 22;
    blur( srcGray, srcGray, Size(3,3) );
    Canny(srcGray, mask, cannyTh, cannyTh*3, 3);

    cv::dilate(mask, mask, element);//进行膨胀操作

    imshow(imageName + "_mask", mask);

    TicToc t1;
    vector<float> vecCornersResponse;
    oclGoodFeaturesToTrack( srcGray,
                            corners,
                            featureTrackParam.maxCorners,
                            featureTrackParam.qualityLevel,
                            featureTrackParam.minDistance,
                            mask,
                            featureTrackParam.blockSize,
                            3,
                            featureTrackParam.useHarrisDetector,
                            featureTrackParam.k,
                            vecCornersResponse);
    cout<<"  - goodFeaturesToTrack time: "<<t1.toc()<<" ms"<<endl;

    Mat srcImageForDraw = srcGray.clone();
    if(srcImageForDraw.channels() == 1)
        cvtColor(srcImageForDraw, srcImageForDraw, CV_GRAY2BGR);
    for( int i = 0; i < corners.size(); i++ ){
        circle( srcImageForDraw, corners[i], 2, Scalar(0,255,0), 1, 8, 0 );
    }
    imshow(imageName+string("_raw_corner"), srcImageForDraw);

    if(0)
    {
        /*【加权均值】*/
        TicToc t3;
        float disTh = 100; //10pixel
        vector<bool> vecVisited(corners.size(), false);
        filterCorners.reserve(corners.size());
        for(int i=0; i<corners.size(); i++)
        {
            if(!vecVisited[i])
            {
                vecVisited[i] = true;
                int counter = 1;
                float weightNum = vecCornersResponse[i];
                Point2f numPoints = corners[i] * vecCornersResponse[i];
                for(int j=i+1; j<corners.size(); j++) //查找相邻点
                {
                    if(!vecVisited[j])
                    {
                        Point2f errorToAdj = corners[j]-corners[i];
                        float disToAdj = errorToAdj.dot(errorToAdj);
                        if(disToAdj < disTh)
                        {
                            vecVisited[j] = true;
                            numPoints += corners[j] * vecCornersResponse[j];
                            weightNum += vecCornersResponse[j];
                            counter++;
                        }
                    }
                }

                if(counter > 1)
                    meanCornors.push_back(numPoints/weightNum); //!@todo 使用response作为权重
                else
                    singleCorners.push_back(numPoints/weightNum);
            }
        }
        cout<<"  - filterCorners time: "<<t3.toc()<<" ms"<<endl;

        srcImageForDraw = srcGray.clone();
        if(srcImageForDraw.channels() == 1)
            cvtColor(srcImageForDraw, srcImageForDraw, CV_GRAY2BGR);
        for(auto &ele:meanCornors){
            circle( srcImageForDraw, ele, 2, Scalar(0,255,0), 1, 8, 0 );
        }
        for(auto &ele:singleCorners){
            circle( srcImageForDraw, ele, 2, Scalar(0,255,0), 1, 8, 0 );
        }
        cv::imshow(imageName+"_meanFilter", srcImageForDraw);


        /*【优化角点亚像素级别精度】*/
        // 只对取过均值的进行这种操作
        if(false && !meanCornors.empty())
        {
            TicToc t2;
            TermCriteria criteria(TermCriteria::MAX_ITER|TermCriteria::EPS, 100, 0.001);
            cornerSubPix(srcGray, meanCornors, Size(5,5), Size(-1,-1), criteria);
            cout<<"  - cornerSubPix time: "<<t2.toc()<<" ms"<<endl;
        }

        filterCorners  = meanCornors;
        for(auto &ele:singleCorners)
            filterCorners.push_back(ele);

        srcImageForDraw = srcGray.clone();
        if(srcImageForDraw.channels() == 1)
            cvtColor(srcImageForDraw, srcImageForDraw, CV_GRAY2BGR);
        for(auto &ele:filterCorners){
            circle( srcImageForDraw, ele, 2, Scalar(0,255,0), 1, 8, 0 );
        }
        cv::imshow(imageName+"_meanFilter_and_sub_pixel", srcImageForDraw);
    }
    else
        filterCorners = corners;

    /*【生成关键点】*/
    vecKeyPoints.reserve(filterCorners.size());
    for(int i=0; i<filterCorners.size(); i++)
        vecKeyPoints.emplace_back(KeyPoint(filterCorners[i], featureTrackParam.blockSize, 0, 0, 0, i));
}




struct FeatureTrackParamUsingGrid
{
    double qualityLevel = 0.1;
    int minDistance = 3;
    int blockSize = 7;
    bool useHarrisDetector = false;
    double k = 0.04;
    int maxCorners=10;
} featureTrackParamUsingGrid;
void computeShiTomasiCornorForGrid(const Mat &_srcGray, const Mat &mask, vector<KeyPoint> &vecKeyPoints,
                            const string &imageName)
{
    vector<Point2f> corners, filterCorners;

    Mat srcGray = _srcGray.clone();

    vector<float> vecCornersResponse;
    oclGoodFeaturesToTrack( srcGray,
                            corners,
                            featureTrackParamUsingGrid.maxCorners,
                            featureTrackParamUsingGrid.qualityLevel,
                            featureTrackParamUsingGrid.minDistance,
                            mask,
                            featureTrackParamUsingGrid.blockSize,
                            3,
                            featureTrackParamUsingGrid.useHarrisDetector,
                            featureTrackParamUsingGrid.k,
                            vecCornersResponse);

    Mat srcImageForDraw = srcGray.clone();
    if(srcImageForDraw.channels() == 1)
        cvtColor(srcImageForDraw, srcImageForDraw, CV_GRAY2BGR);
    for( size_t i = 0; i < corners.size(); i++ ){
        circle( srcImageForDraw, corners[i], 2, Scalar(0,255,0), 1, 8, 0 );
    }

    filterCorners = corners;

    /*【生成关键点】*/
    vecKeyPoints.reserve(filterCorners.size());
    for(size_t i=0; i<filterCorners.size(); i++)
        vecKeyPoints.emplace_back(KeyPoint(filterCorners[i], featureTrackParam.blockSize, 0, 0, 0, i));
}

struct greaterThanPtr : public std::binary_function<const float *, const float *, bool>
{
    bool operator () (const float * a, const float * b) const
    // Ensure a fully deterministic result of the sort
    { return (*a > *b) ? true : (*a < *b) ? false : (a > b); }
};

void oclGoodFeaturesToTrack( InputArray _image, OutputArray _corners,
                             int maxCorners, double qualityLevel, double minDistance,
                             InputArray _mask, int blockSize, int gradientSize,
                             bool useHarrisDetector, double harrisK, vector<float> &vecCornersResponse)
{
    //    CV_INSTRUMENT_REGION()

    CV_Assert( qualityLevel > 0 && minDistance >= 0 && maxCorners >= 0 );
    CV_Assert( _mask.empty() || (_mask.type() == CV_8UC1 && _mask.sameSize(_image)) );

    //    CV_OCL_RUN(_image.dims() <= 2 && _image.isUMat(),
    //               ocl_goodFeaturesToTrack(_image, _corners, maxCorners, qualityLevel, minDistance,
    //                                    _mask, blockSize, gradientSize, useHarrisDetector, harrisK))

    Mat image = _image.getMat(), eig, tmp;
    if (image.empty())
    {
        _corners.release();
        return;
    }

    // Disabled due to bad accuracy
    //    CV_OVX_RUN(false && useHarrisDetector && _mask.empty() &&
    //               !ovx::skipSmallImages<VX_KERNEL_HARRIS_CORNERS>(image.cols, image.rows),
    //               openvx_harris(image, _corners, maxCorners, qualityLevel, minDistance, blockSize, gradientSize, harrisK))

    if( useHarrisDetector )
        cornerHarris( image, eig, blockSize, gradientSize, harrisK );
    else
        cornerMinEigenVal( image, eig, blockSize, gradientSize );

    double maxVal = 0;
    minMaxLoc( eig, 0, &maxVal, 0, 0, _mask );
    threshold( eig, eig, maxVal*qualityLevel, 0, THRESH_TOZERO );
    dilate( eig, tmp, Mat());

    Size imgsize = image.size();
    std::vector<const float*> tmpCorners;

    // collect list of pointers to features - put them into temporary image
    Mat mask = _mask.getMat();
    for( int y = 1; y < imgsize.height - 1; y++ )
    {
        const float* eig_data = (const float*)eig.ptr(y);
        const float* tmp_data = (const float*)tmp.ptr(y);
        const uchar* mask_data = mask.data ? mask.ptr(y) : 0;

        for( int x = 1; x < imgsize.width - 1; x++ )
        {
            float val = eig_data[x];
            if( val != 0 && val == tmp_data[x] && (!mask_data || mask_data[x]) )
                tmpCorners.push_back(eig_data + x);
        }
    }

    std::vector<Point2f> corners;
    size_t i, j, total = tmpCorners.size(), ncorners = 0;

    if (total == 0)
    {
        _corners.release();
        return;
    }

    std::sort( tmpCorners.begin(), tmpCorners.end(), greaterThanPtr() );

    if (minDistance >= 1)
    {
        // Partition the image into larger grids
        int w = image.cols;
        int h = image.rows;

        const int cell_size = cvRound(minDistance);
        const int grid_width = (w + cell_size - 1) / cell_size;
        const int grid_height = (h + cell_size - 1) / cell_size;

        std::vector<std::vector<Point2f> > grid(grid_width*grid_height);

        minDistance *= minDistance;

        for( i = 0; i < total; i++ )
        {
            int ofs = (int)((const uchar*)tmpCorners[i] - eig.ptr());
            int y = (int)(ofs / eig.step);
            int x = (int)((ofs - y*eig.step)/sizeof(float));

            bool good = true;

            int x_cell = x / cell_size;
            int y_cell = y / cell_size;

            int x1 = x_cell - 1;
            int y1 = y_cell - 1;
            int x2 = x_cell + 1;
            int y2 = y_cell + 1;

            // boundary check
            x1 = std::max(0, x1);
            y1 = std::max(0, y1);
            x2 = std::min(grid_width-1, x2);
            y2 = std::min(grid_height-1, y2);

            for( int yy = y1; yy <= y2; yy++ )
            {
                for( int xx = x1; xx <= x2; xx++ )
                {
                    std::vector <Point2f> &m = grid[yy*grid_width + xx];

                    if( m.size() )
                    {
                        for(j = 0; j < m.size(); j++)
                        {
                            float dx = x - m[j].x;
                            float dy = y - m[j].y;

                            if( dx*dx + dy*dy < minDistance )
                            {
                                good = false;
                                goto break_out;
                            }
                        }
                    }
                }
            }

break_out:

            if (good)
            {
                grid[y_cell*grid_width + x_cell].push_back(Point2f((float)x, (float)y));

                corners.push_back(Point2f((float)x, (float)y));
                vecCornersResponse.push_back(*tmpCorners[i]);
                ++ncorners;

                if( maxCorners > 0 && (int)ncorners == maxCorners )
                    break;
            }
        }
    }
    else
    {
        for( i = 0; i < total; i++ )
        {
            int ofs = (int)((const uchar*)tmpCorners[i] - eig.ptr());
            int y = (int)(ofs / eig.step);
            int x = (int)((ofs - y*eig.step)/sizeof(float));

            corners.push_back(Point2f((float)x, (float)y));
            vecCornersResponse.push_back(*tmpCorners[i]);
            ++ncorners;
            if( maxCorners > 0 && (int)ncorners == maxCorners )
                break;
        }
    }

    Mat(corners).convertTo(_corners, _corners.fixedType() ? _corners.type() : CV_32F);
}

const int EDGE_THRESHOLD = 19;
const int nfeatures = 1000;
void computeShiTomasiCornorUsingGrid(const Mat &_srcGray, vector<KeyPoint> &vecKeyPoints,
                                     const string &imageName)
{
    const float W = 40;

    const int minBorderX = EDGE_THRESHOLD-3; //EDGE_THRESHOLD: 19
    const int minBorderY = minBorderX;
    const int maxBorderX = _srcGray.cols-EDGE_THRESHOLD+3;
    const int maxBorderY = _srcGray.rows-EDGE_THRESHOLD+3;

    vector<cv::KeyPoint> vToDistributeKeys;
    vToDistributeKeys.reserve(nfeatures*10);

    const float width = (maxBorderX-minBorderX);
    const float height = (maxBorderY-minBorderY);

    // 计算每行每列格子数目
    const int nCols = width/W;
    const int nRows = height/W;

    // 重新计算每个格子的大小
    const int wCell = ceil(width/nCols);
    const int hCell = ceil(height/nRows);

    Mat srcGray = _srcGray.clone(), mask;
    double cannyTh = 22;
    blur( srcGray, srcGray, Size(3,3) );
    Canny(srcGray, mask, cannyTh, cannyTh*3, 3);
    cv::dilate(mask, mask, element);
    imshow("mask", mask);

    for(int i=0; i<nRows; i++)
    {
        const float iniY =minBorderY + i*hCell;
        float maxY = iniY+hCell+6; //!@attention 对栅格边界上的角点的处理

        if(iniY>=maxBorderY-3)
            continue;
        if(maxY>maxBorderY)
            maxY = maxBorderY;

        for(int j=0; j<nCols; j++)
        {
            const float iniX =minBorderX+j*wCell;
            float maxX = iniX+wCell+6;

            if(iniX>=maxBorderX-6)
                continue;
            if(maxX>maxBorderX)
                maxX = maxBorderX;

            vector<cv::KeyPoint> vKeysCell;

            const Mat roiImage = srcGray.rowRange(iniY,maxY).colRange(iniX,maxX); //srcGray是模糊过后的
            const Mat roiMask = mask.rowRange(iniY,maxY).colRange(iniX,maxX);

            computeShiTomasiCornorForGrid(roiImage, roiMask, vKeysCell, imageName);

            if(!vKeysCell.empty())
            {
                for(vector<cv::KeyPoint>::iterator vit=vKeysCell.begin(); vit!=vKeysCell.end();vit++)
                {
                    (*vit).pt.x +=j*wCell; //fast角点在图像中的坐标
                    (*vit).pt.y +=i*hCell;
                    vToDistributeKeys.push_back(*vit);
                }
            }
        }
    }

    vecKeyPoints = vToDistributeKeys;
    for(KeyPoint &ele:vecKeyPoints)
    {
        ele.pt.x += minBorderX; //特征点的x坐标
        ele.pt.y += minBorderY;
    }
}

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


float shiTomasiScore(const cv::Mat& img, int u, int v)
{
    assert(img.type() == CV_8UC1);

    float dXX = 0.0;
    float dYY = 0.0;
    float dXY = 0.0;
    const int halfbox_size = 4;
    const int box_size = 2*halfbox_size;
    const int box_area = box_size*box_size;
    const int x_min = u-halfbox_size;
    const int x_max = u+halfbox_size;
    const int y_min = v-halfbox_size;
    const int y_max = v+halfbox_size;

    if(x_min < 1 || x_max >= img.cols-1 || y_min < 1 || y_max >= img.rows-1)
        return 0.0; // patch is too close to the boundary

    const int stride = img.step.p[0];
    for( int y=y_min; y<y_max; ++y )
    {
        const uint8_t* ptr_left   = img.data + stride*y + x_min - 1;
        const uint8_t* ptr_right  = img.data + stride*y + x_min + 1;
        const uint8_t* ptr_top    = img.data + stride*(y-1) + x_min;
        const uint8_t* ptr_bottom = img.data + stride*(y+1) + x_min;
        for(int x = 0; x < box_size; ++x, ++ptr_left, ++ptr_right, ++ptr_top, ++ptr_bottom)
        {
            float dx = *ptr_right - *ptr_left;
            float dy = *ptr_bottom - *ptr_top;
            dXX += dx*dx;
            dYY += dy*dy;
            dXY += dx*dy;
        }
    }

    // Find and return smaller eigenvalue:
    dXX = dXX / (2.0 * box_area);
    dYY = dYY / (2.0 * box_area);
    dXY = dXY / (2.0 * box_area);
    return 0.5 * (dXX + dYY - sqrt( (dXX + dYY) * (dXX + dYY) - 4 * (dXX * dYY - dXY * dXY) ));
}

#endif
