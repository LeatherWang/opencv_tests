#ifndef LINE_CORNERS_API
#define LINE_CORNERS_API

#include <iostream>
#include <vector>
#include <string>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include "opencv2/line_descriptor/descriptor.hpp"

#include <tic_toc.h>

using namespace std;
using namespace cv;

class LineCornersAPI
{
public:
    LineCornersAPI();
    ~LineCornersAPI(){};

    void addImage(const Mat &mCurrentImg, vector<KeyPoint> &vecKeyPoints, bool openClahe);

private:
    cv::Ptr<cv::CLAHE> clahe;

    cv::Ptr<cv::line_descriptor::BinaryDescriptor> mBDetector;
    bool bUsingMoreOneOctave=false;

    bool isOnLineSeg(const Point2f &startPoint, const Point2f &endPoint, const Point2f &inputPoint, float &response);
};

#endif
