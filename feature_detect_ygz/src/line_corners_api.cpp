
#include "line_corners_api.h"

const double angle2Ran = M_PI/180.0f;
const double angleDisTh = 60*angle2Ran;
const double minAngleDisTh = M_PI_2 - angleDisTh;
const double maxAngleDisTh = M_PI_2 + angleDisTh;
const double point2PointDisTh = 20*20;
const float onLineSegTh = 50.0f;

LineCornersAPI::LineCornersAPI()
{
    clahe = cv::createCLAHE(3.0, cv::Size(8, 8));

    cv::line_descriptor::BinaryDescriptor::Params tmParams;
    tmParams.numOfOctave_ = 5;
    if(tmParams.numOfOctave_ > 1)
        bUsingMoreOneOctave = true;

    tmParams.widthOfBand_ = 6;
    tmParams.reductionRatio = 1.2; //每层金字塔降采样比例，这里修改了源码
    tmParams.ksize_ = 3; //高斯模糊，用于生成高斯金字塔
    mBDetector = cv::line_descriptor::BinaryDescriptor::createBinaryDescriptor(tmParams);
}

void LineCornersAPI::addImage(const Mat &mCurrentImg, vector<KeyPoint> &vecKeyPoints, bool openClahe)
{
    // 外部进行直方图均衡化
    if(openClahe)
        clahe->apply(mCurrentImg, mCurrentImg);

    /*【步骤0】: 检测线段*/
    vector<cv::line_descriptor::KeyLine> keylinesOctave;
    mBDetector->detect(mCurrentImg, keylinesOctave);

    if(keylinesOctave.empty())
        return;

    //! @attention 修改，融合不同层相同的线段，使用最长距离
    vector<cv::line_descriptor::KeyLine> keylines;
    keylines.reserve(keylinesOctave.size());
    if(1 && bUsingMoreOneOctave)
    {
        std::map<int, int> mapClassIDVecID;
        uint size = keylinesOctave.size();
        for (uint i = 0; i < size; i++)
        {
            auto iter = mapClassIDVecID.find(keylinesOctave[i].class_id);
            if(iter != mapClassIDVecID.end())
            {
                const cv::line_descriptor::KeyLine &curKeyline = keylinesOctave[i];
                cv::line_descriptor::KeyLine &originKeyline = keylines[iter->second];

                float maxLineLength = curKeyline.lineLength;
                cv::Point2f startPointMax = curKeyline.getStartPoint();
                cv::Point2f endPointMax = curKeyline.getEndPoint();

                // 修正终点
                cv::Point2f errorPoint = originKeyline.getEndPoint() - startPointMax;
                float dis = sqrt(errorPoint.dot(errorPoint));
                if(maxLineLength < dis)
                {
                    maxLineLength = dis;
                    endPointMax = originKeyline.getEndPoint();
                }

                // 修正起点
                errorPoint = endPointMax - originKeyline.getStartPoint();
                dis = sqrt(errorPoint.dot(errorPoint));
                if(maxLineLength < dis)
                {
                    maxLineLength = dis;
                    startPointMax = originKeyline.getStartPoint();
                }

                originKeyline.lineLength = maxLineLength;
                originKeyline.response = originKeyline.lineLength/max(mCurrentImg.size().width, mCurrentImg.size().height);
                originKeyline.startPointX = startPointMax.x;
                originKeyline.startPointY = startPointMax.y;
                originKeyline.endPointX = endPointMax.x;
                originKeyline.endPointY = endPointMax.y;
            }
            else
            {
                mapClassIDVecID.insert(make_pair(keylinesOctave[i].class_id, keylines.size()));
                keylines.push_back(keylinesOctave[i]);
            }
        }
    }
    else
    {
        keylines = keylinesOctave;
    }

//    cout<<"keylines.size(): "<<keylines.size()<<endl;
    if(keylines.size() == 0)
        return;

    for (uint i = 0; i < keylines.size(); i++)
        keylines[i].class_id = i;

    Mat imageCornersResponse = cv::Mat::zeros(mCurrentImg.size(), CV_32F);

    /*【步骤1】: 计算线段的交叉点*/
    int klNum = keylines.size();
    for(int i=0; i<klNum; i++)
    {
        vector<cv::Point2f> vecLinePoints1;
        vecLinePoints1.push_back(keylines[i].getStartPoint());
        vecLinePoints1.push_back(keylines[i].getEndPoint());
        //ax+by+c=0
        cv::Point2f errorPoint = vecLinePoints1[1] - vecLinePoints1[0];
        cv::Vec3f lineEquationCoff1;
        lineEquationCoff1[0] = -errorPoint.y;
        lineEquationCoff1[1] = errorPoint.x;
        lineEquationCoff1[2] = errorPoint.y*vecLinePoints1[0].x - errorPoint.x*vecLinePoints1[0].y;

        for(int j=i+1; j<klNum; j++)
        {
            double errorAngle = fabs(keylines[i].angle - keylines[j].angle);
            while(errorAngle > M_PI)
                errorAngle -= M_PI;
            if(minAngleDisTh < errorAngle && errorAngle < maxAngleDisTh) //角度阈值
            {
                vector<cv::Point2f> vecLinePoints2;
                vecLinePoints2.push_back(keylines[j].getStartPoint());
                vecLinePoints2.push_back(keylines[j].getEndPoint());

                cv::Vec3f lineEquationCoff2;
                cv::Point2f errorPoint = vecLinePoints2[1] - vecLinePoints2[0];
                lineEquationCoff2[0] = -errorPoint.y;
                lineEquationCoff2[1] = errorPoint.x;
                lineEquationCoff2[2] = errorPoint.y*vecLinePoints2[0].x - errorPoint.x*vecLinePoints2[0].y;

                cv::Vec3f crossPointVec = lineEquationCoff1.cross(lineEquationCoff2);

                if(crossPointVec[2] != 0)
                {
                    Point2f crossPoint = Point2f(crossPointVec[0]/crossPointVec[2],crossPointVec[1]/crossPointVec[2]);

                    if(0<crossPoint.x && crossPoint.x<mCurrentImg.size().width &&
                            0<crossPoint.y && crossPoint.y<mCurrentImg.size().height)
                    {
                        float response1=0.0f, response2=0.0f;
                        if(isOnLineSeg(vecLinePoints1[0], vecLinePoints1[1], crossPoint, response1) &&
                                isOnLineSeg(vecLinePoints2[0], vecLinePoints2[1], crossPoint, response2))
                        {
                            Point2i point(std::round(crossPoint.x), std::round(crossPoint.y));
                            if(point.x >= mCurrentImg.size().width)
                                point.x = mCurrentImg.size().width-1;
                            if(point.y >= mCurrentImg.size().height)
                                point.y = mCurrentImg.size().height-1;
                            float responseSum = response1+response2;

                            //float cornerScore = shiTomasiScore(mCurrentImg, point.x, point.y);

                            if(imageCornersResponse.at<float>(point) < responseSum)
                                imageCornersResponse.at<float>(point) = responseSum;
                        }
                    }
                }
            }
        }
    }

    /*【步骤2】: 非极大值抑制*/
    Mat imageCorners = cv::Mat::zeros(mCurrentImg.size(), mCurrentImg.type());
    int i_s[8] = { -1, -1, -1, 0, 0, 1, 1, 1 };
    int j_s[8] = { -1, 0, 1, -1, 1, -1, 0, 1 };
    for (int i = 1; i < imageCornersResponse.size().height - 1; i++)
    {
        for (int j = 1; j < imageCornersResponse.size().width - 1; j++)
        {
            if(imageCornersResponse.at<float>(i,j) > 0.0f)
            {
                int k = 0;
                for (; k < 8; k++)
                {
                    if (imageCornersResponse.at<float>(i,j) < imageCornersResponse.at<float>(i+i_s[k],j+j_s[k]))
                        break;
                }
                if (k == 8)
                    imageCorners.at<uchar>(i,j) = 255;
            }
        }
    }

    // 边缘点剔除


    // 优化亚像素，不可行，交叉位置检测出来的四个点，一经优化变成了一个点

    Mat imageForDrawKps = mCurrentImg.clone();
    if(imageForDrawKps.channels() == 1)
        cvtColor(imageForDrawKps, imageForDrawKps, CV_GRAY2BGR);
    Mat imageForDrawLine = imageForDrawKps.clone();

    for(int row=0; row < imageCorners.size().height; row++)
    {
        uchar* rowPtr = imageCorners.ptr(row);
        for(int col=0; col < imageCorners.size().width; col++)
        {
            if(rowPtr[col] > 0)
            {
                vecKeyPoints.emplace_back(
                            KeyPoint(Point2f(col, row), 9));
                cv::circle(imageForDrawKps, Point(col, row), 2, cv::Scalar(0,255,0));
            }
        }
    }

    for (uint i = 0; i < keylines.size(); i++)
    {
        if(keylines[i].octave < 1)
            cv::line(imageForDrawLine, keylines[i].getStartPoint(), keylines[i].getEndPoint(),
                     cv::Scalar(0, 0, 255), 1);
        else
            cv::line(imageForDrawLine, keylines[i].getStartPoint(), keylines[i].getEndPoint(),
                     cv::Scalar(0, 128, 255), 1);
    }

    //        imshow("imageCornersResponse", imageCornersResponse);
    //        imshow("imageForDrawKps", imageForDrawKps);
    imshow("imageForDrawLine", imageForDrawLine);
    //imwrite(string("./line_detect_res/") + to_string(i)+".bmp",imageForDrawLine);
}

bool LineCornersAPI::isOnLineSeg(const Point2f &startPoint, const Point2f &endPoint, const Point2f &inputPoint, float &response)
{
    Point2f errorPoint = endPoint - startPoint;
    float lineSegLength = sqrt(errorPoint.dot(errorPoint));
    errorPoint = inputPoint - startPoint;
    float len1 = sqrt(errorPoint.dot(errorPoint));
    errorPoint = inputPoint - endPoint;
    float len2 = sqrt(errorPoint.dot(errorPoint));
    if((len1+len2) > (lineSegLength+onLineSegTh))
        return false;

    float dis = len1+len2 - lineSegLength;
    if(dis > 1)
        response = 1/dis;
    else
        response = 1;
    return true;
}


