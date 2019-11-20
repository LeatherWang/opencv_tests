
#include "read_intrinsic.h"
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include "oclGoodFeatureTracker.h"

void cornerSusanCompute( int, void* );
Mat gSrcImage;
string susanWinName = "susan";

struct SusanParam
{
    int t=25;
    int g=18;
} susanParam;

int OffSetX[37] =
{
    -1, 0, 1,
    -2, -1, 0, 1, 2,
    -3, -2, -1, 0, 1, 2, 3,
    -3, -2, -1, 0, 1, 2, 3,
    -3, -2, -1, 0, 1, 2, 3,
    -2, -1, 0, 1, 2,
    -1, 0, 1
};
int OffSetY[37] =
{
    -3, -3, -3,
    -2, -2, -2, -2, -2,
    -1, -1, -1, -1, -1, -1, -1,
    0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1,
    2, 2, 2, 2, 2,
    3, 3, 3
};

int main(int argc, char **argv)
{
    if(argc < 2)
    {
        cout<<"Usage error!!"<<endl;
        return 0;
    }
    string file_path = string(argv[1]) + "/camera.yaml"; // camera_fisheye_ankobot
    cout<<"intrinsic file: "<<file_path<<endl;
    //cv::Mat DistCoef_Zero = cv::Mat::zeros(4, 1, CV_64F);
    if(!readIntrinsic(file_path, intrinsicAndUndistort.K_Mat,
                      intrinsicAndUndistort.DistCoef, intrinsicAndUndistort.imageSize))
        return 1;

    string inputDir = string(argv[1])+"/cam0";
    std::string fileExtension = ".jpg";
    std::vector<std::string> imageFilenames;
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

    namedWindow( susanWinName, CV_WINDOW_AUTOSIZE );
    createTrackbar( "t", susanWinName, &susanParam.t, 50, cornerSusanCompute );
    createTrackbar( "g", susanWinName, &susanParam.g, 50, cornerSusanCompute );

    int size = imageFilenames.size();
    for(int i=0; i<size; i++) //300
    {
        Mat imageRaw = imread(imageFilenames[i], IMREAD_GRAYSCALE);
        cv::remap(imageRaw, imageRaw, intrinsicAndUndistort.mapx, intrinsicAndUndistort.mapy, cv::INTER_LINEAR);

        //            fastNlMeansDenoising(srcImg1, srcImg1);
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(imageRaw, imageRaw);

        GaussianBlur(imageRaw, imageRaw, Size(3,3), 0);

        imageRaw.copyTo(gSrcImage);

        cornerSusanCompute( 0, 0 );

        waitKey();
    }
    return 0;
}

void cornerSusanCompute( int, void* )
{
    Mat tmpImage = cv::Mat::zeros(gSrcImage.size(), gSrcImage.type());
    int height = gSrcImage.size().height;
    int width = gSrcImage.size().width;
    int same = 0;
    for (int i = 3; i < height - 3; i++)
    {
        for (int j = 3; j < width - 3; j++)
        {
            same = 0;
            uchar centerPixel = gSrcImage.at<uchar>(i, j);

            // 阈值t越高，接近的点就越多，就越不能分辨出边缘或者角点
            for (int k = 0; k < 37; k++)
                if(abs(gSrcImage.at<uchar>(i+OffSetY[k], j+OffSetX[k]) - centerPixel) < susanParam.t)
                    same++;

            // 阈值g越高，角点的质量越差
            if (same < susanParam.g)
                tmpImage.at<uchar>(i,j) = susanParam.g-same;
        }
    }

    //非极大值抑制
    Mat resImage = gSrcImage.clone();
    int i_s[8] = { -1, -1, -1, 0, 0, 1, 1, 1 };
    int j_s[8] = { -1, 0, 1, -1, 1, -1, 0, 1 };
    for (int i = 4; i < height - 4; i++)
    {
        for (int j = 4; j < width -4; j++)
        {
            if(tmpImage.at<uchar>(i,j) != 0)
            {
                int k = 0;
                for (; k < 8; k++)
                {
                    if (tmpImage.at<uchar>(i,j) < tmpImage.at<uchar>(i+i_s[k],j+j_s[k]))
                        break;
                }
                if (k == 8)
                    resImage.at<uchar>(i,j) = 255;
            }
        }
    }

    imshow("gSrcImage", gSrcImage);
    imshow(susanWinName, resImage);

}
