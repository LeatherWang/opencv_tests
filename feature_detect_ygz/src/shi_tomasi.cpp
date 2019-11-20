
#include "read_intrinsic.h"
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include "shi_tomasi_api.h"

void cornerShiTomasi_demo( int, void* );
Mat gSrcImage;
string shiTomasiWinName = "shiTomasi";
int qualityLevel = 10;
int main(int argc, char **argv)
{
    if(argc < 2)
    {
        cout<<"Usage error!!"<<endl;
        return 0;
    }
    string file_path = string(argv[1]) + "/camera.yaml"; // camera_fisheye_ankobot
    cout<<"intrinsic file: "<<file_path<<endl;
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

    namedWindow( shiTomasiWinName, CV_WINDOW_AUTOSIZE );
    createTrackbar( "max corner", shiTomasiWinName, &featureTrackParamUsingMask.maxCorners, 1000,
                    cornerShiTomasi_demo );
    createTrackbar( "blockSize", shiTomasiWinName, &featureTrackParamUsingMask.blockSize, 30,
                    cornerShiTomasi_demo );
    createTrackbar( "qualityLevel", shiTomasiWinName, &qualityLevel, 100,
                    cornerShiTomasi_demo );
    createTrackbar( "minDistance", shiTomasiWinName, &featureTrackParamUsingMask.minDistance, 50,
                    cornerShiTomasi_demo );

    int size = imageFilenames.size();
    for(int i=0; i<size; i++) //300
    {
        Mat imageRaw = imread(imageFilenames[i], IMREAD_GRAYSCALE);
        cv::remap(imageRaw, imageRaw, intrinsicAndUndistort.mapx, intrinsicAndUndistort.mapy, cv::INTER_LINEAR);



        //            fastNlMeansDenoising(srcImg1, srcImg1);
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(imageRaw, imageRaw);

        imageRaw.copyTo(gSrcImage);

        cornerShiTomasi_demo( 0, 0 );

        waitKey();
    }
    return 0;
}

void cornerShiTomasi_demo( int, void* )
{
    if( featureTrackParamUsingMask.maxCorners < 1 ) featureTrackParamUsingMask.maxCorners = 1;
    vector<KeyPoint> vecKeyPoints;
    if(qualityLevel > 0)
        featureTrackParamUsingMask.qualityLevel = qualityLevel/100.0f;


    computeShiTomasiCornorUsingMask(gSrcImage, vecKeyPoints, "gSrcImage"); //内部会进行均值滤波
    Mat srcImageForDrawKp = gSrcImage.clone();
    if(srcImageForDrawKp.channels() == 1)
        cvtColor(srcImageForDrawKp, srcImageForDrawKp, CV_GRAY2BGR);
    for(auto &ele:vecKeyPoints)
        circle( srcImageForDrawKp, ele.pt, 4, Scalar(0,255,0), 2, 8, 0 );

    putText(srcImageForDrawKp, string("kpt: ")+to_string(vecKeyPoints.size()),
            Point(srcImageForDrawKp.size().width/2, 20), cv::FONT_HERSHEY_PLAIN, 1.2, Scalar(255,0,255));

    imshow( shiTomasiWinName, srcImageForDrawKp );
}
