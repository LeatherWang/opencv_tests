#include <iostream>  
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>    
#include <opencv2/core/core.hpp>
#include <string>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

using namespace cv;
using namespace std;

Scalar low_range(60, 100, 80); //!@todo
Scalar high_range(95, 255, 180);

struct BlobDetector
{
    int H_low = 60;
    int H_high = 100;
    int S_low = 100;
    int S_high = 255;
    int V_low = 60;
    int V_high = 180;
} blobDetector;

void chooseBolbCB( int, void* );
Mat gSrcImage;
string boldsPicWinName = "boldsPic";
int main(int argc, char **argv)
{
    if(argc < 2)
    {
        cout<<"Usage error!!"<<endl;
        return 0;
    }


    string inputDir = string(argv[1]);
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

    namedWindow( boldsPicWinName, CV_WINDOW_AUTOSIZE );
    createTrackbar( "H_high", boldsPicWinName, &blobDetector.H_high, 180,
                    chooseBolbCB );
    createTrackbar( "H_low", boldsPicWinName, &blobDetector.H_low, blobDetector.H_high,
                    chooseBolbCB );
    createTrackbar( "S_high", boldsPicWinName, &blobDetector.S_high, 255,
                    chooseBolbCB );
    createTrackbar( "S_low", boldsPicWinName, &blobDetector.S_low, blobDetector.S_high,
                    chooseBolbCB );
    createTrackbar( "V_high", boldsPicWinName, &blobDetector.V_high, 255,
                    chooseBolbCB );
    createTrackbar( "V_low", boldsPicWinName, &blobDetector.V_low, blobDetector.V_high,
                    chooseBolbCB );

    int size = imageFilenames.size();
    for(int i=0; i<size; i++) //300
    {
        Mat imageRaw = imread(imageFilenames[i], IMREAD_COLOR);
        imageRaw.copyTo(gSrcImage);
        chooseBolbCB( 0, 0 );
        cv::imshow("origin", imageRaw);
        waitKey();
    }
    return 0;
}

void chooseBolbCB( int, void* )
{
    cv::Mat image_bgr;
    image_bgr = gSrcImage.clone();

    // 色调（H），饱和度（S），明度（V）
    Scalar low_range(blobDetector.H_low, blobDetector.S_low, blobDetector.V_low); //!@todo
    Scalar high_range(blobDetector.H_high, blobDetector.S_high, blobDetector.V_high);
    Mat image_hsv, image_hsv_to_bgr, dst;
    cv::cvtColor(image_bgr, image_hsv, CV_BGR2HSV);
    cv::inRange(image_hsv, low_range, high_range, dst); //dst是二进制
    cv::cvtColor(image_hsv, image_hsv_to_bgr, CV_HSV2BGR);

    cv::imshow("before_morphology", dst);
    // 形态学去噪
    Mat element = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    morphologyEx(dst, dst, MORPH_OPEN, element);

//    cv::imshow("dst", dst);
//    waitKey();

    morphologyEx(dst, dst, MORPH_CLOSE, element);

//    cv::imshow("dst", dst);
//    waitKey();

    imshow( boldsPicWinName, dst );
}

