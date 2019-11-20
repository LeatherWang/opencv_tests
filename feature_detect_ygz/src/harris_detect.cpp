
#include "read_intrinsic.h"

void on_CornerHarris( int, void* );
string WINDOW_NAME="win_to_adjust_th";
Mat g_srcImage;
int thresh = 30;
int max_thresh = 175;
int main(int argc, char **argv)
{
    string file_path = "./camera.yaml"; // camera_fisheye_ankobot
    cout<<"intrinsic file: "<<file_path<<endl;
    if(!readIntrinsic(file_path, intrinsicAndUndistort.K_Mat,
                      intrinsicAndUndistort.DistCoef, intrinsicAndUndistort.imageSize))
        return 1;

    std::string imageFileName = "./3.jpg";
    if(argc>1)
        imageFileName = argv[1];
    cv::Mat imageRaw = cv::imread(imageFileName, cv::IMREAD_GRAYSCALE);

    cv::remap(imageRaw, imageRaw, intrinsicAndUndistort.mapx,
              intrinsicAndUndistort.mapy, cv::INTER_LINEAR);

    //        fastNlMeansDenoising(srcImg1, srcImg1);
    //        fastNlMeansDenoising(srcImg2, srcImg2);

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    clahe->apply(imageRaw, imageRaw);

    if(imageRaw.empty())
    {
        cout<< "error to open image!" << endl;
        return 0;
    }

    imageRaw.copyTo(g_srcImage);

    namedWindow( WINDOW_NAME, CV_WINDOW_AUTOSIZE );
    createTrackbar( "thresh adjust", WINDOW_NAME, &thresh, max_thresh, on_CornerHarris );
    on_CornerHarris( 0, 0 );

    waitKey();
    return 0;
}

void on_CornerHarris( int, void* )
{
    Mat srcImageGray = g_srcImage.clone();
    if(srcImageGray.channels() == 3)
        cvtColor(srcImageGray, srcImageGray, CV_BGR2GRAY);

    cv::Mat dstImage = Mat::zeros( srcImageGray.size(), CV_32FC1 );
    cornerHarris(srcImageGray, dstImage, 7, 3, 0.04, BORDER_DEFAULT );

    Mat normImage, scaledImage;
    normalize(dstImage, normImage, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
    convertScaleAbs(normImage, scaledImage);

    Mat srcImage = g_srcImage.clone();
    if(srcImage.channels() == 1)
        cvtColor(srcImage, srcImage, CV_GRAY2BGR);
    for( int j = 0; j < normImage.rows ; j++ )
    { for( int i = 0; i < normImage.cols; i++ )
        {
            if( (int) normImage.at<float>(j,i) > thresh+100 )
            {
                circle(srcImage, Point( i, j ), 2,  Scalar(10,10,255), 1, 8, 0 );
                circle(scaledImage, Point( i, j ), 2,  Scalar(0,10,255), 1, 8, 0 );
            }
        }
    }

    imshow(WINDOW_NAME, srcImage );
    imshow("scaledImage", scaledImage );
}








