#include <iostream>  
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>    
#include <opencv2/core/core.hpp>
#include <string>

using namespace std;
using namespace cv;

cv::Size imageSize;
cv::Mat K_left, D_left;

bool LoadMyntFisheyeParam_1(const string &file_path, cv::Mat &mapx_left, cv::Mat &mapy_left)
{
    cv::FileStorage fsSettings(file_path, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }


    fsSettings["leftIntrinsic"] >> K_left;
    fsSettings["leftdistortion"] >> D_left;

    fsSettings["image_width"] >> imageSize.width;
    fsSettings["image_height"] >> imageSize.height;

    if(K_left.empty() || D_left.empty() ||
            imageSize.width==0 || imageSize.height==0)
    {
        cerr << "there is empty intrinsic!!" << endl;
        exit(-1);
    }

    if(K_left.type() != CV_64F)
        K_left.convertTo(K_left, CV_64F);
    if(D_left.type() != CV_64F)
        D_left.convertTo(D_left, CV_64F);

    return true;
}


bool LoadMyntFisheyeParam_2(const string &file_path, cv::Mat &mapx_left, cv::Mat &mapy_left)
{
    cv::FileStorage fs(file_path, cv::FileStorage::READ);
    if (!fs.isOpened())
        return false;
    cv::Mat K_ = cv::Mat::eye(3, 3, CV_64F);;
    cv::Mat D_(4,1,CV_64F);

    fs["D_fisheye"] >> D_;
    fs["K_old"] >> K_;

    imageSize.height = fs["image.height"];
    imageSize.width = fs["image.width"];

    if(K_.empty() || D_.empty() ||
            imageSize.width==0 || imageSize.height==0)
    {
        cerr << "there is empty intrinsic!!" << endl;
        exit(-1);
    }

    K_left = K_.clone();
    D_left = D_.clone();

    return true;
}


void estimateNewCameraMatrixForUndistortRectify(InputArray K, InputArray D, const Size &image_size, InputArray R,
    OutputArray P, double balance, const Size& new_size, double fov_scale)
{
    CV_Assert( K.size() == Size(3, 3)       && (K.depth() == CV_32F || K.depth() == CV_64F));
    CV_Assert(D.empty() || ((D.total() == 4) && (D.depth() == CV_32F || D.depth() == CV_64F)));

    int w = image_size.width, h = image_size.height;
    balance = std::min(std::max(balance, 0.0), 1.0);

    cv::Mat points(1, 4, CV_64FC2);
    Vec2d* pptr = points.ptr<Vec2d>();

    pptr[0] = Vec2d(w/2, 40);
    pptr[1] = Vec2d(w-41, h/2);
    pptr[2] = Vec2d(w/2, h-41);
    pptr[3] = Vec2d(40, h/2);

    fisheye::undistortPoints(points, points, K, D, R);
    cv::Scalar center_mass = mean(points);

    std::cout << center_mass[0] << ", " << center_mass[1] << std::endl;

    cv::Vec2d cn(center_mass.val);

    double aspect_ratio = (K.depth() == CV_32F) ? K.getMat().at<float >(0,0)/K.getMat().at<float> (1,1)
                                                : K.getMat().at<double>(0,0)/K.getMat().at<double>(1,1);

    // convert to identity ratio
    cn[0] *= aspect_ratio;
    for(size_t i = 0; i < points.total(); ++i)
        pptr[i][1] *= aspect_ratio;

    double minx = DBL_MAX, miny = DBL_MAX, maxx = -DBL_MAX, maxy = -DBL_MAX;
    for(size_t i = 0; i < points.total(); ++i)
    {
        miny = std::min(miny, pptr[i][1]); //pptr在556行转换成了归一化坐标系上了
        maxy = std::max(maxy, pptr[i][1]);
        minx = std::min(minx, pptr[i][0]);
        maxx = std::max(maxx, pptr[i][0]);
    }

    double f1 = w * 0.5/(cn[0] - minx);
    double f2 = w * 0.5/(maxx - cn[0]);
    double f3 = h * 0.5 * aspect_ratio/(cn[1] - miny);
    double f4 = h * 0.5 * aspect_ratio/(maxy - cn[1]);

    double fmin = std::min(f1, std::min(f2, std::min(f3, f4)));
    double fmax = std::max(f1, std::max(f2, std::max(f3, f4)));

    double f = balance * fmin + (1.0 - balance) * fmax;
    f *= fov_scale > 0 ? 1.0/fov_scale : 1.0;

    cv::Vec2d new_f(f, f), new_c = -cn * f + Vec2d(w, h * aspect_ratio) * 0.5;

    // restore aspect ratio
    new_f[1] /= aspect_ratio;
    new_c[1] /= aspect_ratio;

    if (new_size.area() > 0)
    {
        double rx = new_size.width /(double)image_size.width;
        double ry = new_size.height/(double)image_size.height;

        new_f[0] *= rx;  new_f[1] *= ry;
        new_c[0] *= rx;  new_c[1] *= ry;
    }

    Mat(Matx33d(new_f[0], 0, new_c[0],
                0, new_f[1], new_c[1],
                0,        0,       1)).convertTo(P, P.empty() ? K.type() : P.type());
}

int main(int argc, char** argv)
{
    if(argc < 3)
    {
        cout << "error" << endl;
        return 0;
    }

    Mat image_raw = imread(argv[1], IMREAD_GRAYSCALE);
    string intrinsicFile = argv[2];


    cv::Mat mapx_left, mapy_left;
//    LoadMyntFisheyeParam_1(intrinsicFile, mapx_left, mapy_left);
    LoadMyntFisheyeParam_2(intrinsicFile, mapx_left, mapy_left);

    cout << "K:\n" << K_left << endl;
    cout << "D:\n" << D_left << endl << endl;



    double radio = 1.4f;
    cv::Mat R = Mat::eye(3,3,CV_64F);
    for(int i=0; i<20; i++)
    {
        Mat K_new;
#if 1
        double fov_scale = radio;
        estimateNewCameraMatrixForUndistortRectify(K_left, D_left, imageSize, R, K_new, 0, imageSize, fov_scale);

#else
        K_new = K_left / (radio*1.5);
        K_new.at<double>(2,2) = 1;
#endif
        cv::fisheye::initUndistortRectifyMap(K_left, D_left, R, K_new, imageSize, CV_32FC1, mapx_left, mapy_left);
        cout << "K_new: " << K_new << endl << endl;

        Mat image_processed;
        cv::remap(image_raw, image_processed, mapx_left, mapy_left, cv::INTER_LINEAR);
        string strName = "K_scale_with_" + to_string(radio);
        imshow(strName, image_processed);

        waitKey();

        imwrite(strName+".png", image_processed);

        radio -= 0.1;
    }


	return 0;
}

