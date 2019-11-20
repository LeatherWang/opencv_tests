

#include <iostream>
#include <opencv2/opencv.hpp>
#include <iomanip>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

using namespace cv;
using namespace std;

double duration_ms, start;
bool verbose = true;
void plotHistgram(const cv::Mat &srcImage, const string imageWinName)
{
    int height = 150;	//直方图高度
    int scale = 2;	//垂直缩放比
    int horvizon_scale = 3;	//水平缩放比

    MatND dstHist;//得到的直方图
    int dims = 1;//得到的直方图的维数 灰度图的维数为1
    float hranges[2] = { 1, 255 };  //直方图统计的灰度值范围
    const float *ranges[1] = { hranges };   // 这里需要为const类型，二维数组用来指出每个区间的范围
    int bin = 255;//直方图横坐标的区间数 即横坐标被分成多少份
    int channels = 0;//图像得通道 灰度图的通道数为0
    calcHist(&srcImage, 1, &channels, Mat(), dstHist, dims, &bin, ranges);

    double minValue = 0;
    double maxValue = 0;
    minMaxLoc(dstHist, &minValue, &maxValue, 0, 0); //找到直方图中的最大值和最小值

    int shift_vertical = 13;	//直方图偏移值,偏移用于显示水平坐标
    int shift_horvizon = 30;	//直方图偏移值,偏移用于显示垂直坐标
    //绘制出直方图
    cv::Mat dstImage(height*scale, bin*horvizon_scale + shift_horvizon,
                     CV_8UC3, cv::Scalar(0, 0, 0));		//创建一个彩色三通道矩阵,大小a*b,填充0
    int hpt = cv::saturate_cast<int>((dstImage.rows - shift_vertical)*0.95); //最大值对应的Y坐标,防止溢出
    for (int i = 0; i < bin; i++)
    {
        float binValue = dstHist.at<float>(i);
        int realValue = cv::saturate_cast<int>(binValue * hpt / maxValue);
        cv::rectangle(dstImage,
                      cv::Point(i*horvizon_scale + shift_horvizon, dstImage.rows - 1 - shift_vertical),
                      cv::Point((i + 1)*horvizon_scale + shift_horvizon - 1, dstImage.rows - realValue - shift_vertical),
                      cv::Scalar(255, 255, 255), 1, 8, 0);
    }

    //绘制垂直刻度
    string str;
    CvFont font;
    double font_size = 1;//字体大小
    cvInitFont(&font, CV_FONT_HERSHEY_PLAIN, 1, 1, 0, 1, 8);//字体结构初始化
    cv::Size text_size;
    for (int i = hpt; i>=0; )
    {
        str = std::to_string(maxValue*i/hpt);
        str = str.substr(0, 5);
        //在图像中显示文本字符串
        text_size = cv::getTextSize(str, CV_FONT_HERSHEY_PLAIN, font_size, 1, NULL);	//获得字体大小
        cv::putText(dstImage, str,
                    cvPoint(0, dstImage.rows-i-shift_vertical + text_size.height/2),
                    cv::FONT_HERSHEY_PLAIN, font_size, cv::Scalar(0, 255, 0), 1, 8, 0);
        i -= hpt / 10;	//只显示10个刻度
    }

    //刻画水平刻度
    for (int i = bin; i >= 0;)
    {
        str = std::to_string(i);
        //在图像中显示文本字符串
        text_size = cv::getTextSize(str, CV_FONT_HERSHEY_PLAIN, font_size, 1, NULL);	//获得字体大小
        putText(dstImage, str,
                cvPoint(i*horvizon_scale + shift_horvizon - text_size.width/2, dstImage.rows),
                cv::FONT_HERSHEY_PLAIN, font_size, cv::Scalar(0, 0, 255), 1, 8, 0);
        i -= bin / 20;	//只显示20个刻度
    }
    //显示统计信息
    char strChar[100];
    sprintf(strChar, "bin=%d  Ranges from %d to %d", bin, (int)hranges[0], (int)hranges[1]);
    cv::putText(dstImage, strChar,
                cvPoint(dstImage.cols/5, 30), cv::FONT_HERSHEY_PLAIN,
                (double)1.3, cv::Scalar(255, 0, 0), 1, 8, 0);
    cv::imshow(imageWinName, dstImage);
}


int main(int argc, char **argv)
{
    string inputDir = string(argv[1]);
    if (!boost::filesystem::exists(inputDir) && !boost::filesystem::is_directory(inputDir))
    {
        std::cerr << "# ERROR: Cannot find input directory " << inputDir << "." << std::endl;
        return 1;
    }
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




    for(int i=0; i<imageFilenames.size(); i++)
    {
        cv::Mat imageRaw = cv::imread(imageFilenames[i], -1);

        cv::imshow("imageRaw", imageRaw);
        plotHistgram(imageRaw, "imageRaw graph");


        if(0)
        {
            if(imageRaw.channels() == 3)
                cvtColor(imageRaw, imageRaw, CV_BGR2GRAY);
            cv::Mat boostContrastImage;
            double a = 1.4;
            double b = 80.0f*(1.0f-a); //这里的b与亮度没有任何关系，仅仅用于对比度调整，经过(100,100)这个点
            imageRaw.convertTo(boostContrastImage, imageRaw.type(), a, b);
            cv::imshow("boostContrastImage", boostContrastImage);
            plotHistgram(boostContrastImage, "boostContrastImage graph");
        }

        if(0)
        {
            Mat imageEnhance;

            Mat kernel = (Mat_<float>(3, 3) << -1, -1, -1, -1, 9, -1, -1, -1, -1);
            filter2D(imageRaw, imageEnhance, CV_8UC1, kernel);
            cv::imshow("imageEnhance1", imageEnhance);

            kernel = (Mat_<float>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
            filter2D(imageRaw, imageEnhance, CV_8UC1, kernel);
            cv::imshow("imageEnhance2", imageEnhance);
        }

        if(0)
        {
            cv::Mat bilImg, medianImage;
            cv::bilateralFilter(imageRaw, bilImg, 5, 30, 30);
            cv::medianBlur(imageRaw, medianImage, 5);
            imshow("bilateralFilter", bilImg);
            imshow("medianImage", medianImage);
        }


        // 直方图均衡化
        if(0)
        {
            if(imageRaw.channels() == 3)
                cvtColor(imageRaw, imageRaw, CV_BGR2GRAY);
            cv::Mat imageClaheHist, imageEqualizeHist;
            cv::Mat gaussBlurImage;
            cv::GaussianBlur(imageRaw, gaussBlurImage, cv::Size(1,1), 0, 0, BORDER_DEFAULT);

            start = double(getTickCount());
            cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
            clahe->apply(gaussBlurImage, imageClaheHist);
            if(verbose)
            {
                duration_ms = (double(getTickCount()) - start) * 1000 / getTickFrequency();
                std::cout << "   It took " << duration_ms << " ms" << std::endl;
            }

            start = double(getTickCount());
            equalizeHist(gaussBlurImage, imageEqualizeHist);
            if(verbose)
            {
                duration_ms = (double(getTickCount()) - start) * 1000 / getTickFrequency();
                std::cout << "   It took " << duration_ms << " ms" << std::endl;
            }


            cv::imshow("claheHist", imageClaheHist);
            cv::imshow("equalizeHist", imageEqualizeHist);
        }

        // 提升亮度，不改变色度和饱和度
        if(1)
        {
            Mat imageXYZ;
            if(imageRaw.channels() == 1)
                cvtColor(imageRaw, imageRaw, CV_GRAY2BGR);
            cv::cvtColor(imageRaw, imageXYZ, CV_BGR2HSV);

            cv::Mat imChs[3];
            cv::split(imageXYZ, imChs);

//            for (int i = 0; i < imChs[2].rows; i++)
//            {
//                uchar* pData = imChs[2].ptr<uchar>(i);
//                for (int j = 0; j < imChs[2].cols; j++)
//                {
//                    int value = pData[j]*1.2;
//                    pData[j] = value>255?255:value;
//                }
//            }

            // 使用卷积运算代替上面的操作
            cv::Mat kernel = (cv::Mat_<float>(1,1)<<1.2);
            cv::filter2D(imChs[2], imChs[2], CV_8UC1, kernel);

            cv::Mat imBoostBright;
            merge(imChs, 3, imBoostBright);
            cv::cvtColor(imBoostBright, imBoostBright, CV_HSV2BGR);
            cv::imshow("image boost bright", imBoostBright);
        }


        // gamma校正
        if(0)
        {
            if(imageRaw.channels() == 1)
                cvtColor(imageRaw, imageRaw, CV_GRAY2BGR);
            double gamma_ = 1.4;
            Mat lookUpTable(1, 256, CV_8U);
            uchar * p = lookUpTable.ptr();
            for(int i = 0; i <256; ++ i)
                p [i] = saturate_cast <uchar>(pow(i/255.0, gamma_)* 255.0);

            Mat imageGamma = imageRaw.clone();
            if(imageRaw.channels() == 1)
            {
                LUT(imageRaw, lookUpTable, imageGamma);
            }
            else
            {
                Mat imageGammaChs[3];
                Mat imageRawChs[3];
                split(imageRaw, imageRawChs);
                for(int i=0; i<3; i++)
                {
                    imageGammaChs[i] = Mat(imageRaw.size(), CV_8UC1);
                    LUT(imageRawChs[i], lookUpTable, imageGammaChs[i]);
                }
                merge(imageGammaChs, 3, imageGamma);
                cv::imshow("gamma_no_norm", imageGamma);
                normalize(imageGamma, imageGamma, 0, 255, CV_MINMAX); //归一化，增加对比度
            }
            cv::imshow("gamma", imageGamma);
        }

        // 去掉一个通道
        //    if(imageRaw.channels() == 1)
        //        cv::cvtColor(imageRaw, imageRaw, CV_GRAY2BGR);
        //    cv::Mat imageChanl[3];
        //    cv::split(imageRaw, imageChanl);
        //    imageChanl[1] = cv::Mat(imageChanl[1].size(), CV_8UC1, Scalar::all(0));
        //    cv::merge(imageChanl, 3, imageRaw);
        //    cv::imshow("image", imageRaw);


        cv::waitKey();
    }
    return 0;
}
