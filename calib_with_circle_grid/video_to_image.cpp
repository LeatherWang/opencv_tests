#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <string>
#include <chrono>
#include <fstream>

using namespace std;
using namespace cv;

void video2image(string video, string datasetPath)
{
    std::string imageFilePath = datasetPath + "data/";
    std::string imageTimeStampFileName = datasetPath + "data.csv";

    ofstream ofTimeStamp;
    ofTimeStamp.open(imageTimeStampFileName, ios::trunc);
    ofTimeStamp << std::fixed;
    if(!ofTimeStamp.is_open())
    {
        cout << "error, can not open: " << imageTimeStampFileName << endl;
        return;
    }

    VideoCapture capture(video);
    if(!capture.isOpened())
    {
        cerr << "Failed to open a video: " << video <<endl;
        return;
    }

    long totalFrameNumber = capture.get(CV_CAP_PROP_FRAME_COUNT);
    cout << "total frames is:" << totalFrameNumber << "." << endl;

    long frameToStart = 1;
    capture.set(CV_CAP_PROP_POS_FRAMES, frameToStart);
    cout << "from" << frameToStart << "read" << endl;

    double rate = capture.get(CV_CAP_PROP_FPS);
    cout << "rate is:" << rate << endl;

    double time_interval = 1.0f / rate;
    std::chrono::nanoseconds timeStamp_ns = std::chrono::duration_cast< std::chrono::nanoseconds >(
            std::chrono::system_clock::now().time_since_epoch()
        );
    double timeStampBase = double(timeStamp_ns.count())/1e9;

    uint frameCounter = 0;
    Mat frame;
    while(1)
    {
        capture >> frame;
        if(frame.empty())
            break;
        uint64 stamp_ = (timeStampBase + frameCounter*time_interval)*1e9;
        string filename = imageFilePath + to_string(stamp_) + ".png";

        ofTimeStamp << stamp_ << endl;
        imwrite(filename, frame);
        if(frame.empty())
            cout << "error" << endl;

        imshow("Extractedframe", frame);
//        waitKey();

        frameCounter++;
        cerr << "\rframe counter: " << frameCounter;
    }
    cerr << endl;
    capture.release();

    cout << "Finished! \n" << endl;
}

int main(int argc,char** argv)
{
    if(argc < 3)
    {
        cout << "error, usage: " << endl;
        return 0;
    }

    string videoFromfile = argv[1];
    string datasetPath = argv[2];
    if(datasetPath.empty())
    {
        cout << "error, datasetPath: " << datasetPath << endl;
        return 0;
    }

    if(datasetPath.back() != '/')
        datasetPath.push_back('/');

    video2image(videoFromfile, datasetPath);


    return 0;
}











