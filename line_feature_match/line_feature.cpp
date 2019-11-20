#include <iostream>
#include <chrono>
#include <cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/line_descriptor/descriptor.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>

using namespace cv;
using namespace std;
using namespace cv::line_descriptor;
struct sort_descriptor_by_queryIdx
{
    inline bool operator()(const vector<DMatch>& a, const vector<DMatch>& b){
        return ( a[0].queryIdx < b[0].queryIdx );
    }
};
struct sort_lines_by_response
{
    inline bool operator()(const KeyLine& a, const KeyLine& b){
        return ( a.response > b.response );
    }
};
void ExtractLineSegment(const Mat &img, const Mat &image2, vector<KeyLine> &keylines,vector<KeyLine> &keylines2);
int main(int argc, char**argv)
{
    if(argc != 3)
    {
        cerr << endl << "Usage: ./Line path_to_image1 path_to_image2" << endl;
        return 1;
    }
    string imagePath1=string(argv[1]);
    string imagePath2=string(argv[2]);
    cout<<"import two images"<<endl;
    Mat image1=imread(imagePath1);
    Mat image2=imread(imagePath2);

    imshow("ima1",image1);
    imshow("ima2",image2);
    waitKey(0);
    if(image1.data==NULL)
    {
        cout<<"the path is wrong"<<endl;
    }

    vector<KeyLine> keylines,keylines2;


    ExtractLineSegment(image1,image2,keylines,keylines2);

    return 0;

}
void ExtractLineSegment(const Mat &img, const Mat &image2, vector<KeyLine> &keylines,vector<KeyLine> &keylines2)
{
    Mat mLdesc,mLdesc2;

    vector<vector<DMatch>> lmatches;

    Ptr<BinaryDescriptor> lbd = BinaryDescriptor::createBinaryDescriptor();
    Ptr<line_descriptor::LSDDetector> lsd = line_descriptor::LSDDetector::createLSDDetector();

    cout<<"extract lsd line segments"<<endl;
    lsd->detect(img, keylines, 1.2,1);
    Mat imageAddedLine;
    drawKeylines(img, keylines, imageAddedLine, Scalar::all( -1 ));
    imshow( "imageAddedLine", imageAddedLine );
    waitKey();

    lsd->detect(image2,keylines2,1.2,1);
    int lsdNFeatures = 50;
    cout<<"filter lines"<<endl;
    if(keylines.size()>lsdNFeatures)
    {
        sort(keylines.begin(), keylines.end(), sort_lines_by_response());
        keylines.resize(lsdNFeatures);
        for( int i=0; i<lsdNFeatures; i++)
            keylines[i].class_id = i;
    }
    if(keylines2.size()>lsdNFeatures)
    {
        sort(keylines2.begin(), keylines2.end(), sort_lines_by_response());
        keylines2.resize(lsdNFeatures);
        for(int i=0; i<lsdNFeatures; i++)
            keylines2[i].class_id = i;
    }

    cout<<"lbd describle"<<endl;
    lbd->compute(img, keylines, mLdesc);
    lbd->compute(image2,keylines2,mLdesc2);//计算特征线段的描述子
    BFMatcher* bfm = new BFMatcher(NORM_HAMMING, false);
    bfm->knnMatch(mLdesc, mLdesc2, lmatches, 2);
    vector<DMatch> matches;
    for(size_t i=0;i<lmatches.size();i++)
    {
        const DMatch& bestMatch = lmatches[i][0];
        const DMatch& betterMatch = lmatches[i][1];
        float  distanceRatio = bestMatch.distance / betterMatch.distance;
        if (distanceRatio < 0.75)
            matches.push_back(bestMatch);
    }

    cv::Mat outImg;
    std::vector<char> mask( lmatches.size(), 1 );
    drawLineMatches( img, keylines, image2, keylines2, matches, outImg, Scalar::all( -1 ), Scalar::all( -1 ), mask,
                     DrawLinesMatchesFlags::DEFAULT );

    imshow( "Matches", outImg );
    waitKey();

}
