
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/aruco/charuco.hpp>


#include <iostream>
#include <string>
#include <algorithm>

using namespace cv;
using namespace std;


//"{@outfile |<none> | Output image }"
//"{d        |       | dictionary: DICT_4X4_50=0, DICT_4X4_100=1, DICT_4X4_250=2,"
//"DICT_4X4_1000=3, DICT_5X5_50=4, DICT_5X5_100=5, DICT_5X5_250=6, DICT_5X5_1000=7, "
//"DICT_6X6_50=8, DICT_6X6_100=9, DICT_6X6_250=10, DICT_6X6_1000=11, DICT_7X7_50=12,"
//"DICT_7X7_100=13, DICT_7X7_250=14, DICT_7X7_1000=15, DICT_ARUCO_ORIGINAL = 16}"
//"{id       |       | Marker id in the dictionary }"
//"{ms       | 1000  | Marker size in pixels }"
//"{bb       | 1     | Number of bits in marker borders }"
//"{si       | false | show generated image }";

int main(int argc, char** argv)
{

    Ptr<aruco::Dictionary> dictionary =
        aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(cv::aruco::DICT_4X4_50));
    cv::Ptr<cv::aruco::CharucoBoard> board = cv::aruco::CharucoBoard::create(5, 7, 0.04, 0.02, dictionary);

    cv::Mat boardImage;
    board->draw( cv::Size(500, 700), boardImage, 20, 1 );

    imwrite("charuco.jpg",boardImage);
    imshow("boardImage", boardImage);
    waitKey();

    return 0;
}
