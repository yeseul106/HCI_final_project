#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
    VideoCapture cap(0); // real-time camera capture
    //VideoCapture cap("highway.mp4");
    if (!cap.isOpened()) { cout << "file not found" << endl; return -1; }

    //namedWindow("frame", 1);
    while (1)
    {
        Mat frame;
        cap >> frame;
        if (frame.empty())
        {
            cout << "Video over" << endl;
            break;
        }
        cvtColor(frame, frame, COLOR_BGR2GRAY);

        Mat dst, cdst;
        Canny(frame, dst, 100, 200);       // edge detection
        imshow("edge", dst);
        cvtColor(dst, cdst, COLOR_GRAY2BGR);

        vector<Vec4i> lines;
        HoughLinesP(dst, lines, 1, CV_PI / 180, 50, 100, 20);
        for (size_t i = 0; i < lines.size(); i++) {
            Vec4i l = lines[i];
            line(cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, LINE_AA);
        }  //칼라 영상 위에 red 라인 그리기


        imshow("source", frame);
        imshow("detected lines", cdst);

        if (waitKey(30) >= 0) break;
    }
    return 0;
}
