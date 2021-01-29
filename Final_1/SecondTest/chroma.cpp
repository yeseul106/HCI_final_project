#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include <cstdio>
#include <iostream>
#pragma warning (disable:4996)

using namespace cv;
using namespace std;

int value = 0; int s = 0; int h = 0;
void trackbar(int, void*) {}

int main()
{
	//namedWindow("img_result");
	//createTrackbar("s", "img_result", &s, 255, trackbar);
	//createTrackbar("value", "img_result", &value, 255, trackbar);
	//createTrackbar("h", "img_result", &h, 255, trackbar);
	VideoCapture cap(0);
	if (!cap.isOpened()) { cout << "file not found" << endl; }

	while (1)
	{
		Mat imgHSV;
		Mat frame;
		cap >> frame;
		cvtColor(frame, imgHSV, COLOR_BGR2HSV);

		Mat imgThresholded;
		inRange(imgHSV, Scalar(79, 158, 55), Scalar(150, 255, 255), imgThresholded);

		Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
		//erode(imgThresholded, imgThresholded, element);
		morphologyEx(imgThresholded, imgThresholded, MORPH_OPEN, element);

		/*픽셀 하나하나 접근해서 그리는 코드*/
		int x1 = frame.cols; int x2 = 0; int y1 = frame.rows;  int y2 = 0;
		for (int i = 1; i < frame.rows - 1; i++) {
			for (int j = 1; j < frame.cols - 1; j++) {
				if (imgThresholded.at<uchar>(i, j) == 255 && imgThresholded.at<uchar>(i - 1, j) == 255 && imgThresholded.at<uchar>(i + 1, j) == 255 &&
					imgThresholded.at<uchar>(i, j + 1) == 255 && imgThresholded.at<uchar>(i, j - 1) == 255)
				{
					if (y1 > i)
						y1 = i;
					if (y2 < i)
						y2 = i;
				}
			}
		}
		for (int i = 1; i < frame.cols - 1; i++) {
			for (int j = 1; j < frame.rows - 1; j++) {
				if (imgThresholded.at<uchar>(j, i) == 255 && imgThresholded.at<uchar>(j - 1, i) == 255 && imgThresholded.at<uchar>(j + 1, i) == 255 &&
					imgThresholded.at<uchar>(j, i + 1) == 255 && imgThresholded.at<uchar>(j, i - 1) == 255)
				{

					if (x1 > i)
						x1 = i;
					if (x2 < i)
						x2 = i;
				}
			}
		}
		Mat src = imread("cute.jpg", IMREAD_COLOR);

		if ((x2 - x1) > 0 && (y2 - y1) > 0) {
			rectangle(frame, Point(x1, y1), Point(x2, y2), Scalar(0, 0, 255), 2);
			//printf("x좌표 : (%d %d) y좌표: (%d %d)", x1, x2, y1, y2);
			resize(src, src, Size(x2 - x1, y2 - y1), INTER_AREA);
			Mat roi(frame, Rect(x1, y1, x2 - x1, y2 - y1));
			src.copyTo(roi);

		}

		imshow("cap", frame);
		imshow("dst", imgThresholded);
		if (waitKey(30) >= 0) break;
	}

	waitKey(0);
	return 0;
}

