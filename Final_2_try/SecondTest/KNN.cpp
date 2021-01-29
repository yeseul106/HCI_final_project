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
	Mat img;
	img = imread("fashion.png", IMREAD_GRAYSCALE);
	namedWindow("original", WINDOW_AUTOSIZE);
	imshow("original", img);
	while (1) {
		int key = waitKey(100);
		string labelArray[10] = { "T-shirt","pants","shirt","dress","jacket","shoes","blouse","sneakers","bag","boots" };

		if (key == '1') {
			Mat train_features(900, 784, CV_32FC1);
			Mat labels(900, 1, CV_32FC1);
			// 각 숫자 영상을 row vector로 만들어서 train_features에 저장한다. 
			for (int r = 0; r < 30; r++) {     // 옷 이미지 세로로 30개
				for (int c = 0; c < 30; c++) { // 옷 이미지 가로로 30개
					int i = 0;
					for (int y = 0; y < 28; y++) { // 28x28
						for (int x = 0; x < 28; x++) {
							train_features.at<float>(r * 30 + c, i++)
								= img.at<uchar>(r * 28 + y, c * 28 + x);
						}
					}
				}
			}
			//imshow("feature", train_features.row(0));

			// 각 숫자 영상에 대한 레이블을 저장한다. 
			for (int i = 0; i < 900; i++) {  // 의상의 카테고리 10개가 각각 90개씩
				labels.at<float>(i, 0) = (i / 90);
			}

			// 학습 단계
			Ptr<ml::KNearest> knn = ml::KNearest::create();
			Ptr<ml::TrainData> trainData = ml::TrainData::create(train_features, ml::ROW_SAMPLE, labels);
			knn->train(trainData);

			// 테스트 단계
			Mat predictedLabels;
			float accuracy = 0;
			for (int i = 0; i < 900; i++) { // 의상 900개에 대해서
				Mat test = train_features.row(i); //train 데이터를 test 데이터로 사용
				knn->findNearest(test, 3, predictedLabels);  // k=3
				float prediction = predictedLabels.at<float>(0);

				if (predictedLabels.at<float>(0) == labels.at<float>(i, 0))
					accuracy++;
				//라벨 인덱스에 따른 의상 종류를 분류 (10종류)
				int indexprediction = prediction;
				cout << "테스트 샘플" << i + 1 << "의 라벨 = " << prediction << " => " << labelArray[indexprediction] << '\n';
				//cout << accuracy << "\n";
			}
			//정확도 구하기
			float result = (accuracy / 900.0) * 100;
			cout << "정확도: " << result << "%" << endl;
		}
		else if (key == '2') {

			//테스트 데이터 받기
			/*string fileName;
			cout << "파일명을 확장자까지 정확히 입력하세요.(종료를 원할 시 break입력!) : ";
			getline(cin, fileName);

			if (fileName.compare("break") == 0) {
				cout << "종료합니다..." << endl;
				break;
			}
			*/

			Mat img2;
			img2 = imread("fashion.png", IMREAD_GRAYSCALE);
			Mat train_features(900, 784, CV_32FC1);
			Mat labels(900, 1, CV_32FC1);
			// 각 숫자 영상을 row vector로 만들어서 train_features에 저장한다. 
			for (int r = 0; r < 30; r++) {     // 옷 이미지 세로로 30개 
				for (int c = 0; c < 30; c++) { // 옷 이미지 가로로 30개
					int i = 0;
					for (int y = 0; y < 28; y++) { // 28x28
						for (int x = 0; x < 28; x++) {
							train_features.at<float>(r * 30 + c, i++)
								= img2.at<uchar>(r * 28 + y, c * 28 + x);
						}
					}
				}
			}

			// 각 숫자 영상에 대한 레이블을 저장한다. 
			for (int i = 0; i < 900; i++) {  // 의상의 카테고리 10개가 각각 90개씩
				labels.at<float>(i, 0) = (i / 90);
			}

			// 학습 단계
			Ptr<ml::KNearest> knn = ml::KNearest::create();
			Ptr<ml::TrainData> trainData = ml::TrainData::create(train_features, ml::ROW_SAMPLE, labels);
			knn->train(trainData);

			//테스트 이미지 가져오기 (크기 28 X 28로 설정)
			Mat test_img = imread("pants.jpg", IMREAD_GRAYSCALE);
			Mat show_test;
			resize(test_img, show_test, Size(200, 200));
			imshow("show_test", show_test);
			resize(test_img, test_img, Size(28, 28));
			//imshow("testimg", test_img);

			/*Mat pog;
			Ptr<BackgroundSubtractor> pMOG2;
			pMOG2 = createBackgroundSubtractorMOG2();
			pMOG2->apply(test_img, pog);*/
			//전경분리하기 (교수님이 주시는 예제 돌려보쟈!)

			threshold(test_img, test_img, 0, 255, THRESH_OTSU | THRESH_BINARY_INV);


			imshow("testimg_threshold", test_img);
			test_img = test_img.reshape(0, 1);
			test_img.convertTo(test_img, CV_32FC1);
			//imshow("f", test_img);

			//테스트 데이터 1차원 행렬로 바꾸기


			//테스트 단계
			Mat predictedLabels;
			knn->findNearest(test_img, 3, predictedLabels);  // k=3
			float prediction = predictedLabels.at<float>(0);

			//라벨 인덱스에 따른 의상 종류를 분류 (10종류)
			int indexprediction = prediction;
			//imshow("testimg", test_img);
			cout << "테스트 샘플의 라벨 = " << prediction << " => " << labelArray[indexprediction] << '\n';
			//cout << accuracy << "\n";


		}
		else if (key == '3') {
			Mat skeleton = img.clone();
			threshold(skeleton, skeleton, 127, 255, cv::THRESH_BINARY);

			//imshow("threshold", skeleton);
			Mat skel(img.size(), CV_8UC1, Scalar(0)); // skeleton = 0
			Mat element = getStructuringElement(MORPH_CROSS, Size(3, 3));
			Mat temp, eroded;

			do
			{
				erode(skeleton, eroded, element);
				dilate(eroded, temp, element);
				subtract(skeleton, temp, temp);  // 빼기 : outline 돌출 부분
				bitwise_or(skel, temp, skel); //OR : 기존의 skeleto에 합침.
				eroded.copyTo(skeleton);
			} while ((countNonZero(skeleton) != 0));

			imshow("skeletonization", skel);
			//여기까지 골격화 했음.

			//train 이미지는 그대로 해보쟈 한번? (지금 코드는 test이미지만 골격)
			Mat train_features(900, 784, CV_32FC1);
			Mat sklabels(900, 1, CV_32FC1);
			string sklabelArray[7] = { "top","pants","dress","shoes","sneakers","bag","boots" };

			// 각 숫자 영상을 row vector로 만들어서 train_features에 저장한다. 
			for (int r = 0; r < 30; r++) {     // 옷 이미지 세로로 30개
				for (int c = 0; c < 30; c++) { // 옷 이미지 가로로 30개
					int i = 0;
					for (int y = 0; y < 28; y++) { // 28x28
						for (int x = 0; x < 28; x++) {
							train_features.at<float>(r * 30 + c, i++)
								= skel.at<uchar>(r * 28 + y, c * 28 + x);
						}
					}
				}
			}
			// 각 숫자 영상에 대한 레이블을 저장한다. 
			for (int i = 0; i < 900; i++) {  // 의상의 카테고리 10개가 각각 90개씩
				int temp = i / 90;
				if (temp == 2 || temp == 4 || temp == 6)
					sklabels.at<float>(i, 0) = 0;
				else if (temp == 0)
					sklabels.at<float>(i, 0) = 0;
				else if (temp == 3)
					sklabels.at<float>(i, 0) = 2;
				else if (temp == 5)
					sklabels.at<float>(i, 0) = 3;
				else if (temp == 1)
					sklabels.at<float>(i, 0) = 1;
				else if (temp == 7)
					sklabels.at<float>(i, 0) = 4;
				else if (temp == 8 || temp == 9)
					sklabels.at<float>(i, 0) = temp - 3;

			}

			// 학습 단계
			Ptr<ml::KNearest> knn2 = ml::KNearest::create();
			Ptr<ml::TrainData> trainData = ml::TrainData::create(train_features, ml::ROW_SAMPLE, sklabels);
			knn2->train(trainData);

			// 테스트 단계
			Mat predictedLabels;
			float accuracy = 0;
			for (int i = 0; i < 900; i++) { // 의상 900개에 대해서
				Mat test = train_features.row(i); //train 데이터를 test 데이터로 사용
				knn2->findNearest(test, 3, predictedLabels);  // k=3
				float prediction = predictedLabels.at<float>(0);

				if (predictedLabels.at<float>(0) == sklabels.at<float>(i, 0))
					accuracy++;
				//라벨 인덱스에 따른 의상 종류를 분류 (10종류)
				int indexprediction = prediction;
				cout << "테스트 샘플" << i + 1 << "의 라벨 = " << prediction << " => " << sklabelArray[indexprediction] << '\n';
				//cout << accuracy << "\n";
			}
			//정확도 구하기
			float result = (accuracy / 900.0) * 100;
			cout << "정확도: " << result << "%" << endl;
		}
	}
}

