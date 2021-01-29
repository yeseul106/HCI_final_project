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
			// �� ���� ������ row vector�� ���� train_features�� �����Ѵ�. 
			for (int r = 0; r < 30; r++) {     // �� �̹��� ���η� 30��
				for (int c = 0; c < 30; c++) { // �� �̹��� ���η� 30��
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

			// �� ���� ���� ���� ���̺��� �����Ѵ�. 
			for (int i = 0; i < 900; i++) {  // �ǻ��� ī�װ� 10���� ���� 90����
				labels.at<float>(i, 0) = (i / 90);
			}

			// �н� �ܰ�
			Ptr<ml::KNearest> knn = ml::KNearest::create();
			Ptr<ml::TrainData> trainData = ml::TrainData::create(train_features, ml::ROW_SAMPLE, labels);
			knn->train(trainData);

			// �׽�Ʈ �ܰ�
			Mat predictedLabels;
			float accuracy = 0;
			for (int i = 0; i < 900; i++) { // �ǻ� 900���� ���ؼ�
				Mat test = train_features.row(i); //train �����͸� test �����ͷ� ���
				knn->findNearest(test, 3, predictedLabels);  // k=3
				float prediction = predictedLabels.at<float>(0);

				if (predictedLabels.at<float>(0) == labels.at<float>(i, 0))
					accuracy++;
				//�� �ε����� ���� �ǻ� ������ �з� (10����)
				int indexprediction = prediction;
				cout << "�׽�Ʈ ����" << i + 1 << "�� �� = " << prediction << " => " << labelArray[indexprediction] << '\n';
				//cout << accuracy << "\n";
			}
			//��Ȯ�� ���ϱ�
			float result = (accuracy / 900.0) * 100;
			cout << "��Ȯ��: " << result << "%" << endl;
		}
		else if (key == '2') {

			//�׽�Ʈ ������ �ޱ�
			/*string fileName;
			cout << "���ϸ��� Ȯ���ڱ��� ��Ȯ�� �Է��ϼ���.(���Ḧ ���� �� break�Է�!) : ";
			getline(cin, fileName);

			if (fileName.compare("break") == 0) {
				cout << "�����մϴ�..." << endl;
				break;
			}
			*/

			Mat img2;
			img2 = imread("fashion.png", IMREAD_GRAYSCALE);
			Mat train_features(900, 784, CV_32FC1);
			Mat labels(900, 1, CV_32FC1);
			// �� ���� ������ row vector�� ���� train_features�� �����Ѵ�. 
			for (int r = 0; r < 30; r++) {     // �� �̹��� ���η� 30�� 
				for (int c = 0; c < 30; c++) { // �� �̹��� ���η� 30��
					int i = 0;
					for (int y = 0; y < 28; y++) { // 28x28
						for (int x = 0; x < 28; x++) {
							train_features.at<float>(r * 30 + c, i++)
								= img2.at<uchar>(r * 28 + y, c * 28 + x);
						}
					}
				}
			}

			// �� ���� ���� ���� ���̺��� �����Ѵ�. 
			for (int i = 0; i < 900; i++) {  // �ǻ��� ī�װ� 10���� ���� 90����
				labels.at<float>(i, 0) = (i / 90);
			}

			// �н� �ܰ�
			Ptr<ml::KNearest> knn = ml::KNearest::create();
			Ptr<ml::TrainData> trainData = ml::TrainData::create(train_features, ml::ROW_SAMPLE, labels);
			knn->train(trainData);

			//�׽�Ʈ �̹��� �������� (ũ�� 28 X 28�� ����)
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
			//����и��ϱ� (�������� �ֽô� ���� ��������!)

			threshold(test_img, test_img, 0, 255, THRESH_OTSU | THRESH_BINARY_INV);


			imshow("testimg_threshold", test_img);
			test_img = test_img.reshape(0, 1);
			test_img.convertTo(test_img, CV_32FC1);
			//imshow("f", test_img);

			//�׽�Ʈ ������ 1���� ��ķ� �ٲٱ�


			//�׽�Ʈ �ܰ�
			Mat predictedLabels;
			knn->findNearest(test_img, 3, predictedLabels);  // k=3
			float prediction = predictedLabels.at<float>(0);

			//�� �ε����� ���� �ǻ� ������ �з� (10����)
			int indexprediction = prediction;
			//imshow("testimg", test_img);
			cout << "�׽�Ʈ ������ �� = " << prediction << " => " << labelArray[indexprediction] << '\n';
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
				subtract(skeleton, temp, temp);  // ���� : outline ���� �κ�
				bitwise_or(skel, temp, skel); //OR : ������ skeleto�� ��ħ.
				eroded.copyTo(skeleton);
			} while ((countNonZero(skeleton) != 0));

			imshow("skeletonization", skel);
			//������� ���ȭ ����.

			//train �̹����� �״�� �غ��� �ѹ�? (���� �ڵ�� test�̹����� ���)
			Mat train_features(900, 784, CV_32FC1);
			Mat sklabels(900, 1, CV_32FC1);
			string sklabelArray[7] = { "top","pants","dress","shoes","sneakers","bag","boots" };

			// �� ���� ������ row vector�� ���� train_features�� �����Ѵ�. 
			for (int r = 0; r < 30; r++) {     // �� �̹��� ���η� 30��
				for (int c = 0; c < 30; c++) { // �� �̹��� ���η� 30��
					int i = 0;
					for (int y = 0; y < 28; y++) { // 28x28
						for (int x = 0; x < 28; x++) {
							train_features.at<float>(r * 30 + c, i++)
								= skel.at<uchar>(r * 28 + y, c * 28 + x);
						}
					}
				}
			}
			// �� ���� ���� ���� ���̺��� �����Ѵ�. 
			for (int i = 0; i < 900; i++) {  // �ǻ��� ī�װ� 10���� ���� 90����
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

			// �н� �ܰ�
			Ptr<ml::KNearest> knn2 = ml::KNearest::create();
			Ptr<ml::TrainData> trainData = ml::TrainData::create(train_features, ml::ROW_SAMPLE, sklabels);
			knn2->train(trainData);

			// �׽�Ʈ �ܰ�
			Mat predictedLabels;
			float accuracy = 0;
			for (int i = 0; i < 900; i++) { // �ǻ� 900���� ���ؼ�
				Mat test = train_features.row(i); //train �����͸� test �����ͷ� ���
				knn2->findNearest(test, 3, predictedLabels);  // k=3
				float prediction = predictedLabels.at<float>(0);

				if (predictedLabels.at<float>(0) == sklabels.at<float>(i, 0))
					accuracy++;
				//�� �ε����� ���� �ǻ� ������ �з� (10����)
				int indexprediction = prediction;
				cout << "�׽�Ʈ ����" << i + 1 << "�� �� = " << prediction << " => " << sklabelArray[indexprediction] << '\n';
				//cout << accuracy << "\n";
			}
			//��Ȯ�� ���ϱ�
			float result = (accuracy / 900.0) * 100;
			cout << "��Ȯ��: " << result << "%" << endl;
		}
	}
}

