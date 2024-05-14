#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <iostream>

using namespace std;
using namespace cv;

vector<int> getContourCenter(const std::vector<cv::Point>& contour) {
    Moments M = moments(contour);
    if (M.m00 == 0) {
        return {};
    }
    int x = M.m10 / M.m00;
    int y = M.m01 / M.m00;
    return { x, y };
}

float getContourExtent(const vector<Point>& contour) {
    double area = contourArea(contour);
    Rect boundingRect = cv::boundingRect(contour);
    double rect_area = boundingRect.width * boundingRect.height;
    if (rect_area > 0) {
        return static_cast<float>(area / rect_area);
    }
    return 0.0f;
}

bool Aprox(int a, int b, int error) {
    return abs(a - b) < error;
}

void correctMainContour(int prev_cx, vector<vector<Point>>& contours, vector<Point>& MainContour, int& contourCenterX) {
    if (abs(prev_cx - contourCenterX) > 5) {
        for (size_t i = 0; i < contours.size(); ++i) {
            vector<int> tmp_cx = getContourCenter(contours[i]);
            if (!tmp_cx.empty()) {
                if (Aprox(tmp_cx[0], prev_cx, 5)) {
                    MainContour = contours[i];
                    vector<int> contourCenter = getContourCenter(MainContour);
                    if (!contourCenter.empty()) {
                        contourCenterX = contourCenter[0];
                    }
                    break;
                }
            }
        }
    }
}

void ProcessImage(cv::Mat& image) {
    int contourCenterX = 0;
    std::vector<vector<cv::Point>> contours;
    std::vector<Point> MainContour;
    int middleX, middleY;

    Mat imgray;
    cvtColor(image, imgray, cv::COLOR_BGR2GRAY);
    Mat thresh;
    threshold(imgray, thresh, 100, 255, cv::THRESH_BINARY_INV);

    findContours(thresh, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    if (!contours.empty()) {
        MainContour = *std::max_element(contours.begin(), contours.end(),
            [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                return cv::contourArea(a) < cv::contourArea(b);
            });

        int height = image.rows;
        int width = image.cols;
        middleX = width / 2;
        middleY = height / 2;

        int prev_cX = contourCenterX;
        std::vector<cv::Point> prev_MC = MainContour;

        std::vector<int> contourCenter = getContourCenter(MainContour);
        if (!contourCenter.empty()) {
            contourCenterX = contourCenter[0];
            if (std::abs(prev_cX - contourCenterX) > 5) {
                correctMainContour(prev_cX, contours, MainContour, contourCenterX);
            }
        } else {
            contourCenterX = 0;
        }

        int dir = (middleX - contourCenterX) * getContourExtent(MainContour);

        drawContours(image, std::vector<std::vector<cv::Point>>{MainContour}, -1, cv::Scalar(0, 255, 0), 3);
        circle(image, cv::Point(contourCenterX, middleY), 7, cv::Scalar(255, 255, 255), -1);
        circle(image, cv::Point(middleX, middleY), 3, cv::Scalar(0, 0, 255), -1);

        putText(image, std::to_string(middleX - contourCenterX), cv::Point(contourCenterX + 20, middleY),
        FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(200, 0, 200), 2);
        putText(image, "Weight: " + std::to_string(getContourExtent(MainContour)),
        Point(contourCenterX + 20, middleY + 35), cv::FONT_HERSHEY_SIMPLEX, 0.5, Scalar(200, 0, 200), 1);


        cout << dir << endl;
    }
}

int main() {
    cv::VideoCapture cap(2); // Open the default camera
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open the webcam." << std::endl;
        return 1;
    }

    cv::Mat frame;
    while (true) {
        cap >> frame; // Capture frame from the camera
        if (frame.empty()) {
            std::cerr << "Error: Could not capture frame." << std::endl;
            break;
        }

        ProcessImage(frame); // Process the captured frame

        cv::imshow("Processed Frame", frame);
        if (cv::waitKey(30) == 27) { // Press ESC to exit
            break;
        }
    }

    cap.release(); // Release the camera
    cv::destroyAllWindows();
    return 0;
}
