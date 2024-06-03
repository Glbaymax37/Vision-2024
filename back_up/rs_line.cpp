#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <iostream>

using namespace std;
using namespace cv;

Mat thresh;

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



int main() {
    // Create a RealSense pipeline, encapsulating the actual device and sensors
    rs2::pipeline p;

    // Create a configuration for configuring the pipeline with a non default profile
    rs2::config cfg;

    // Add desired streams to configuration
    int frame_width = 640;
    int frame_height = 480;
    cfg.enable_stream(RS2_STREAM_COLOR, frame_width, frame_height, RS2_FORMAT_BGR8, 30);

    // Instruct pipeline to start streaming with the requested configuration
    p.start(cfg);

    while (true) {
        // Wait for the next set of frames from the camera
        rs2::frameset frames = p.wait_for_frames();
        rs2::frame color_frame = frames.get_color_frame();

        // Convert RealSense frame to OpenCV Mat
        Mat frame(Size(color_frame.as<rs2::video_frame>().get_width(),
                       color_frame.as<rs2::video_frame>().get_height()),
                  CV_8UC3, (void*)color_frame.get_data(), Mat::AUTO_STEP);

         int contourCenterX = 0;


    std::vector<vector<cv::Point>> contours;
    std::vector<Point> MainContour;
    int middleX, middleY;

    Mat imgray;
    cvtColor(frame, imgray, cv::COLOR_BGR2GRAY);
    
    threshold(imgray, thresh, 100, 255, cv::THRESH_BINARY_INV);

    findContours(thresh, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    if (!contours.empty()) {
        MainContour = *std::max_element(contours.begin(), contours.end(),
            [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                return cv::contourArea(a) < cv::contourArea(b);
            });

        int height = frame.rows;
        int width = frame.cols;
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

        drawContours(frame, std::vector<std::vector<cv::Point>>{MainContour}, -1, cv::Scalar(0, 255, 0), 3);
        circle(frame, cv::Point(contourCenterX, middleY), 7, cv::Scalar(255, 255, 255), -1);
        circle(frame, cv::Point(middleX, middleY), 3, cv::Scalar(0, 0, 255), -1);

        putText(frame, std::to_string(middleX - contourCenterX), cv::Point(contourCenterX + 20, middleY),
        FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(200, 0, 200), 2);
        putText(frame, "Weight: " + std::to_string(getContourExtent(MainContour)),
        Point(contourCenterX + 20, middleY + 35), cv::FONT_HERSHEY_SIMPLEX, 0.5, Scalar(200, 0, 200), 1);

        cout << dir << endl;

        cv::imshow("Processed Frame", frame);
        if (cv::waitKey(30) == 27) { // Press ESC to exit
            break;
        }
    }
    }
    return 0;
}
