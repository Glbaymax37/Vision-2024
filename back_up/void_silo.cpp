#include <opencv2/opencv.hpp>  // Include OpenCV API
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <iostream>
#include <vector>
#include <time.h>
#include <fstream>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/float32.hpp"
#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/int32.hpp>

using namespace cv;
using namespace std;

int thres_hold = 52;

int dialedx = 49;
int dialedy = 52;

int erodex = 5;
int erodey = 6;

Mat frame;
vector<Point> titik;

void onTrackbar(int, void*) {
}

double angle();
double slope(Point p1, Point p2);

double angle() {
    Point a = titik[0];
    Point b = titik[1];
    Point c = titik[2];

    double m1 = slope(b, a);
    double m2 = slope(b, c);

    double angle = atan((m2 - m1) / (1 + m1 * m2));
    angle = round(angle * 180 / CV_PI);

    if (angle < 0) {
        angle = 180 + angle;
    }

    putText(frame, to_string(angle), Point(b.x - 40, b.y + 40), FONT_HERSHEY_DUPLEX, 2, Scalar(0, 0, 255), 2, LINE_AA);
    return angle;
}

double slope(Point p1, Point p2) {
    return static_cast<double>(p2.y - p1.y) / (p2.x - p1.x);
}

void silo(){
    rclcpp::NodeOptions options;
    auto node = std::make_shared<rclcpp::Node>("Posisi_Silo", options);

    auto publisher = node->create_publisher<geometry_msgs::msg::Twist>("Posisi_Silo", 10);

    rclcpp::WallRate loop_rate(10); 
    VideoCapture cap(2); // 0 untuk webcam utama, bisa diganti dengan path video

    if (!cap.isOpened()) {
        std::cout << "Error: Webcam tidak bisa diakses\n";
        // return -1;
    }

    // Buat jendela untuk menampung trackbar
    namedWindow("Trackbars", WINDOW_AUTOSIZE);
    createTrackbar("Threshold", "Trackbars", &thres_hold, 255, onTrackbar);

    createTrackbar("dialedx", "Trackbars", &dialedx, 180, onTrackbar);
    createTrackbar("dialedy", "Trackbars", &dialedy, 180, onTrackbar);
    createTrackbar("erodex", "Trackbars", &erodex, 180, onTrackbar);
    createTrackbar("erodey", "Trackbars", &erodey, 180, onTrackbar);


    double min_aspect_ratio = 0.2; // minimum width/height ratio
    double max_aspect_ratio = 5.0; 

    // // Loop utama
     while (rclcpp::ok()) {
        cap >> frame;
        flip(frame, frame, -1);

        if (frame.empty()) {
            std::cout << "Error: Tidak ada frame yang didapat\n";
            break;
        }

        Mat displayFrame = Mat::zeros(480, 640, CV_8UC3);
        int x1 = (640 - 320) / 2;  // Top-left corner X coordinate
        int y1 = (480 - 240) / 2;  // Top-left corner Y coordinate
        int x2 = x1 + 320;         // Bottom-right corner X coordinate
        int y2 = y1 + 240;    
        // Ubah warna ke HSV
        frame(cv::Rect(x1, y1, 320, 240)).copyTo(displayFrame(cv::Rect(x1, y1, 320, 240)));

        Mat gray;

        Mat thres = cv::Mat::zeros(frame.size(), frame.type());
        cvtColor(displayFrame, gray, COLOR_BGR2GRAY); // Ubah citra ke grayscale
        threshold(gray, thres, thres_hold, 255, THRESH_BINARY_INV); 

        Mat threskan;
        threshold(gray, threskan, thres_hold, 255, THRESH_BINARY);

        cv::Mat frame2 = cv::Mat::zeros(displayFrame.size(), displayFrame.type());
        displayFrame.copyTo(frame2);

        cv::Mat thres2 = cv::Mat::zeros(displayFrame.size(), displayFrame.type());
        thres.copyTo(thres2);

        cvtColor(thres2, thres2, COLOR_GRAY2BGR);
        erode(thres2, thres2, getStructuringElement(MORPH_RECT, Size(2, 2)));
        dilate(thres2, thres2, getStructuringElement(MORPH_RECT, Size(8, 8)));

        Mat gabung;
        bitwise_and(frame2, thres2, gabung);

        Mat putih;
        threshold(gabung, putih, thres_hold, 255, THRESH_BINARY);
        cvtColor(gabung, putih, COLOR_BGR2GRAY);

        cv::Mat frame3 = cv::Mat::zeros(frame.size(), frame.type());
        gabung.copyTo(frame3);

        Mat graykhus;
        cvtColor(frame3, graykhus, COLOR_BGR2GRAY);

        Mat hsv;
        cvtColor(frame3, hsv, COLOR_BGR2HSV);

        Mat graylah;

        Mat threslag;
        threshold(graykhus, threslag, thres_hold, 255, THRESH_BINARY);

        Mat last_thres;
        cvtColor(frame3, graylah, COLOR_BGR2GRAY);
        threshold(graylah, last_thres, thres_hold, 255, THRESH_BINARY);

        erode(last_thres, last_thres, getStructuringElement(MORPH_RECT, Size(erodex, erodey)));
        dilate(last_thres, last_thres, getStructuringElement(MORPH_RECT, Size(dialedx, dialedy)));

        Point koordinat(0, 0);

        // Temukan kontur di dalam mask
        std::vector<std::vector<Point>> contours;
        std::vector<Vec4i> hierarchy;
        findContours(last_thres, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        
        double maxArea = 0;
        int maxAreaIdx = -1;

        double area;

        for (size_t i = 0; i < contours.size(); ++i) {
            area = contourArea(contours[i]);
            if (area > maxArea) {
                maxArea = area;
                maxAreaIdx = static_cast<int>(i);
            }
        }

        if (maxAreaIdx != -1 && maxArea > 50) {
            // Process the largest contour only
            const auto& contour = contours[maxAreaIdx];

            // Your existing code for processing the contour goes here
            Moments oMoments = moments(contour);
            double dM01 = oMoments.m01;
            double dM10 = oMoments.m10;
            double dArea = oMoments.m00;

            //Rectangle Object 1
            Rect boundingBox = boundingRect(contour);
            rectangle(frame, boundingBox, Scalar(255, 0, 0), 2);

            // Circle Object1
            Point circleCenter(boundingBox.x + boundingBox.width / 2, boundingBox.y + boundingBox.height / 2);
            int circleRadius = (boundingBox.width + boundingBox.height) / 4;
            circle(frame, circleCenter, circleRadius, Scalar(0, 0, 255), 2);

            // Point Object1
            Point objectCenter(boundingBox.x + boundingBox.width / 2, boundingBox.y + boundingBox.height / 2);
            Point frameCenter(frame.cols / 2, frame.rows);

            //Rectangle background object 1
            Size textSize = getTextSize("Object1", FONT_HERSHEY_SIMPLEX, 0.5, 2, nullptr);
            rectangle(frame, Point(boundingBox.x, boundingBox.y - textSize.height - 10), Point(boundingBox.x + textSize.width, boundingBox.y), Scalar(255, 0, 0), FILLED);

            //garis dan keterangan
            putText(frame, "Object1", Point(boundingBox.x, boundingBox.y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 2);
            line(frame, frameCenter, objectCenter, Scalar(0, 255, 0), 2);
            circle(frame, Point(objectCenter), 5, Scalar(100, 255, 100), FILLED);

            // Koordinat Object 1
            string coordinateText1 = "(" + to_string(objectCenter.x) + ", " + to_string(objectCenter.y) + ")";
            putText(frame, coordinateText1, Point(objectCenter.x - 20, objectCenter.y - 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(250, 50, 255), 2);

            //Rectangle background Depth
            rectangle(frame, Point(500, 430), Point(640, 470), Scalar(200, 0, 0), FILLED);
            //Rectangle Keterangan
            rectangle(frame, Point(550, 410), Point(640, 430), Scalar(0, 0, 0), FILLED);

            putText(frame, "SILO", Point(560, 425), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 2);

            // Publish the message
            geometry_msgs::msg::Twist message;
            message.linear.x = objectCenter.x; // Replace with actual value
            message.linear.y = objectCenter.y; // Replace with actual value
            message.angular.z = maxArea; // Replace with actual value (e.g., area)
            publisher->publish(message);
        }

        imshow("Frame", frame);
        imshow("gabung", gabung);
        imshow("Puh", frame3);
        imshow("lam", last_thres);

        rclcpp::spin_some(node); // Handle callbacks

        // Tunggu tombol 'q' ditekan untuk keluar
        if (waitKey(1) == 'q') {
            break;
        }
    }




}

int main(int argc, char** argv) {
    // Initialize ROS 2
    rclcpp::init(argc, argv);


    while(true){

        silo();


    }
    

    // Tutup webcam dan destroy windows
    frame.release();
    destroyAllWindows();
    rclcpp::shutdown();

    return 0;
    
}
