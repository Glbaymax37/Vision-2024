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

int thres_hold = 80;

int dialedx = 67;
int dialedy = 18;

int erodex = 1;
int erodey = 16;

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




int main(int argc, char** argv) {
    // Initialize ROS 2
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions options;
    auto node = std::make_shared<rclcpp::Node>("Posisi_Silo", options);

    auto publisher = node->create_publisher<geometry_msgs::msg::Twist>("Posisi_Silo", 10);

    rclcpp::WallRate loop_rate(10); 
    VideoCapture cap(2); // 0 untuk webcam utama, bisa diganti dengan path video

    if (!cap.isOpened()) {
        std::cout << "Error: Webcam tidak bisa diakses\n";
        return -1;
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

    const int maxObjects = 5;
    vector<double> areas(maxObjects, 0); // Simpan area dari lima objek terbesar
    vector<int> maxAreaIdxs(maxObjects, -1); // Simpan indeks dari lima objek terbesar


    int targetX = 320;
    Rect closestRect; 

    // Loop utama
    while (rclcpp::ok()) {
        cap >> frame;
        // flip(frame, frame, -1);

        if (frame.empty()) {
            std::cout << "Error: Tidak ada frame yang didapat\n";
            break;
        }


        Mat gray;

        Mat thres = cv::Mat::zeros(frame.size(), frame.type());
        cvtColor(frame, gray, COLOR_BGR2GRAY); // Ubah citra ke grayscale
        threshold(gray, thres, thres_hold, 255, THRESH_BINARY_INV); 

        Mat threskan;
        threshold(gray, threskan, thres_hold, 255, THRESH_BINARY);

        cv::Mat frame2 = cv::Mat::zeros(frame.size(), frame.type());
        frame.copyTo(frame2);

        cv::Mat thres2 = cv::Mat::zeros(frame.size(), frame.type());
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

        fill(areas.begin(), areas.end(), 0);
        fill(maxAreaIdxs.begin(), maxAreaIdxs.end(), -1);


        for (size_t i = 0; i < contours.size(); ++i) {
            double area = contourArea(contours[i]);
            if (area > areas[maxObjects - 1]) {
                areas[maxObjects - 1] = area;
                maxAreaIdxs[maxObjects - 1] = i;

                // Urutkan areas dan maxAreaIdxs
        for (int j = maxObjects - 2; j >= 0; --j) {
                    if (areas[j] < areas[j + 1]) {
                        swap(areas[j], areas[j + 1]);
                        swap(maxAreaIdxs[j], maxAreaIdxs[j + 1]);
                    } else {
                        break;
                    }
                }
            }
        }


     double minDistance = DBL_MAX;

    for (int i = 0; i < maxObjects; ++i) {
            if (maxAreaIdxs[i] != -1) {

       

        const auto& contour = contours[i];

        Moments oMoments = moments(contour);
        double dM01 = oMoments.m01;
        double dM10 = oMoments.m10;
        double dArea = oMoments.m00;

      

        Rect boundingBox = boundingRect(contour);
        rectangle(frame, boundingBox, Scalar(255, 0, 0), 2);

        Point circleCenter(boundingBox.x + boundingBox.width / 2, boundingBox.y + boundingBox.height / 2);
        int circleRadius = (boundingBox.width + boundingBox.height) / 4;
        circle(frame, circleCenter, circleRadius, Scalar(0, 0, 255), 2);

        Point objectCenter(boundingBox.x + boundingBox.width / 2, boundingBox.y + boundingBox.height / 2);
        Point frameCenter(frame.cols / 2, frame.rows);

        Size textSize = getTextSize("Object", FONT_HERSHEY_SIMPLEX, 0.5, 2, nullptr);
        rectangle(frame, Point(boundingBox.x, boundingBox.y - textSize.height - 10), Point(boundingBox.x + textSize.width, boundingBox.y), Scalar(255, 0, 0), FILLED);

        // Rect detectionRegion(100, 200, 100, 400);

        //     // Draw the detection region on the frame
        // rectangle(frame, detectionRegion, Scalar(0, 255, 0), 2);
        
        
        int centerX = boundingBox.x + boundingBox.width / 2; 
        double distance = abs(centerX - targetX);

         int tolerance = 120;
       

   

            
        if (distance <= tolerance && distance < minDistance) {
        minDistance = distance;
        closestRect = boundingRect(contours[i]);
            putText(frame, "*", objectCenter, FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 255), 2);
            


        titik.push_back(Point(0,480));
        circle(frame,Point(0,480), 5, Scalar(255, 0, 0), -1);

  

        titik.push_back(Point(320, 480));
        circle(frame, Point(320, 480), 5, Scalar(255, 0, 0), -1);
        arrowedLine(frame, titik[1], titik[0], Scalar(255, 0, 0), 3);

        titik.push_back(objectCenter);
        circle(frame, objectCenter, 5, Scalar(255, 0, 0), -1);
        arrowedLine(frame, titik[1], objectCenter, Scalar(255, 0, 0), 3);

        double degrees = angle();

        Point a = titik[0];
        Point b = titik[1];
        Point c = objectCenter;

        double m1 = slope(b, a);
        double m2 = slope(b, c);

        double angle = atan((m2 - m1) / (1 + m1 * m2));
        angle = round(angle * 180 / CV_PI);

        string posisi;

        if (angle < 0) {
            angle = 180 + angle;
        }

        if(angle > 90){
            posisi = "\nPosisi Object = Kiri";
        }
        if(angle < 90){
            posisi = "\nPosisi Object = Kanan";
        }
        if(angle == 90){
            
            posisi = "\nPosisi Object = Tengah";
        }

        ostringstream angleText;
        angleText << fixed << setprecision(2) << angle;

        string angleString = "\nSudut Object (" + to_string(angle)+")";

        putText(frame, "Object", Point(boundingBox.x, boundingBox.y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 2);
        line(frame, frameCenter, objectCenter, Scalar(0, 255, 0), 2);
        circle(frame, Point(objectCenter), 5, Scalar(100, 255, 100), FILLED);

        string coordinateText = "(" + to_string(objectCenter.x) + ", " + to_string(objectCenter.y) + ")";
        putText(frame, coordinateText, Point(objectCenter.x - 20, objectCenter.y - 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(250, 50, 255), 2);

        rectangle(frame, Point(500, 430), Point(640, 470), Scalar(200, 0, 0), FILLED);
        rectangle(frame, Point(550, 410), Point(640, 430), Scalar(0, 0, 0), FILLED);

        putText(frame, "SILO", Point(560, 425), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 2);

        putText(frame, to_string(angle), Point(500,30), FONT_HERSHEY_DUPLEX, 0.8, Scalar(0, 0, 255), 2, LINE_AA);

        geometry_msgs::msg::Twist message;
        message.linear.x = objectCenter.x;
        message.linear.y = objectCenter.y;
        message.linear.z = angle;
        message.angular.z = area;
        publisher->publish(message);
        

        }

    }


    else{
        geometry_msgs::msg::Twist message;
        message.linear.x = 0;
        message.linear.y = 0;
        message.linear.z = 0;
        message.angular.z = 0;
        publisher->publish(message);

    }
    
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
    


    // Tutup webcam dan destroy windows
    cap.release();
    destroyAllWindows();
    rclcpp::shutdown();

    return 0;
}
