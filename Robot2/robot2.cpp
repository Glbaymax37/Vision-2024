#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
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
using namespace rs2;

#define COLOR_ROWS 80
#define COLOR_COLS 250

cv::Rect roi; 


Mat thresh;
Mat thresh2;
Mat white_mask;



int whmin = 0;
int wsmin = 0;
int wvmin = 0;
int whmax = 255;
int wsmax = 255;
int wvmax = 255;


vector<Point> titik;
Ptr<BackgroundSubtractor> pBackSub = createBackgroundSubtractorKNN();
Mat frames,fgMask,snapshot;
Mat hsvArray1, hsvArray2;

string warna;
string hsvWindowName1 = "hsvObject1";
string hsvWindowName2 = "hsvObject2";
Scalar hsvObject1;

float hue,saturation;
int threshold_value = 200;

double fpsLive;
int num_frames = 1;
double angle();
double slope(Point p1, Point p2);
struct ObjectData {
    int b, g, r;
};
vector<ObjectData> objectDataList;
void resetObjects() {
    objectDataList.clear();
    cout << "Objects reset." << endl;
}
void applyColormap(const Mat& depth, Mat& colorizedDepth) {
    applyColorMap(depth, colorizedDepth, COLORMAP_JET);
}
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

    putText(frames, to_string(angle), Point(b.x - 40, b.y + 40), FONT_HERSHEY_DUPLEX, 2, Scalar(0, 0, 255), 2, LINE_AA);
    return angle;
}
double slope(Point p1, Point p2) {
    return static_cast<double>(p2.y - p1.y) / (p2.x - p1.x);
}

// Variabel global untuk menyimpan nilai trackbar
int low_h = 30, low_s = 50, low_v = 50;
int high_h = 130, high_s = 255, high_v = 255;

int dialedx = 24;
int dialedy = 46;

int erodex = 5;
int erodey = 5;

int min_width = 1;
int max_width = 10000;
int thres_hold = 55;


pipeline pipekan;
rs2::config configs;
int keyVal;

int b, g, r;

int lowhue1 = 164, smin1 = 117,vmin1 = 0;
int upperhue1 = 178, smax1 = 255,vmax1 = 255;


int lowhuepadi = 164, sminpadi = 105, vminpadi = 0;
int upperhuepadi = 178 ,smaxpadi = 255, vmaxpadi = 255;


int lowhue2 = 105, smin2 = 36,vmin2 = 0;
int upperhue2 = 119, smax2 = 255,vmax2 = 255;

int konstanta1 = 7;
int konstanta2 = 10;

int nilai_bawah;
int saturasi_bawah;
int value_bawah;
int nilai_atas;
int saturasi_atas;
int value_atas;

int sensitivity = 15;

int smin;
int smax;
int vmin;
int vmax;

// Line_Follower
class Image {
private:
    cv::Mat image;
    int contourCenterX;
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Point> MainContour;
    int height, width;
    int middleX, middleY;
    int dir;
    std::vector<cv::Point> prev_MC;
    int prev_cX;
    
public:
    Image() : contourCenterX(0) {}

    void setImage(const cv::Mat& img) {
        image = img;
    }

    cv::Mat getImage() const {
        return image;
    }

    void Process() {
        cv::Mat imgray;

        cv::cvtColor(image, imgray, cv::COLOR_BGR2GRAY);
        cv::threshold(imgray, thresh, thres_hold, 255, cv::THRESH_BINARY);
        Scalar White_lower(whmin,wsmin,wvmin);
        Scalar Upper_white(whmax,wsmax,wvmax);
        inRange(imgray,White_lower,Upper_white,white_mask);
        threshold(imgray, thresh2, thres_hold, 255, cv::THRESH_BINARY);
        cv::findContours(white_mask, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

        prev_MC = MainContour;
        if (!contours.empty()) {
            MainContour = *std::max_element(contours.begin(), contours.end(), [](const auto& c1, const auto& c2) {
                return cv::contourArea(c1) < cv::contourArea(c2);
            });

            height = image.rows;
            width = image.cols;

            middleX = width / 2;
            middleY = height / 2;

            prev_cX = contourCenterX;
            if (!getContourCenter(MainContour).empty()) {
                contourCenterX = getContourCenter(MainContour)[0];
                if (std::abs(prev_cX - contourCenterX) > 5)
                    correctMainContour(prev_cX);
            } else {
                contourCenterX = 0;
            }

            dir = (middleX - contourCenterX) * getContourExtent(MainContour);

            cv::drawContours(image, std::vector<std::vector<cv::Point>>{MainContour}, -1, cv::Scalar(0, 255, 0), 3);
            cv::circle(image, cv::Point(contourCenterX, middleY), 7, cv::Scalar(255, 255, 255), -1);
            cv::circle(image, cv::Point(middleX, middleY), 3, cv::Scalar(0, 0, 255), -1);

            cv::putText(image, std::to_string(middleX - contourCenterX), cv::Point(contourCenterX + 20, middleY), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(200, 0, 200), 2);
            cv::putText(image, "Weight:" + std::to_string(getContourExtent(MainContour)), cv::Point(contourCenterX + 20, middleY + 35), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 0, 200), 1);
            cout << dir << endl;
        }
    }

    std::vector<int> getContourCenter(const std::vector<cv::Point>& contour) {
        cv::Moments M = cv::moments(contour);
        if (M.m00 == 0)
            return {};
        int x = static_cast<int>(M.m10 / M.m00);
        int y = static_cast<int>(M.m01 / M.m00);
        return {x, y};
    }

    float getContourExtent(const std::vector<cv::Point>& contour) {
        float area = cv::contourArea(contour);
        cv::Rect boundingRect = cv::boundingRect(contour);
        float rect_area = boundingRect.width * boundingRect.height;
        if (rect_area > 0)
            return area / rect_area;
        return 0;
    }

    bool Aprox(int a, int b, int error) {
        return std::abs(a - b) < error;
    }

    void correctMainContour(int prev_cx) {
        if (std::abs(prev_cx - contourCenterX) > 5) {
            for (size_t i = 0; i < contours.size(); ++i) {
                if (!getContourCenter(contours[i]).empty()) {
                    int tmp_cx = getContourCenter(contours[i])[0];
                    if (Aprox(tmp_cx, prev_cx, 5)) {
                        MainContour = contours[i];
                        if (!getContourCenter(MainContour).empty())
                            contourCenterX = getContourCenter(MainContour)[0];
                    }
                }
            }
        }
    }
};
void onTrackbar(int, void*) {
    
}

void SlicePart(cv::Mat im, std::vector<Image>& images, int slices) {
    int height = im.rows;
    int width = im.cols;
    int sl = height / slices;

    for (int i = 0; i < slices; ++i) {
        int part = sl * i;
        cv::Mat crop_img = im(cv::Rect(0, part, width, sl)).clone();
        images[i].setImage(crop_img);
        images[i].Process();
    }
}


void Silo() {
    rclcpp::NodeOptions options;
    auto node = std::make_shared<rclcpp::Node>("Posisi_Silo", options);

    auto publisher = node->create_publisher<geometry_msgs::msg::Twist>("Posisi_Silo", 10);

    rclcpp::WallRate loop_rate(10); 

    namedWindow("Trackbars", WINDOW_AUTOSIZE);
    createTrackbar("Threshold", "Trackbars", &thres_hold, 255, onTrackbar);

    createTrackbar("dialedx", "Trackbars", &dialedx, 180, onTrackbar);
    createTrackbar("dialedy", "Trackbars", &dialedy, 180, onTrackbar);
    createTrackbar("erodex", "Trackbars", &erodex, 180, onTrackbar);
    createTrackbar("erodey", "Trackbars", &erodey, 180, onTrackbar);
    

    // Tambahkan trackbar untuk nilai batas bawah
    createTrackbar("Low H", "Trackbars", &low_h, 180, onTrackbar);
    createTrackbar("Low S", "Trackbars", &low_s, 255, onTrackbar);
    createTrackbar("Low V", "Trackbars", &low_v, 255, onTrackbar);

    // Tambahkan trackbar untuk nilai batas atas
    createTrackbar("High H", "Trackbars", &high_h, 255, onTrackbar);
    createTrackbar("High S", "Trackbars", &high_s, 255, onTrackbar);
    createTrackbar("High V", "Trackbars", &high_v, 255, onTrackbar);


    std::vector<Image> images(3); // Change 10 to the number of slices you want
    int slices = images.size();

    //    while (rclcpp::ok) {
        rs2::frameset frames = pipekan.wait_for_frames();
        rs2::frame color_frame = frames.get_color_frame();
        rs2::frame depth_frame = frames.get_depth_frame();
        Mat frame(cv::Size(640, 480), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
        //resize(frame,frame,Size(500,500));

        // Ubah warna ke HSV
        Mat displayFrame = Mat::zeros(480, 640, CV_8UC3);
        int x1 =  390 /2;//(640 - 320) / 2;  // Top-left corner X coordinate
        int y1 = 0;//(480 - 240) / 2;  // Top-left corner Y coordinate
        int x2 = x1 + 320;         // Bottom-right corner X coordinate
        int y2 = y1 + 10;    
        // Ubah warna ke HSV
    frame(cv::Rect(x1, y1, 290, 360)).copyTo(displayFrame(cv::Rect(x1, y1, 290, 360)));


        Mat gray;

       Mat thres = cv::Mat::zeros(displayFrame.size(), displayFrame.type());
        cvtColor(displayFrame, gray, COLOR_BGR2GRAY); // Ubah citra ke grayscale
        threshold(gray, thres, thres_hold, 255, THRESH_BINARY_INV); // Lakukan thresholding pada citra grayscale



        Mat threskan;
        threshold(gray, threskan, thres_hold, 255, THRESH_BINARY);

        cv::Mat frame2 = cv::Mat::zeros(displayFrame.size(), displayFrame.type());
        displayFrame.copyTo(frame2);

        cv::Mat thres2 = cv::Mat::zeros(displayFrame.size(), displayFrame.type());
        thres.copyTo(thres2);

        cvtColor(thres2,thres2,COLOR_GRAY2BGR);
        erode(thres2,thres2, getStructuringElement(MORPH_RECT, Size(2, 2)));
        dilate(thres2, thres2, getStructuringElement(MORPH_RECT, Size(8, 8)));


        Mat gabung;
        bitwise_and(frame2,thres2,gabung);

        Mat putih;
        threshold(gabung, putih, thres_hold, 255, THRESH_BINARY);
        cvtColor(gabung,putih,COLOR_BGR2GRAY);

        cv::Mat frame3 = cv::Mat::zeros(displayFrame.size(), displayFrame.type());
        gabung.copyTo(frame3);

        Mat graykhus;
        cvtColor(frame3,graykhus,COLOR_BGR2GRAY);

        Mat threslag;
        threshold(graykhus, threslag, thres_hold, 255, THRESH_BINARY);

        
        Mat hsv;
        cvtColor(frame3, hsv, COLOR_BGR2HSV);

        // Tentukan range warna yang ingin di-segmentasi
        Scalar lower_bound(low_h, low_s, low_v);
        Scalar upper_bound(high_h, high_s, high_v);

        // Buat mask untuk segmentasi warna
        Mat mask;
        inRange(hsv, lower_bound, upper_bound, mask);

        Mat graylah;

          Mat last_thres;
        cvtColor(frame3,graylah, COLOR_BGR2GRAY);
        threshold(graylah,last_thres,thres_hold,255,THRESH_BINARY);

        erode(last_thres, last_thres, getStructuringElement(MORPH_RECT, Size(erodex, erodey)));
        dilate(last_thres, last_thres, getStructuringElement(MORPH_RECT, Size(dialedx, dialedy)));





        // Temukan kontur di dalam mask
        std::vector<std::vector<Point>> contours;
        std::vector<Vec4i> hierarchy;
        findContours(last_thres, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

         double maxArea = 0;
        int maxAreaIdx = -1;

        for (size_t i = 0; i < contours.size(); ++i) {
            double area = contourArea(contours[i]);
            if (area > maxArea) {
                maxArea = area;
                maxAreaIdx = static_cast<int>(i);
            }
        }
           if (maxAreaIdx != -1 && maxArea > 100) {
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

            geometry_msgs::msg::Twist message;
            message.linear.x = objectCenter.x;  // Kecepatan linier pada sumbu x
            message.linear.y = objectCenter.y; 


            publisher->publish(message);
        

        }        

       

        // Tampilkan hasil segmentasi
        imshow("Frame", frame);
        // imshow("hasil", threskan);
        // imshow("thres",thres);
        // imshow("gabung",gabung);
        // imshow("Putih",putih);
        imshow("Puh",frame3);
        imshow("KING",mask);
        //imshow("lord",threslag);
        imshow("lam",last_thres);

       
       
       
        
         keyVal = cv::waitKey(1) & 0xFF;

    //     if (keyVal == 113) { // 'Esc' u
    //         break;
    //     }
    // //}
}



    
void padi(){
 while (true) {
        // kamera RealSense
        frameset frames = pipekan.wait_for_frames();
        frame color_frame = frames.get_color_frame();
        depth_frame depth_frame = frames.get_depth_frame();
        const int width = depth_frame.as<video_frame>().get_width();
        const int height = depth_frame.as<video_frame>().get_height();

        Mat depth(Size(width, height), CV_16U, (void*)depth_frame.get_data(), Mat::AUTO_STEP);
        Mat frame(Size(640, 480), CV_8UC3, (void*)color_frame.get_data(), Mat::AUTO_STEP);
        const void* depth_data = depth_frame.get_data();
        
        Mat depth_float;
        depth.convertTo(depth_float, CV_8U, 255.0 / 2000.0);

        Mat colorizedDepth;
        applyColormap(depth_float, colorizedDepth);

        Mat hsv;
        cvtColor(frame,hsv,COLOR_BGR2HSV);


        Scalar lower(lowhuepadi, sminpadi, vminpadi);
		Scalar upper(upperhuepadi, smaxpadi, vmaxpadi);
     

        int minDistance = 500;
        // int maxDistance = 1000;
        int maxDistance = 1500;




        Mat mask;
        inRange(hsv,lower,upper,mask);

        Mat object_mask = depth > minDistance;//depth

        Mat object_mask2 = depth < maxDistance;

        Mat segmented_image = Mat::zeros(frame.size(), frame.type());
        frame.copyTo(segmented_image, object_mask);


        Mat segmented_image2 = Mat::zeros(frame.size(), frame.type());
        frame.copyTo(segmented_image2, object_mask2);

        Mat combined_mask2;
        bitwise_and(segmented_image, segmented_image2, combined_mask2);

        Mat combined_mask;
        bitwise_and(mask, object_mask2, combined_mask);


        erode(combined_mask, combined_mask, getStructuringElement(MORPH_RECT, Size(2, 2)));
        dilate(combined_mask, combined_mask, getStructuringElement(MORPH_RECT, Size(2, 2)));

        //morpologi close
        erode(combined_mask, combined_mask, getStructuringElement(MORPH_RECT, Size(2, 2)));
        dilate(combined_mask, combined_mask, getStructuringElement(MORPH_RECT, Size(2, 2)));

        vector<vector<Point>> contours;
        findContours(combined_mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        int contours_count = 0;

        for (size_t i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        double epsilon = 0.01 * arcLength(contours[i], true);
        vector<Point> approx;
        approxPolyDP(contours[i], approx, epsilon, true);

        if (200 < area && area < 10000) {
            //if (approx.size() > 5 && approx.size() < 15){
                
                // cout << "Ellipse" << endl;
            drawContours(frame, contours, (int)i, Scalar(0, 255, 0), 2);
            contours_count++;
            cout << contours_count << endl;

            Moments m = moments(contours[i]);
            Point center(m.m10 / m.m00, m.m01 / m.m00);
            circle(frame, center, 3, Scalar(255, 255, 255), -1);
            putText(frame, to_string(contours_count), center - Point(25, 25), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 2);
            string jumlah_con = "\njumlah (" + to_string(contours_count)+")";
            string messageData = "\nBarelang v" + jumlah_con;
            }
        }
        imshow("gabungan_Mask_Padi", combined_mask);
        //imshow("Segmented Image_Padi", segmented_image);
        //imshow("Segmented Image2_Padi", segmented_image2);
        imshow("Gabungan2_Padi", combined_mask2);
        imshow("frame_Padi", frame);
        imshow("mask_Padi", mask);
    
        keyVal = cv::waitKey(1) & 0xFF;

        if (keyVal == 113) { // 'Esc' u
            break;
        }
        else if (keyVal == 116) { // 't' key for snapshot
            snapshot = frame.clone();
            imshow("Snapshot_Padi", snapshot);
        }
        else if (keyVal == 117){
            destroyWindow("mask_Padi");
            destroyWindow("frame_Padi");
            destroyWindow("gabungan_Mask_Padi");
            destroyWindow("Gabungan2_Padi");
            
        }
       
    }
}

class MyKalmanFilter {
private:
    KalmanFilter kf;

public:
    MyKalmanFilter() {
        kf = KalmanFilter(4, 2);
        kf.measurementMatrix = (Mat_<float>(2, 4) << 1, 0, 0, 0, 0, 1, 0, 0);
        kf.transitionMatrix = (Mat_<float>(4, 4) << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1);
    }

    pair<int, int> predict(int coordX, int coordY) {
        Mat measured = (Mat_<float>(2, 1) << static_cast<float>(coordX), static_cast<float>(coordY));
        kf.correct(measured);
        Mat predicted = kf.predict();
        int x = static_cast<int>(predicted.at<float>(0));
        int y = static_cast<int>(predicted.at<float>(1));
        return make_pair(x, y);
    }
};

MyKalmanFilter kfWrapper;

std::clock_t startTime, endTime;



///////////////////////////////////////////////BOLA/////////////////////////////////////////////////////////////
int received_data = 0;

int received_data2 = 0;
// Fungsi callback untuk subscriber
void chatterCallback(const std_msgs::msg::Int32::SharedPtr msg)
{
    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Received: '%d'", msg->data);
    // Simpan nilai pesan yang diterima ke dalam variabel global
    received_data = msg->data;
}

void chatterCallback2(const std_msgs::msg::Int32::SharedPtr msg)
{
    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Received: '%d'", msg->data);
    // Simpan nilai pesan yang diterima ke dalam variabel global
    received_data2 = msg->data;
}


void bola(){   
    rclcpp::NodeOptions options;
    auto node = std::make_shared<rclcpp::Node>("Posisi_Bola", options);

    auto publisher = node->create_publisher<geometry_msgs::msg::Twist>("posisi_bola", 10);
       auto subscriber = node->create_subscription<std_msgs::msg::Int32>(
        "pemai", 10, chatterCallback);

 

    rclcpp::WallRate loop_rate(10);

    // while (rclcpp::ok) {
        startTime = clock();
        rclcpp::spin_some(node);
        rs2::frameset frames = pipekan.wait_for_frames();
        rs2::frame color_frame = frames.get_color_frame();
        rs2::frame depth_frame = frames.get_depth_frame();
        Mat frame(cv::Size(640, 480), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
        cout << "Received message value in loop: " << received_data << std::endl;
        //resize(frame,frame,Size(500,500));


        Mat depth_mat(Size(640, 480), CV_16UC1, (void*)depth_frame.get_data(), Mat::AUTO_STEP);
        const void* depth_data = depth_frame.get_data();

        pBackSub->apply(frame, fgMask);

        erode(fgMask, fgMask, getStructuringElement(MORPH_RECT, Size(3, 3)));
        dilate(fgMask, fgMask, getStructuringElement(MORPH_RECT, Size(3, 3)));

        
        endTime = clock();


        double seconds =  (double(endTime) - double(startTime) / double(CLOCKS_PER_SEC));
      


        fpsLive = double(num_frames) / seconds;

        keyVal = cv::waitKey(1) & 0xFF;
        if (keyVal == 113) { // 'q' key for exit
        //     break;
        // }
        // else if (keyVal == 114) { // 'r' key for reset
            resetObjects();
            destroyWindow("Color Object 1");
            destroyWindow("Color Object 2");
        }
        else if (keyVal == 116) { // 't' key for snapshot
            snapshot = frame.clone();
            imshow("Snapshot", snapshot);
        }
         
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        Mat hsvFrame1;
        cvtColor(frame, hsvFrame1, COLOR_BGR2HSV);

        cv::Mat hsvFrame2 = cv::Mat::zeros(hsvFrame1.size(), hsvFrame1.type());
        hsvFrame1.copyTo(hsvFrame2);

        Mat combinedFrame;

        //morpologi open
        erode(hsvFrame1, hsvFrame1, getStructuringElement(MORPH_RECT, Size(8, 8)));
        dilate(hsvFrame1, hsvFrame1, getStructuringElement(MORPH_RECT, Size(8, 8)));
        //morpologi close
        erode(hsvFrame1, hsvFrame1, getStructuringElement(MORPH_RECT, Size(8, 8)));
        dilate(hsvFrame1, hsvFrame1, getStructuringElement(MORPH_RECT, Size(8, 8)));
        
        
        //morpologi open
        erode(hsvFrame2, hsvFrame2, getStructuringElement(MORPH_RECT, Size(8, 8)));
        dilate(hsvFrame2, hsvFrame2, getStructuringElement(MORPH_RECT, Size(8, 8)));
        //morpologi close
        erode(hsvFrame2, hsvFrame2, getStructuringElement(MORPH_RECT, Size(8, 8)));
        dilate(hsvFrame2, hsvFrame2, getStructuringElement(MORPH_RECT, Size(8, 8)));


        Mat imghsv;
        cvtColor(frame, imghsv, COLOR_BGR2HSV);

        if (received_data == 1){
            nilai_bawah = lowhue1;
            saturasi_bawah = smin1;
            value_bawah = vmin1;

            nilai_atas = upperhue1;
            saturasi_atas = smax1;
            value_atas = vmax1;
        }
        else if(received_data == 2){
            nilai_bawah = lowhue2;
            saturasi_bawah = smin2;
            value_bawah = vmin2;

            nilai_atas = upperhue2;
            saturasi_atas = smax2;
            value_atas = vmax2;
        }


        Scalar lower1(lowhue1, smin1, vmin1);
		Scalar upper1(upperhue1, smax1, vmax1);


        Scalar lower2(lowhue2, smin2, vmin2);
		Scalar upper2(upperhue2, smax2, vmax2);


        Mat maskobject1;
        inRange(hsvFrame1, lower1, upper1, maskobject1);

        Mat maskobject2;
        inRange(hsvFrame2,lower2,upper2,maskobject2);


        vector<vector<Point>> contoursobject1;
        findContours(maskobject1, contoursobject1, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        vector<vector<Point>> contoursobject2;
        findContours(maskobject2, contoursobject2, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        Mat maskpembanding;
        inRange(imghsv, lower1, upper1, maskpembanding);

        combinedFrame = frame.clone();

        double maxArea = 0;
        int maxAreaIdx = -1;


        double maxArea2 = 0;
        int maxAreaIdx2 = -1;

        Point barang1;
        Point barang2;

        for (size_t i = 0; i < contoursobject1.size(); ++i) {
            double area = contourArea(contoursobject1[i]);
            if (area > maxArea) {
                maxArea = area;
                maxAreaIdx = static_cast<int>(i);
            }
        }
      
        // Check if a contour with sufficient area is found
        if (maxAreaIdx != -1 && maxArea > 100) {
            // Process the largest contour only
            const auto& contour = contoursobject1[maxAreaIdx];

            // Your existing code for processing the contour goes here
            Moments oMoments = moments(contour);
            double dM01 = oMoments.m01;
            double dM10 = oMoments.m10;
            double dArea = oMoments.m00;

            //Rectangle Object 1
            Rect boundingBox = boundingRect(contour);
            rectangle(combinedFrame, boundingBox, Scalar(255, 0, 0), 2);

            // Circle Object1
            Point circleCenter(boundingBox.x + boundingBox.width / 2, boundingBox.y + boundingBox.height / 2);
            int circleRadius = (boundingBox.width + boundingBox.height) / 4;
            circle(combinedFrame, circleCenter, circleRadius, Scalar(0, 0, 255), 2);

            // Point Object1
            Point objectCenter(boundingBox.x + boundingBox.width / 2, boundingBox.y + boundingBox.height / 2);
            Point frameCenter(combinedFrame.cols / 2, combinedFrame.rows);

            //Rectangle background object 1
            Size textSize = getTextSize("Object1", FONT_HERSHEY_SIMPLEX, 0.5, 2, nullptr);
            rectangle(combinedFrame, Point(boundingBox.x, boundingBox.y - textSize.height - 10), Point(boundingBox.x + textSize.width, boundingBox.y), Scalar(255, 0, 0), FILLED);

            //garis dan keterangan
            putText(combinedFrame, "Object1", Point(boundingBox.x, boundingBox.y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 2);
            line(combinedFrame, frameCenter, objectCenter, Scalar(0, 255, 0), 2);
            circle(combinedFrame, Point(objectCenter), 5, Scalar(100, 255, 100), FILLED);

            // Koordinat Object 1
            string coordinateText1 = "(" + to_string(objectCenter.x) + ", " + to_string(objectCenter.y) + ")";
            putText(combinedFrame, coordinateText1, Point(objectCenter.x - 20, objectCenter.y - 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(250, 50, 255), 2);

            //Rectangle background Depth
            rectangle(combinedFrame, Point(500, 430), Point(640, 470), Scalar(200, 0, 0), FILLED);
            //Rectangle Keterangan
            rectangle(combinedFrame, Point(550, 410), Point(640, 430), Scalar(0, 0, 0), FILLED);

            //Depth
            uint16_t distance = depth_mat.at<uint16_t>(circleCenter);
            putText(combinedFrame, to_string(distance) + "mm", Point(500, 460), FONT_HERSHEY_PLAIN, 2, Scalar(255, 255, 255), 2);
            //cout << "jarak Object 1 = " << distance << "mm" << endl;

            //Keterangan Depth
            putText(combinedFrame, "Object_1", Point(560, 425), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 2);

            // Prediksi Kalman Filter
            auto predicted = kfWrapper.predict(objectCenter.x, objectCenter.y);
            circle(combinedFrame, Point(predicted.first, predicted.second), 20, Scalar(255, 0, 0), 5);
            //cout << "Object 1 - Actual: (" << objectCenter.x << ", " << objectCenter.y << ")  Predicted: (" << predicted.first << ", " << predicted.second << ")" << endl;


            //Perhitungan SUDUT
            titik.push_back(Point(0,480));
            circle(combinedFrame,Point(0,480), 5, Scalar(255, 0, 0), -1);
   
            titik.push_back(Point(320, 480));
            circle(combinedFrame, Point(320, 480), 5, Scalar(255, 0, 0), -1);
            arrowedLine(combinedFrame, titik[1], titik[0], Scalar(255, 0, 0), 3);

            titik.push_back(objectCenter);
            circle(combinedFrame, objectCenter, 5, Scalar(255, 0, 0), -1);
            arrowedLine(combinedFrame, titik[1], objectCenter, Scalar(255, 0, 0), 3);

            double degrees = angle();
            //cout << "Angle: " << abs(degrees) << endl;

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
            fpsLive = double(num_frames) / seconds;
            //cout << "Estimated frames per second : " << fpsLive << endl;
            //putText(frame, "FPS: " + to_string(int(fpsLive)), { 50, 50 }, FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 0), 2);



            ostringstream angleText;
            angleText << fixed << setprecision(2) << angle;
            //string angleString = angleText.str();
            string angleString = "\nSudut Object (" + to_string(angle)+")";
            string fps = "\nFPS ("+ to_string(fpsLive)+")";
            // Ubah objek cv::Point objectCenter menjadi string dengan format yang sesuai
            string objectCenterString = "(" + to_string(objectCenter.x) + ", " + to_string(objectCenter.y) + ")";
            string distancel = "\nJarak Object (" + to_string(distance) + ")";
            // Gabungkan kedua string tersebut menjadi satu pesan
            string messageData = "\nBarelang V" + fps + posisi + angleString + distancel +" \nKoordinat Titik Tengah: ("+ objectCenterString + ")";
            // Masukkan pesan ke dalam message
            putText(frame, to_string(angle), Point(500,30), FONT_HERSHEY_DUPLEX, 0.8, Scalar(0, 0, 255), 2, LINE_AA);


             barang1 = objectCenter;

        }
       
        
        for (size_t i = 0; i < contoursobject2.size(); ++i) {
            double area2 = contourArea(contoursobject2[i]);
            if (area2 > maxArea2) {
                maxArea2 = area2;
                maxAreaIdx2 = static_cast<int>(i);
            }
        }
        if (maxAreaIdx2 != -1 && maxArea2 > 100) {
            // Process the largest contour for contoursobject2
            const auto& contour2 = contoursobject2[maxAreaIdx2];

            // Your existing code for processing the contour goes here
            Moments oMoments2 = moments(contour2);
            double dM012 = oMoments2.m01;
            double dM102 = oMoments2.m10;
            double dArea2 = oMoments2.m00;

            //Rectangle Object2
            Rect boundingBox2 = boundingRect(contour2);
            rectangle(combinedFrame, boundingBox2, Scalar(0, 0, 255), 2);

            // Circle Object2
            Point circleCenter2(boundingBox2.x + boundingBox2.width / 2, boundingBox2.y + boundingBox2.height / 2);
            int circleRadius2 = (boundingBox2.width + boundingBox2.height) / 4;
            circle(combinedFrame, circleCenter2, circleRadius2, Scalar(255, 0, 0), 2);

            // Point Object2
            Point objectCenter2(boundingBox2.x + boundingBox2.width / 2, boundingBox2.y + boundingBox2.height / 2);
            Point frameCenter2(combinedFrame.cols / 2, combinedFrame.rows);

            //Rectangle background object 2
            Size textSize2 = getTextSize("Object2", FONT_HERSHEY_SIMPLEX, 0.5, 2, nullptr);
            rectangle(combinedFrame, Point(boundingBox2.x, boundingBox2.y - textSize2.height - 10), Point(boundingBox2.x + textSize2.width, boundingBox2.y), Scalar(0, 0, 255), FILLED);

            //garis dan keterangan
            putText(combinedFrame, "Object2", Point(boundingBox2.x, boundingBox2.y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 2);
            line(combinedFrame, frameCenter2, objectCenter2, Scalar(100, 255, 100), 2);
            circle(combinedFrame, Point(objectCenter2), 5, Scalar(100, 255, 100), FILLED);

            // Koordinat Object 2
            string coordinateText2 = "(" + to_string(objectCenter2.x) + ", " + to_string(objectCenter2.y) + ")";
            putText(combinedFrame, coordinateText2, Point(objectCenter2.x - 20, objectCenter2.y - 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(200, 255, 0), 2);

            //Rectangle background Depth
            rectangle(combinedFrame, Point(0, 430), Point(140, 470), Scalar(0, 0, 255), FILLED);
            //Rectangle Keterangan
            rectangle(combinedFrame, Point(0, 410), Point(90, 430), Scalar(0, 0, 0), FILLED);

            //Keterangan Depth
            putText(combinedFrame,"Object_2", Point(0, 425), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 2);
            //Depth
            uint16_t distance = depth_mat.at<uint16_t>(circleCenter2);
            putText(combinedFrame, to_string(distance) + "mm", Point(0, 460), FONT_HERSHEY_PLAIN, 2, Scalar(255, 255, 255), 2);
            cout << "jarak Object 2 = " << distance << "mm" << endl;

           barang2 = objectCenter2;

        }

            geometry_msgs::msg::Twist message;

            message.linear.x = barang1.x;  // Kecepatan linier pada sumbu x
            message.linear.y = barang1.y;
            message.linear.z = 0;
            message.angular.x = barang2.x;  // Kecepatan linier pada sumbu x
            message.angular.y = barang2.y;  // Kecepatan linier pada sumbu y
            message.angular.z = 0;
            publisher->publish(message);

        //imshow("Frame_Bola", frame);
        imshow("Mask1_Bola", maskobject1);
        imshow("Mask2_Bola", maskobject2);
        imshow("Frame", combinedFrame);
        
        imshow("pembanding_Bola", maskpembanding);


    // }

}   

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);

    rclcpp::NodeOptions options;
    auto node = std::make_shared<rclcpp::Node>("Posisi_Bola", options);

    auto subscriber = node->create_subscription<std_msgs::msg::Int32>(
        "Command", 10, chatterCallback2);

    configs.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    configs.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    pipekan.start(configs);
    auto device = pipekan.get_active_profile().get_device();
    auto device_product_line = device.get_info(RS2_CAMERA_INFO_PRODUCT_LINE);
    auto video_stream_profile = pipekan.get_active_profile().get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
    double fps = video_stream_profile.fps();
    cout << "Frames per second camera : " << fps << endl;

    ofstream file("data.txt");
    //int num_frames = 1;

    if (!file.is_open()) {
        std::cerr << "Gagal membuka file!" << std::endl;
        return 1; // Keluar dari program dengan kode kesalahan
    }
   
    cout << "Capturing " << num_frames << " frames" << endl;
   
    Mat frame, colorArray,fgMask;
    snapshot = Mat(Size(640, 480), CV_8UC3, Scalar(0, 0, 0));
    resize(snapshot,snapshot,Size(500,500));
    imshow("Snapshot", snapshot);


    namedWindow("Trackbars", (640, 200));
	    createTrackbar("Sat Min", "Trackbars", &smin, 255);
	    createTrackbar("Sat Max", "Trackbars", &smax, 255);
	    createTrackbar("Val Min", "Trackbars", &vmin, 255);
	    createTrackbar("Val Max", "Trackbars", &vmax, 255);
        createTrackbar("Konsanta 1", "Trackbars", &konstanta1, 255);
	    createTrackbar("Konstanta 2", "Trackbars", &konstanta2, 255);
        createTrackbar("Threshold", "Processed Frame", &threshold_value, 255);

    

  bool isPadiRunning = false;
 rclcpp::WallRate loop_rate(10);


while (true) {
    rclcpp::spin_some(node);
     cout << "Received message value in loop: " << received_data2 << std::endl;

    keyVal = cv::waitKey(1) & 0xFF;
    int lowhue = hue - konstanta1;
    int upperhue = hue + konstanta1;
    
        if(received_data2 == 98){
            bola();
        }          
        else if (received_data2 == 111) {
            Silo();
        } 
            
               
               

    if (isPadiRunning && keyVal == 113) { 
        isPadiRunning = false;
        
        //break;
    }
    
    if(keyVal == 100){
        break;
    }
}

    file.close();
    // Tutup pipeline RealSense
    rclcpp::shutdown();
    pipekan.stop(); 
    destroyAllWindows();

    return 0;
}