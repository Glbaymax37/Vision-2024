#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <unistd.h>
#include <SDL2/SDL.h>
#include <thread>
#include <math.h>
#include <mutex>

#include <opencv2/opencv.hpp>  
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

//#include <csv/writer.hpp>

#include <bits/stdc++.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
//#include "iostream"
#include "time.h"
#include "string"
#include "cmath"
#include "cstring"
//#include "chrono"
#include <bits/stdc++.h>
//#include "gnuplot-iostream.h"

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"

#include <std_msgs/msg/int32.hpp>
#include "std_msgs/msg/float32.hpp"


#include "device.cpp"

//Device atas("192.168.0.70",5555);
Device roda1("192.168.0.1",5555);
Device roda2("192.168.0.2",5555);
Device roda3("192.168.0.3",5555);
Device roda4("192.168.0.4",5555);
Device IMU("192.168.0.5",5555);
Device megaatas("192.168.0.6",5555);

using namespace std;
using namespace std::this_thread;
using namespace std::chrono;

///////////////Update////////////////////////////////////////////
using namespace cv;

int thres_hold = 200;

int dialedx = 6;
int dialedy = 6;

int erodex = 4;
int erodey = 4;

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


/////////////////////////////////////////UpdateSIlo///////////////////////////////


#define R_robot 28.265
#define R_roda 13.6
#define Droda  0.1
#define PPR 200
#define MAXSPEED 0.1

#define PORT	 5555
#define MAXLINE 1024

//   Csv::Row header;
//   Csv::Writer *writer;


int sockfd;
struct sockaddr_in cli_addr;
std::string senderIP;
int sizeReceive;
char buffer[MAXLINE];
clock_t startcount;
char terima[MAXLINE];

float poslinearx;
float poslineary;
float poslinearz;
float posangularx;
float posangulary;
float posangularz;

float red_linear_x_;
float red_linear_y_;
float red_angular_z_;

float blue_linear_x_;
float blue_linear_y_;
float blue_angular_z_;

float purple_linear_x_;
float purple_linear_y_;
float purple_angular_z_;



float der;
float camx;
float camy;
float der2;
float camx2;
float camy2;
////////////baru//////////////////
const int angleM1 = 45;
const int angleM2 = 135;
const int angleM3 = 225;
const int angleM4 = 315;

float lx,ly,lt,XFilt,YFilt,TFilt,
	  deltaTime,passX,passY,errT,errX,errY,
	  v1,v2,v3,v4,uPrevT,uPrevX,uPrevY,x,y,t;
float prepos1, prepos2, prepos3, prepos4;

volatile long pos1, pos2, pos3, pos4;

float px, py;
float errorx, errory;
float pidx, pidy;
int imu;
int imu1;
int kec1, kec2, kec3, kec4;

float kpx = 0.030;
float kix = 0.005;
float kdx = 0;

float kpi = 0.0005;

float kp = 0.21;
float ki = 0;
float kd = 0;

float kps = 0.1; //0.1
float kis = 0;
float kds = 0;

float error, errors;
float kecsudut;
int j = 0;

float sonic;
float selisih;
int orientasi;

int angz;
int keyVal;
float orien;

int k = 0;
int e = 0;
float b;

class GeometryPublisher : public rclcpp::Node {
public:
  GeometryPublisher() : Node("bola2") {
    publisher_ = this->create_publisher<geometry_msgs::msg::Twist>("Command", 10);

    // red_subscriber_ = this->create_subscription<geometry_msgs::msg::Twist>(
    //     "Posisi_Bola", 10, std::bind(&GeometryPublisher::twist_callback1, this, std::placeholders::_1));

    // blue_subscriber_ = this->create_subscription<geometry_msgs::msg::Twist>(
    //     "Blue_ball", 10, std::bind(&GeometryPublisher::twist_callback2, this, std::placeholders::_1));

    // purple_subscriber_ = this->create_subscription<geometry_msgs::msg::Twist>(
    //     "Silo", 10, std::bind(&GeometryPublisher::twist_callback3, this, std::placeholders::_1));

    timer_ = this->create_wall_timer(std::chrono::milliseconds(1),
                                     std::bind(&GeometryPublisher::publishMessage, this));
  }


private:
  
  void publishMessage() {
    auto message = geometry_msgs::msg::Twist();
    message.angular.z = angz;

    // Set message values based on saved data for each topic
    // if (red_linear_x_ != 0.0f || red_linear_y_ != 0.0f || red_angular_z_ != 0.0f) {
    //   message.linear.x = red_linear_x_;
    //   message.linear.y = red_linear_y_;
    //   message.angular.z = red_angular_z_;
    // } else if (blue_linear_x_ != 0.0f || blue_linear_y_ != 0.0f || blue_angular_z_ != 0.0f) {
    //   message.linear.x = blue_linear_x_;
    //   message.linear.y = blue_linear_y_;
    //   message.angular.z = blue_angular_z_;
    // } else if (purple_linear_x_ != 0.0f || purple_linear_y_ != 0.0f || purple_angular_z_ != 0.0f) {
    //   message.linear.x = purple_linear_x_;
    //   message.linear.y = purple_linear_y_;
    //   message.angular.z = purple_angular_z_;
    // }

    publisher_->publish(message);
  }

//   void twist_callback1(const geometry_msgs::msg::Twist::SharedPtr msg) {
//     red_linear_x_ = msg->linear.x;
//     red_linear_y_ = msg->linear.y;
//     red_angular_z_ = msg->angular.z;
//     // RCLCPP_INFO(this->get_logger(), "Received Red_ball Twist message");
//   }

//   void twist_callback2(const geometry_msgs::msg::Twist::SharedPtr msg) {
//     blue_linear_x_ = msg->linear.x;
//     blue_linear_y_ = msg->linear.y;
//     blue_angular_z_ = msg->angular.z;
//     // RCLCPP_INFO(this->get_logger(), "Received Blue_ball Twist message");
//   }

//   void twist_callback3(const geometry_msgs::msg::Twist::SharedPtr msg) {
//     purple_linear_x_ = msg->linear.x;
//     purple_linear_y_ = msg->linear.y;
//     purple_angular_z_ = msg->angular.z;
//     // RCLCPP_INFO(this->get_logger(), "Received Silo Twist message");
//   }

//   rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr red_subscriber_;
//   rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr blue_subscriber_;
//   rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr purple_subscriber_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
};



class RadiusSubscriber : public rclcpp::Node {
public:
    RadiusSubscriber() : Node("radius_subscriber") {
        subscription_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "koordinat_03", 10, std::bind(&RadiusSubscriber::handleMessage, this, std::placeholders::_1));
    }

    // Getter function to access the last received message
    geometry_msgs::msg::Twist::SharedPtr getLastReceivedMessage() const {
        return last_received_message_;
    }
	double getLinearX() const {
        return poslinearx;
    }
	double getLinearY() const {
        return poslineary;
    }
	double getLinearZ() const {
        return poslinearz;
    }
	double getAngularX() const {
        return posangularx;
    }
	double getAngularY() const {
        return posangulary;
    }
	double getAngularZ() const {
        return posangularz;
    }
private:
    void handleMessage(const geometry_msgs::msg::Twist::SharedPtr msg) {
        // Note: Do not use std::cout here
        // RCLCPP_INFO(this->get_logger(), "Received: Max Radius = %.2f, Coordinates = (%.2f, %.2f)",
        //             msg->angular.x, msg->linear.x, msg->linear.y);
		poslinearx = msg->linear.x;
		poslineary = msg->linear.y;
		poslinearz = msg->linear.z;
		posangularx = msg->angular.x;
		posangulary = msg->angular.y;
		posangularz = msg->angular.z;
        // Store the received message
        last_received_message_ = msg;

    }

    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr subscription_;
    geometry_msgs::msg::Twist::SharedPtr last_received_message_;
};

class RealSensecam : public rclcpp::Node {
public:
    RealSensecam() : Node("realsense_cam") {
        subscription_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "posisi_bola", 10, std::bind(&RealSensecam::handleMessage, this, std::placeholders::_1));
    }

    // Getter function to access the last received message
    geometry_msgs::msg::Twist::SharedPtr getLastReceivedMessage() const {
        return last_received_message_;
    }

	double getder() const {
        return der;
    }
		double getcamx() const {
        return camx;
    }
		double getcamy() const {
        return camy;
    }


	double getder2() const {
        return der2;
    }
		double getcamx2() const {
        return camx2;
    }
		double getcamy2() const {
        return camy2;
    }
private:
    void handleMessage(const geometry_msgs::msg::Twist::SharedPtr msg) {
        // Note: Do not use std::cout here
        // RCLCPP_INFO(this->get_logger(), "Received: Max Radius = %.2f, Coordinates = (%.2f, %.2f)",
        //             msg->angular.x, msg->linear.x, msg->linear.y);
        // RCLCPP_INFO(this->get_logger(), "Publishing: Angular = %.2f",
        //          msg->angular.y);
        //der = msg->linear.x;
        der = msg->linear.z;
		camx =  msg->linear.x;
		camy =  msg->linear.y;

		der2 = msg->angular.z;
		camx2 =  msg->angular.x;
		camy2 =  msg->angular.y;
        // Store the received message
        last_received_message_ = msg;
    }

    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr subscription_;
    geometry_msgs::msg::Twist::SharedPtr last_received_message_;
};

void terimaros(){

	while (rclcpp::ok()) {
		auto subscriber_node = std::make_shared<RadiusSubscriber>();
        // Spin the node without blocking to allow handling of any available messages
        rclcpp::spin_some(subscriber_node);
		auto last_received_message = subscriber_node->getLastReceivedMessage();
		std::cout << "angular.x: " << posangularx << "angular.y: " << posangulary << "angular.z: " << posangularz << std::endl;
		std::cout << "linear.x: " << poslinearx << "linear.y: " << poslineary << "linear.z: " << poslinearz << std::endl;
		}
}

void wheel1 (float velo){
roda1.terimaData(sockfd);
pos1 = -roda1.en1;
}
void wheel2 (float velo){
roda2.terimaData(sockfd);	
pos2 = roda2.en2;
// pos2 = 0;
}
void wheel3 (float velo){
roda3.terimaData(sockfd);
pos3 = roda3.en3;
}
void wheel4 (float velo){
roda4.terimaData(sockfd);
pos4 = roda4.en4;
}
void mekaatas (int mode){
// megaatas.terimaData(sockfd);
megaatas.kirimData(sockfd, std::to_string(mode));
// sonic = megaatas.ultra;
}
void Imu(){
	IMU.terimaData(sockfd);
	// imu1 = (IMU.imuDir*10);
	// imu = float(imu1)/10;	
	imu = IMU.imuDir;
	if (imu < -180){
		imu = imu + 360;
	}
}

void berhenti(){
	for(int i = 0; i < 10; i++){
	roda1.kirimData(sockfd, std::to_string(0));
	roda2.kirimData(sockfd, std::to_string(0));
	roda3.kirimData(sockfd, std::to_string(0));
	roda4.kirimData(sockfd, std::to_string(0));
	sleep_for(microseconds(1));
	}
}

void maju (float ka, float ki, float Tim){
		std::chrono::high_resolution_clock clock;
    auto lastTime = clock.now();
	while (rclcpp::ok()){
	auto currentTime = clock.now();
    std::chrono::duration<float> deltaTime = (currentTime - lastTime);
    float deltaTimeSeconds = deltaTime.count();
    float deltaT = deltaTime.count();  
	roda1.kirimData(sockfd, std::to_string(fmax(-150,fmin(150,-ka))));
	roda2.kirimData(sockfd, std::to_string(fmax(-150,fmin(150,-ki))));
	if(deltaT > Tim){
		lastTime = currentTime;
		break;
	}
	}
}
void fuzzywheel (float p, float arah){
	float pos = -p/5;
	std::chrono::high_resolution_clock clock;
    auto lastTime = clock.now();
	maju (60, 60, 0.1);
	while (rclcpp::ok()){
	auto currentTime = clock.now();
    std::chrono::duration<float> deltaTime = (currentTime - lastTime);
    float deltaTimeSeconds = deltaTime.count();
    float deltaT = deltaTime.count();  
	lastTime = currentTime;

	std::thread th1(wheel1, 0);
	std::thread th2(wheel2, 0);
	std::thread th5(Imu);
	th1.join();
	th2.join();
	th5.join();
	// L = 37
	// R = 6
	float posisi = 6*((pos1 - prepos1) + (pos2 - prepos2))/(2*2020/(3.14));
	float sudut = 6*(pos1 - pos2)/(2*3840/(3.14));
	// error = pos - posisi; 
	//errors = arah - sudut;
	// if ((arah - imu) != 0) errors += 0.07*(arah + imu)*deltaT;
	// if (pos > 0) {
	// 	imu = -imu;
	// 	//orien = - orien;
	// }
	errors = (arah + imu - orien);
	if (pos > 0) {
		errors = - errors;
		//orien = - orien;
	}
	error = pos - cos(toRad(errors))*(posisi); 
	float jarak;
	float kecsudut;
	// x = cos(arah)*error;
	// y = sin(arah)*error;
	if (abs(error) < 0.5){
		berhenti();
		break;
	}
	else if (abs(error) >= 0.5 && abs(error) < 1){
	jarak = 0.5*error/abs(error);
	//kecsudut = fmax(-jarak,fmin(jarak, 0.1*errors*jarak));
	kecsudut = 0.02*errors*jarak;
	}
	else if (abs(error) >= 1 && abs(error) < 3){
	jarak = 0.7*error/abs(error);
	//kecsudut = fmax(-jarak,fmin(jarak, 0.1*errors*jarak));
	kecsudut = 0.02*errors*jarak;
	}
	else{
	//0.8 = 58/50 || 0.9 = 56/50
	jarak = 1*error/abs(error);  //0.95
	//kecsudut = 6*sin(toRad(errors))*jarak;
	kecsudut = 0.02*errors*jarak;
	}
	float motor1 = (jarak + kecsudut)*(3840/(3.14))/12;
	float motor2 = (jarak - kecsudut)*(3840/(3.14))/12;
	// float motor1 = (2*jarak - kecsudut*37)*(3840/(3.14))/12;
	// float motor2 = (2*jarak + kecsudut*37)*(3840/(3.14))/12;

	roda1.kirimData(sockfd, std::to_string(fmax(-150,fmin(150,motor1*65/60))));
	roda2.kirimData(sockfd, std::to_string(fmax(-150,fmin(150,motor2))));

	std::cout <<"v1 "<< motor1 <<" v2 "<< motor2 << std::endl;
	std::cout <<"pos1 "<< pos1 <<" pos2 "<< pos2 << std::endl;
	std::cout <<"jarak "<< jarak <<std::endl;
	std::cout <<"sudut "<< sudut <<std::endl;
	std::cout <<"kecsudut "<< kecsudut <<std::endl;
	std::cout <<"imu "<< imu <<std::endl;
	std::cout <<"error "<< error <<std::endl;
	std::cout <<"orien "<< orien <<std::endl;
	std::cout <<"posisi "<< posisi <<std::endl;
	std::cout <<"prepos1 "<< prepos1 <<std::endl;
	std::cout <<"prepos2 "<< prepos2 <<std::endl;
	// x = cos(arah)*jarak;
	// y = sin(arah)*jarak;
	}
}


void fuzzymaju (float p, float arah){
	float pos = p;
	std::chrono::high_resolution_clock clock;
    auto lastTime = clock.now();
	maju (60, 60, 0.1);
	while (rclcpp::ok()){
	auto currentTime = clock.now();
    std::chrono::duration<float> deltaTime = (currentTime - lastTime);
    float deltaTimeSeconds = deltaTime.count();
    float deltaT = deltaTime.count();  
	lastTime = currentTime;

	std::thread th1(wheel1, 0);
	std::thread th2(wheel2, 0);
	std::thread th5(Imu);
	th1.join();
	th2.join();
	th5.join();
	// L = 37
	// R = 6
	float posisi = 6*((pos1 - prepos1) + (pos2 - prepos2))/(2*3840/(3.14));
	float sudut = 6*(pos1 - pos2)/(2*3840/(3.14));
	// error = pos - posisi; 
	//errors = arah - sudut;
	// if ((arah - imu) != 0) errors += 0.07*(arah + imu)*deltaT;
	// if (pos > 0) {
	// 	imu = -imu;
	// 	//orien = - orien;
	// }
	errors = (arah + imu - orien);
	if (pos > 0) {
		errors = - errors;
		//orien = - orien;
	}
	error = pos - cos(toRad(errors))*(posisi); 
	float jarak;
	float kecsudut;
	// x = cos(arah)*error;
	// y = sin(arah)*error;
	if (abs(error) < 0.5){
		berhenti();
		break;
	}
	else if (abs(error) >= 0.5 && abs(error) < 1){
	jarak = 0.5*error/abs(error);
	//kecsudut = fmax(-jarak,fmin(jarak, 0.1*errors*jarak));
	kecsudut = 0.02*errors*jarak;
	}
	else if (abs(error) >= 1 && abs(error) < 3){
	jarak = 0.7*error/abs(error);
	//kecsudut = fmax(-jarak,fmin(jarak, 0.1*errors*jarak));
	kecsudut = 0.02*errors*jarak;
	}
	else{
	//0.8 = 58/50 || 0.9 = 56/50
	jarak = 0.96*error/abs(error);
	//kecsudut = 6*sin(toRad(errors))*jarak;
	kecsudut = 0.02*errors*jarak;
	}
	float motor1 = (jarak - kecsudut)*(3840/(3.14))/12;
	float motor2 = (jarak + kecsudut)*(3840/(3.14))/12;
	// float motor1 = (2*jarak - kecsudut*37)*(3840/(3.14))/12;
	// float motor2 = (2*jarak + kecsudut*37)*(3840/(3.14))/12;

	roda1.kirimData(sockfd, std::to_string(fmax(-150,fmin(150,-motor1*90/60))));
	roda2.kirimData(sockfd, std::to_string(fmax(-150,fmin(150,-motor2))));

	std::cout <<"v1 "<< motor1 <<" v2 "<< motor2 << std::endl;
	std::cout <<"pos1 "<< pos1 <<" pos2 "<< pos2 << std::endl;
	std::cout <<"jarak "<< jarak <<std::endl;
	std::cout <<"sudut "<< sudut <<std::endl;
	std::cout <<"kecsudut "<< kecsudut <<std::endl;
	std::cout <<"imu "<< imu <<std::endl;
	std::cout <<"error "<< error <<std::endl;
	std::cout <<"orien "<< orien <<std::endl;
	std::cout <<"posisi "<< posisi <<std::endl;
	std::cout <<"prepos1 "<< prepos1 <<std::endl;
	std::cout <<"prepos2 "<< prepos2 <<std::endl;
	// x = cos(arah)*jarak;
	// y = sin(arah)*jarak;
	}
}

void putar (float hadap){

	float atas, bawah;

	while(rclcpp::ok()){
	std::thread th1(wheel1, 0);
	std::thread th2(wheel2, 0);
	std::thread th5(Imu);
	th1.join();
	th2.join();
	th5.join();

	errors = (hadap - imu);
	kecsudut = errors*5;
	// float ers = fmax(70, errors*0.9);
	float ers;
	if (abs(errors) < 2){
		std::cout <<"done "<<std::endl;
		berhenti();
		break;
	}
	else if (abs(errors) >= 2 && abs(errors) < 10){
		ers = 64;
	}
	else if (abs(errors) >= 10 && abs(errors) < 30){
		ers = 65;
	}
	else if (abs(errors) >= 30 && abs(errors) < 50){
		ers = 65;
	}
	else if (abs(errors) >= 50 && abs(errors) < 80){
		ers = 65;
	}
	else {
		ers = 75;
	}
	float motor1 = (kecsudut);
	float motor2 = (-kecsudut);

	if (errors < 0){
		ers = -ers;
	roda1.kirimData(sockfd, std::to_string(fmin(-80,fmax(-90,motor1))));
	roda2.kirimData(sockfd, std::to_string(fmax(80,fmin(90,motor2))));
	}
	else{
	roda1.kirimData(sockfd, std::to_string(fmax(80,fmin(90,motor1))));
	roda2.kirimData(sockfd, std::to_string(fmin(-80,fmax(-90,motor2))));
	}

	std::cout <<"imu "<< imu <<std::endl;
	std::cout <<"error "<< errors <<std::endl;
	std::cout <<"motoor1 "<< motor1 <<std::endl;
	std::cout <<"motor2 "<< motor2 <<std::endl;

	}
}

void putar3 (float hadap){

	float atas, bawah;

	while(rclcpp::ok()){
	std::thread th1(wheel1, 0);
	std::thread th2(wheel2, 0);
	std::thread th5(Imu);
	th1.join();
	th2.join();
	th5.join();

	errors = (hadap - imu);
	kecsudut = errors*5;
	// float ers = fmax(70, errors*0.9);
	float ers;
	if (abs(errors) < 2){
		std::cout <<"done "<<std::endl;
		berhenti();
		break;
	}
	else if (abs(errors) >= 2 && abs(errors) < 10){
		ers = 64;
	}
	else if (abs(errors) >= 10 && abs(errors) < 30){
		ers = 65;
	}
	else if (abs(errors) >= 30 && abs(errors) < 50){
		ers = 65;
	}
	else if (abs(errors) >= 50 && abs(errors) < 80){
		ers = 65;
	}
	else {
		ers = 75;
	}
	float motor1 = (kecsudut);
	float motor2 = (-kecsudut);

	if (errors < 0){
		ers = -ers;
	roda1.kirimData(sockfd, std::to_string(fmin(-60,fmax(-110,motor1))));
	roda2.kirimData(sockfd, std::to_string(fmax(60,fmin(110,motor2))));
	}
	else{
	roda1.kirimData(sockfd, std::to_string(fmax(60,fmin(110,motor1))));
	roda2.kirimData(sockfd, std::to_string(fmin(-60,fmax(-110,motor2))));
	}

	std::cout <<"imu "<< imu <<std::endl;
	std::cout <<"error "<< errors <<std::endl;
	std::cout <<"motoor1 "<< motor1 <<std::endl;
	std::cout <<"motor2 "<< motor2 <<std::endl;

	}
}

void putarsilo (float hadap, int lock){

	while(rclcpp::ok()){
	std::thread th1(wheel1, 0);
	std::thread th2(wheel2, 0);
	std::thread th5(Imu);
	th1.join();
	th2.join();
	th5.join();

	errors = (hadap - imu);
	kecsudut = errors*12;
	// float ers = fmax(70, errors*0.9);
	float ers;
	int ps1 = 0, ps2 = 0;
	if (lock == 1 ){
		ps1 = pos1;
		ps2 = pos2;
	}
	megaatas.kirimData(sockfd, std::to_string(5));
	if (abs(errors) < 2 || ((abs(ps1) == 1 || abs(ps2) == 1))){
		std::cout <<"done "<<std::endl;
		//berhenti();
		break;
	}
	else if (abs(errors) >= 2 && abs(errors) < 10){
		ers = 64;
	}
	else if (abs(errors) >= 10 && abs(errors) < 30){
		ers = 65;
	}
	else if (abs(errors) >= 30 && abs(errors) < 50){
		ers = 65;
	}
	else if (abs(errors) >= 50 && abs(errors) < 80){
		ers = 65;
	}
	else {
		ers = 75;
	}
	float motor1 = (kecsudut);
	float motor2 = (-kecsudut);

	if (errors < 0){
		ers = -ers;
	}
	roda1.kirimData(sockfd, std::to_string(fmax(-120,fmin(120,motor1))));
	roda2.kirimData(sockfd, std::to_string(fmax(-120,fmin(120,motor2))));

	std::cout <<"imu "<< imu <<std::endl;
	std::cout <<"error "<< errors <<std::endl;
	std::cout <<"motor1 "<< motor1 <<std::endl;
	std::cout <<"motor2 "<< motor2 <<std::endl;
	std::cout <<"pos1 "<< pos1 <<std::endl;
	std::cout <<"pos2 "<< pos2 <<std::endl;
	}
}
void bolasilo (RadiusSubscriber::SharedPtr noders, int col){

	auto node = std::make_shared<GeometryPublisher>();
	std::chrono::high_resolution_clock clock;
	auto lastTime = clock.now();

	while (rclcpp::ok()) 
	{
	angz = col;
	sleep_for(microseconds(10));
	rclcpp::spin_some(node);
	// //std::cout <<"error "<<std::endl;

	rclcpp::spin_some(noders);
	// auto last_received_messagers = std::make_shared<RealSensecam>()->getLastReceivedMessage();
	float linxrs = std::make_shared<RealSensecam>()->getcamx();
	float angxrs = std::make_shared<RealSensecam>()->getcamx2();
	std::cout << "linear.x: " << linxrs << std::endl;

	std::thread th1(wheel1, 0);
	std::thread th2(wheel2, 0);
	std::thread th5(Imu);
	th1.join();
	th2.join();
	th5.join();

	orientasi = linxrs;
	selisih = orientasi*12;

	// float motor1 = (selisih)*(3840/(3.14))/12;
	// float motor2 = (-selisih)*(3840/(3.14))/12;

	float motor1 = (selisih);
	float motor2 = (-selisih);

	roda1.kirimData(sockfd, std::to_string(fmax(-90,fmin(90,motor1))));
	roda2.kirimData(sockfd, std::to_string(fmax(-90,fmin(90,motor2))));
	
	std::cout <<"imu "<< linxrs <<std::endl;
	std::cout <<"error "<< selisih <<std::endl;
	std::cout <<"angz "<< angz <<std::endl;
	if ((abs(linxrs) < 5 && abs(linxrs) > 0 ) || abs(linxrs) > 10 ||  (abs(pos1) == 1 || abs(pos2) == 1 )){
		std::cout <<"done "<<std::endl;	
		//berhenti();
		break;
	}
	}

}

void bola2 (RadiusSubscriber::SharedPtr noders, int col){

	auto node = std::make_shared<GeometryPublisher>();
	std::chrono::high_resolution_clock clock;
	auto lastTime = clock.now();

	while (rclcpp::ok()) 
	{
	angz = col;
	sleep_for(microseconds(10));
	rclcpp::spin_some(node);
	// //std::cout <<"error "<<std::endl;

	rclcpp::spin_some(noders);
	// auto last_received_messagers = std::make_shared<RealSensecam>()->getLastReceivedMessage();
	float linxrs = std::make_shared<RealSensecam>()->getcamx();
	float angxrs = std::make_shared<RealSensecam>()->getcamx2();
	std::cout << "linear.x: " << linxrs << std::endl;

	std::thread th1(wheel1, 0);
	std::thread th2(wheel2, 0);
	std::thread th5(Imu);
	th1.join();
	th2.join();
	th5.join();

	orientasi = linxrs;
	selisih = orientasi*14;

	// float motor1 = (selisih)*(3840/(3.14))/12;
	// float motor2 = (-selisih)*(3840/(3.14))/12;

	float motor1 = (selisih);
	float motor2 = (-selisih);

	roda1.kirimData(sockfd, std::to_string(fmax(-90,fmin(90,motor1))));
	roda2.kirimData(sockfd, std::to_string(fmax(-90,fmin(90,motor2))));
	
	std::cout <<"imu "<< linxrs <<std::endl;
	std::cout <<"error "<< selisih <<std::endl;
	std::cout <<"angz "<< angz <<std::endl;
	if (abs(linxrs) < 2.5 && abs(linxrs) > 0 ){
		std::cout <<"done "<<std::endl;	
		berhenti();
		break;
	}
	}

}


void ulang (){
	orien = imu;
	prepos1 = pos1;
	prepos2 = pos2;
}

void tunggu (RadiusSubscriber::SharedPtr noders){
	auto node = std::make_shared<GeometryPublisher>();
	while (rclcpp::ok() ) 
	{
	angz = 12;
	sleep_for(microseconds(10));
	rclcpp::spin_some(node);
	rclcpp::spin_some(noders);
	//auto last_received_messagers = std::make_shared<RealSensecam>()->getLastReceivedMessage();
	float angyrs = std::make_shared<RealSensecam>()->getcamx();


	roda1.kirimData(sockfd, std::to_string(0));
	roda2.kirimData(sockfd, std::to_string(0));
	
	if (angyrs > 3 ){
		std::cout <<"0k "<<std::endl;
		berhenti();
		angz = 0;
		break;
	}
	}
}


void ambil (RadiusSubscriber::SharedPtr noders, int col, int meka){

	float max;
	float kecepatan;
	auto node = std::make_shared<GeometryPublisher>();
	auto realsensenode = std::make_shared<RealSensecam>();
	std::chrono::high_resolution_clock clock;
	auto lastTime = clock.now();
	// int p = 0;
	maju(60, 60, 0.1);
	int ps1 = 0, ps2 = 0;
	while (rclcpp::ok() ) 
	{
	int mega = 0;
	angz = col;
	sleep_for(microseconds(10));
	rclcpp::spin_some(node);
	rclcpp::spin_some(noders);
	// auto last_received_messagers = std::make_shared<RealSensecam>()->getLastReceivedMessage();
	float linxrs = std::make_shared<RealSensecam>()->getcamx();
	float linyrs = std::make_shared<RealSensecam>()->getcamy();

	float angxrs = std::make_shared<RealSensecam>()->getcamx2();
	float angyrs = std::make_shared<RealSensecam>()->getcamy2();
	std::cout << "linear.x: " << linxrs << std::endl;
	std::cout << "linear.y: " << linyrs << std::endl;
	std::cout << "angular.x: " << angxrs << std::endl;
	std::cout << "angular.y: " << angyrs << std::endl;

	std::thread th1(wheel1, 0);
	std::thread th2(wheel2, 0);
	std::thread th5(Imu);
	// std::thread th6();
	th1.join();
	th2.join();
	th5.join();
	// th6.join();

	if (angz == 98){
		max = 420;
		kecepatan = fmin(0.4*400/fmax(1,linyrs), 0.8);
		megaatas.kirimData(sockfd, std::to_string(4));
	}
	else{
		max = 8000;
		kecepatan = 0.8;
		ps1 = pos1;
		ps2 = pos2;
	}
	
	orientasi = linxrs;
	selisih = orientasi*0.06;
	float posisi = 6*((pos1 - prepos1) + (pos2 - prepos2))/(2*3840/(3.14));
	
	float motor1 = (-kecepatan - selisih)*(3840/(3.14))/12;
	float motor2 = (-kecepatan + selisih)*(3840/(3.14))/12;

	// if (selisih == 0)
	// {
		
	// roda1.kirimData(sockfd, std::to_string(fmax(-150,fmin(150,60))));
	// roda2.kirimData(sockfd, std::to_string(fmax(-150,fmin(150,-60))));
	// }
	if(angz != 98){
			bola2 (realsensenode, 111);
			maju(-100, -100, 0.3);
			putarsilo(-75, 1);
			maju(-100, -100, 0.3);
			if(abs(pos1) == 1 && abs(pos2) == 1){
				for(int p = 0; p < 20; p++){
					roda1.kirimData(sockfd, std::to_string(fmax(-120,fmin(120,0))));
					roda2.kirimData(sockfd, std::to_string(fmax(-120,fmin(120,0))));
					megaatas.kirimData(sockfd, std::to_string(meka));
					sleep_for(microseconds(1));
					berhenti();
		
					}		
				break;
			}
			else if(linyrs < 130){
				bola2 (realsensenode, 111);
				maju(-120, -120, 0.3);
			}
		}
	else{
	roda1.kirimData(sockfd, std::to_string(fmax(-120,fmin(120,-motor1))));
	roda2.kirimData(sockfd, std::to_string(fmax(-120,fmin(120,-motor2))));
	}
	std::cout <<"imuambil "<< linyrs <<std::endl;
	std::cout <<"errorambil "<< selisih <<std::endl;
	std::cout <<"angzambil "<< angz <<std::endl;
	std::cout <<"sonic "<< sonic <<std::endl;
	// p += 1;
	// if (linyrs > max && (abs(pos1) == 1 || abs(pos2) == 1)){
	if (linyrs > max || abs(ps1) == 1 || abs(ps2) == 1){
		std::cout <<"done "<<std::endl;
		// mekaatas(1);
		berhenti();
		if(angz == 98){
		// maju(-60, -60, 0.5);
		berhenti();
		}
		// maju(60, 60, 0.1);
		for(int p = 0; p < 2; p++){
		roda1.kirimData(sockfd, std::to_string(fmax(-120,fmin(120,0))));
		roda2.kirimData(sockfd, std::to_string(fmax(-120,fmin(120,0))));
		megaatas.kirimData(sockfd, std::to_string(meka));
		sleep_for(microseconds(1));
		}
		while (rclcpp::ok()){ 
		megaatas.terimaData(sockfd);
		mega = megaatas.ultra;
		roda1.kirimData(sockfd, std::to_string(fmax(-120,fmin(120,0))));
		roda2.kirimData(sockfd, std::to_string(fmax(-120,fmin(120,0))));
		sleep_for(microseconds(1));
		if (mega == 2){
			mega = 0;
			break;
		}
		else{
			megaatas.kirimData(sockfd, std::to_string(1));
		}
		}
		berhenti();
		// for(int p = 0; p < 10; p++){
		// roda1.kirimData(sockfd, std::to_string(fmax(-120,fmin(120,0))));
		// roda2.kirimData(sockfd, std::to_string(fmax(-120,fmin(120,0))));
		// megaatas.kirimData(sockfd, std::to_string(3));
		// sleep_for(microseconds(1));
		// berhenti();
		// }
		if(angz != 98){
			
			// putarsilo(170);
			// maju(-80, -80, 0.5);
		for(int p = 0; p < 20; p++){
		roda1.kirimData(sockfd, std::to_string(fmax(-120,fmin(120,0))));
		roda2.kirimData(sockfd, std::to_string(fmax(-120,fmin(120,0))));
		megaatas.kirimData(sockfd, std::to_string(5));
		sleep_for(microseconds(1));
		berhenti();
		
		}
		maju(80, 80, 1);
		}
		// berhenti();
		break;
	}
	else if (angyrs > max && linyrs < max){
		std::cout <<"nope "<<std::endl;
		// std::cout <<"diam "<<std::endl;
		if(angz == 98){
		
		megaatas.kirimData(sockfd, std::to_string(5));
		// for(int p = 0; p < 1; p++){
		// roda1.kirimData(sockfd, std::to_string(fmax(-120,fmin(120,0))));
		// roda2.kirimData(sockfd, std::to_string(fmax(-120,fmin(120,0))));
		// megaatas.kirimData(sockfd, std::to_string(3));
		// sleep_for(microseconds(1));
		// }
		maju(-150, 150, 1);
		maju(0, 0, 0.5);
		megaatas.kirimData(sockfd, std::to_string(5));
		maju(200, 200, 1.3);
		maju(0, 0, 0.5);
		// maju(150, -150, 0.8);
		//maju(150, -150, 1.2);
		putar(80);
		maju(0, 0, 0.5);
		// int c = 0;
		// while (rclcpp::ok()){ 
		// // megaatas.kirimData(sockfd, std::to_string(3));
		// megaatas.terimaData(sockfd);
		// mega = megaatas.ultra;
		// roda1.kirimData(sockfd, std::to_string(fmax(-120,fmin(120,0))));
		// roda2.kirimData(sockfd, std::to_string(fmax(-120,fmin(120,0))));
		// sleep_for(microseconds(1));
		// // c = c + 1;
		// if (mega == 1){
		// 	mega = 0;
		// 	megaatas.kirimData(sockfd, std::to_string(6));
		// 	sleep_for(microseconds(1));
		// 	// int diam = 0;
		// 	// while (rclcpp::ok()){ 
		// 	// // megaatas.kirimData(sockfd, std::to_string(3));
		// 	// megaatas.terimaData(sockfd);
		// 	// diam = megaatas.ultra;
		// 	// roda1.kirimData(sockfd, std::to_string(fmax(-120,fmin(120,0))));
		// 	// roda2.kirimData(sockfd, std::to_string(fmax(-120,fmin(120,0))));
		// 	// sleep_for(microseconds(1));
		// 	// // c = c + 1;
		// 	// if (diam == 1){
		// 	// 	diam = 0;
		// 	// 	megaatas.kirimData(sockfd, std::to_string(6));
		// 	// 	sleep_for(microseconds(1));
		// 	// 	break;
		// 	// }
		// 	// else{
		// 	// 	megaatas.kirimData(sockfd, std::to_string(6));
		// 	// }
		// 	// }
		// 		break;
		// }
		// else{
		// 	megaatas.kirimData(sockfd, std::to_string(5));
		// }
		// }
		// maju(90, -90, 1);
		std::cout <<"diam "<<std::endl;
		}

		// megaatas.kirimData(sockfd, std::to_string(6));
		// sleep_for(microseconds(1));
		// for(int p = 0; p < 3; p++){
		berhenti();
		sleep_for(microseconds(1));
		std::cout <<"diam "<<std::endl;
		// }

		bola2 (realsensenode, 98);
	}
	else{

	}
	}
}

void kunci (RadiusSubscriber::SharedPtr nodesilo){
	auto node = std::make_shared<GeometryPublisher>();
while (rclcpp::ok() ) 
	{
		rclcpp::spin_some(node);
		float motor1 = (purple_linear_x_ 0.2)(3840/(3.14))/12;
		float motor2 = (-purple_linear_x_0.2)(3840/(3.14))/12;

		roda1.kirimData(sockfd, std::to_string(fmax(-100,fmin(100,motor1))));
		roda2.kirimData(sockfd, std::to_string(fmax(-100,fmin(100,motor2))));
		std::cout << "purple x "<< purple_linear_x_<<std::endl;

	if (abs(purple_linear_x_) < 2 && abs(purple_linear_x_) > 0 ){
		std::cout <<"0k "<<std::endl;
		berhenti();
		break;
	}
	}
}
void tabrak(float hancur){
	while (rclcpp::ok() ) 
	{
	std::thread th1(wheel1, 0);
	std::thread th2(wheel2, 0);
	std::thread th5(Imu);
	// std::thread th6();
	th1.join();
	th2.join();
	th5.join();
	// th6.join();
	roda1.kirimData(sockfd, std::to_string(hancur));
	roda2.kirimData(sockfd, std::to_string(hancur));
	sleep_for(microseconds(1));

	if(abs(pos1) == 1 || abs(pos2) == 2){
		berhenti();
		break;
	}
	}
}
void silo (RadiusSubscriber::SharedPtr noders, int col, int meka){

	float max;
	float kecepatan;
	auto node = std::make_shared<GeometryPublisher>();
	auto realsensenode = std::make_shared<RealSensecam>();
	std::chrono::high_resolution_clock clock;
	auto lastTime = clock.now();
	// int p = 0;
	maju(60, 60, 0.1);
	while (rclcpp::ok() ) 
	{
	angz = col;
	if (angz == 98){
		max = 300;
		kecepatan = 0.7;
	}
	else{
		max = 8000;
		kecepatan = 0.8;
	}
	sleep_for(microseconds(10));
	rclcpp::spin_some(node);
	rclcpp::spin_some(noders);
	// auto last_received_messagers = std::make_shared<RealSensecam>()->getLastReceivedMessage();
	float linxrs = std::make_shared<RealSensecam>()->getcamx();
	float linyrs = std::make_shared<RealSensecam>()->getcamy();

	float angxrs = std::make_shared<RealSensecam>()->getcamx2();
	float angyrs = std::make_shared<RealSensecam>()->getcamy2();
	std::cout << "linear.x: " << linxrs << std::endl;
	std::cout << "linear.y: " << linyrs << std::endl;
	std::cout << "angular.x: " << angxrs << std::endl;
	std::cout << "angular.y: " << angyrs << std::endl;

	std::thread th1(wheel1, 0);
	std::thread th2(wheel2, 0);
	std::thread th5(Imu);
	// std::thread th6();
	th1.join();
	th2.join();
	th5.join();
	// th6.join();

	orientasi = linxrs;
	selisih = orientasi*0.06;
	float posisi = 6*((pos1 - prepos1) + (pos2 - prepos2))/(2*3840/(3.14));
	
	float motor1 = (-kecepatan - selisih)*(3840/(3.14))/12;
	float motor2 = (-kecepatan + selisih)*(3840/(3.14))/12;

	// if (selisih == 0)
	// {
		
	// roda1.kirimData(sockfd, std::to_string(fmax(-150,fmin(150,60))));
	// roda2.kirimData(sockfd, std::to_string(fmax(-150,fmin(150,-60))));
	// }
	if(angz != 98){
			if(abs(linxrs) > 4 && abs(linxrs)  < 8 && (abs(pos1) != 1 && abs(pos2) != 1)){
			bolasilo (realsensenode, 111);
			maju(-90, -90, 0.5);
			}
			else if(abs(linxrs) > 8 && (abs(pos1) != 1 && abs(pos2) != 1)){
			putarsilo(85, 1);
			bolasilo (realsensenode, 111);
			maju(-90, -90, 0.5);
			}
			else{
			maju(-90, -90, 1);
			}
			if(abs(85 - imu) < 3 && (abs(pos1) != 1 && abs(pos2) != 1)){
			putarsilo(85, 1);
			}
			else{
			maju(-90, -90, 0.3);
			}
			if(abs(pos1) == 1 || abs(pos2) == 1){
				for(int p = 0; p < 10; p++){
					roda1.kirimData(sockfd, std::to_string(fmax(-120,fmin(120,0))));
					roda2.kirimData(sockfd, std::to_string(fmax(-120,fmin(120,0))));
					megaatas.kirimData(sockfd, std::to_string(meka));
					sleep_for(microseconds(1));
					berhenti();
		
					}	
				// int trm = 0;
				// while (rclcpp::ok() ) 
				// {
				// megaatas.kirimData(sockfd, std::to_string(2));
				// 	megaatas.terimaData(sockfd);
				// 	trm = megaatas.ultra;
				// 	std::cout <<"kirim2 " <<std::endl;
				// 	if(trm == 2){
				// 		trm = 0;
				// 		break;
				// 	}
				// }	
				break;
			}
		}
	else{
	roda1.kirimData(sockfd, std::to_string(fmax(-120,fmin(120,-motor1))));
	roda2.kirimData(sockfd, std::to_string(fmax(-120,fmin(120,-motor2))));
	}
	std::cout <<"imuambil "<< linyrs <<std::endl;
	std::cout <<"errorambil "<< selisih <<std::endl;
	std::cout <<"angzambil "<< angz <<std::endl;
	std::cout <<"sonic "<< sonic <<std::endl;
	// p += 1;
	}
}

int main(int argc, char** argv) {	

    rclcpp::init(argc, argv);

	//terimaros();
	
    // auto subscriber_node = std::make_shared<RadiusSubscriber>();
	auto realsensenode = std::make_shared<RealSensecam>();
	// auto node = std::make_shared<GeometryPublisher>();
	///////////////////initial ethernet
	struct sockaddr_in servaddr;
	sockfd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
	// Creating socket file descriptor
	if (sockfd < 0 ) {
		perror("socket creation failed");
		exit(EXIT_FAILURE);
	}
	//perror("socket creation failed");

	memset(&servaddr, 0, sizeof(servaddr));

	// Filling server information
	servaddr.sin_family = AF_INET; // IPv4
	//inet_pton(AF_INET,"192.168.0.55",&servaddr.sin_addr);
	servaddr.sin_addr.s_addr = INADDR_ANY;
	servaddr.sin_port = htons(5555);
	
	// Bind the socket with the server address
	if ( bind(sockfd, (const struct sockaddr *)&servaddr,
			sizeof(servaddr)) < 0 )
	{
		perror("bind failed");
		exit(EXIT_FAILURE);
	}

// maju(-80, -80, 2.5);
// maju(80, 80, 2);
//putar(-75);
// megaatas.kirimData(sockfd, std::to_string(1));
// sleep_for(microseconds(1));
// ulang();
// fuzzymaju(-2.3, 0);
// maju(80, 80, 1);
//////////////////// paten ////////////////////	
	fuzzymaju(9.8, 0);  //fuzzymaju(5.15, 0);
	berhenti();
	putar(85);
	// putar(90);
	berhenti();
	ulang();
	fuzzymaju(-5.6, -5);  //fuzzymaju(-3.25, -5);
	// maju(-68, 68, 0.4);
	berhenti();
	// putar(30);
	putar(0);
	berhenti();
	ulang();
	maju(125, 125, 1);
	fuzzymaju(6.7, 0); //fuzzymaju(3.4, 0);
	// maju(100, 80, 5);
	// putar(-70);
	// berhenti();
berhenti();
// for (size_t i = 0; i < 5; i++)
// {
// 	megaatas.kirimData(sockfd, std::to_string(5));
// sleep_for(microseconds(1));
// }
int trm = 0;
while (rclcpp::ok() ) 
	{
	  megaatas.kirimData(sockfd, std::to_string(5));
		megaatas.terimaData(sockfd);
		trm = megaatas.ultra;
		std::cout <<"kirim5 " <<std::endl;
		if(trm == 2){
			trm = 0;
			break;
		}
	}
//putar(-40);
putar(-80);
// berhenti();
ulang();
fuzzymaju(-5, 0); //fuzzymaju(-2.6, 0);
maju(90, 90, 1.5);
//////////////////////////////
		// maju(-80, -80, 2.5);
		// maju(80, 80, 2);
for(int k = 0; k < 5; k++){
	bola2 (realsensenode, 98);
	// maju(-60, -60, 5);
	// maju(60, 60, 2);
	ambil (realsensenode, 98, 1);
	 maju(90, 90, 0.8);
	// // ulang();
	
	// putar(-100);
	// putar(-165);
	// maju(-120, -120, 2);
	// putarsilo(0, 0);
	// putarsilo(75, 0);
	// maju(150, -150, 1);

	// tabrak(90);
	// maju(90, 90, 1);
	// maju(120, -120, 1.2);
	putar(0);
		for(int p = 0; p < 20; p++){
	// 	roda1.kirimData(sockfd, std::to_string(fmax(-120,fmin(120,0))));
	// 	roda2.kirimData(sockfd, std::to_string(fmax(-120,fmin(120,0))));
		megaatas.kirimData(sockfd, std::to_string(1));
		sleep_for(microseconds(1));
		}
	putar(85);
	// tabrak(120);
	ulang();
	maju(-125, -125, 2.5);
	fuzzymaju(-5, 0);      //fuzzymaju(-2.7, 0);
	//putar(85);
	// maju(-70, -70, 1);
	bola2 (realsensenode, 111);
	// maju(-120, -120, 2);
	silo (realsensenode, 111, 2);
	trm = 0;
	while (rclcpp::ok() ) 
	{
	  megaatas.kirimData(sockfd, std::to_string(2));
		megaatas.terimaData(sockfd);
		trm = megaatas.ultra;
		std::cout <<"kirim2 " <<std::endl;
		if(trm == 2){
			trm = 0;
			ulang();
			fuzzymaju(2, 0);
			putar(0);
			putar(-80);
			break;
		}
	}
	// for(int p = 0; p < 10; p++){
	// 	megaatas.kirimData(sockfd, std::to_string(2));
	// 	sleep_for(microseconds(1));
	// 	berhenti();
	// }	
	// maju(70, 70, 0.5);
	// for(int p = 0; p < 30; p++){
	// // 	roda1.kirimData(sockfd, std::to_string(fmax(-120,fmin(120,0))));
	// // 	roda2.kirimData(sockfd, std::to_string(fmax(-120,fmin(120,0))));
	// 	megaatas.kirimData(sockfd, std::to_string(3));
	// 	sleep_for(microseconds(1));
		// berhenti();
		int trm = 0;
	while (rclcpp::ok() ) 
	{
	  megaatas.kirimData(sockfd, std::to_string(3));
		megaatas.terimaData(sockfd);
		trm = megaatas.ultra;
		std::cout <<"kirim3 " <<std::endl;
		if(trm == 2){
			trm = 0;
			break;
		}
	}
		ulang();
		fuzzymaju(-2.8, 0);
	// }
	// putarsilo(-30);
	// putarsilo(-90);
	}


	std::cout <<"^C "<< sonic <<std::endl;
}