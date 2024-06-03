#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <librealsense2/rs.hpp>

using namespace cv;
using namespace std;
using namespace rs2;

Mat thresh;
Mat thresh2;
Mat White_mask;
int thresh_value = 200;

pipeline pipekan;
rs2::config configs;

int hmin = 0;
int smin = 0;
int vmin = 0;

int hmax = 180;
int smax = 255;
int vmax = 255;

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

        Scalar lowwer_white(hmin,smin,vmin);
        Scalar upper_white(hmax,vmax,smax);
        inRange(imgray,lowwer_white,upper_white,White_mask);
        cv::threshold(imgray, thresh, thresh_value, 255, cv::THRESH_BINARY);
        threshold(imgray, thresh2, thresh_value, 255, cv::THRESH_BINARY);
        cv::findContours(White_mask, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

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

cv::Mat RemoveBackground(cv::Mat image, bool b) {
    int up = 100;
    cv::Mat mask;
    cv::Scalar lower(0, 0, 0);
    cv::Scalar upper(up, up, up);
    if (b) {
        cv::inRange(image, lower, upper, mask);
        cv::bitwise_and(image, image, image, mask = mask);
        cv::bitwise_not(image, image, mask = mask);
        image = (255 - image);
    }
    return image;
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

cv::Mat RepackImages(const std::vector<Image>& images) {
    cv::Mat img = images[0].getImage().clone();
    for (size_t i = 1; i < images.size(); ++i) {
        if (i == 0)
            cv::vconcat(img, images[i].getImage(), img);
        else
            cv::vconcat(img, images[i].getImage(), img);
    }
    return img;
}

void ProcessImages() {
    namedWindow("Result", WINDOW_AUTOSIZE);
    createTrackbar("Threshold", "Result", &thresh_value, 255);
    createTrackbar("Hmin", "Result", &hmin, 255);
    createTrackbar("Smin", "Result", &smin, 255);
    createTrackbar("Vmin", "Result", &vmin, 255);
    createTrackbar("Hmax", "Result", &hmax, 255);
    createTrackbar("Smax", "Result", &smax, 255);
    createTrackbar("Vmax", "Result", &vmax, 255);

    std::vector<Image> images(3); // Change 10 to the number of slices you want
    int slices = images.size();

    while (true) {
        frameset frames = pipekan.wait_for_frames();
        frame color_frame = frames.get_color_frame();
        depth_frame depth_frame = frames.get_depth_frame();
        const int width = depth_frame.as<video_frame>().get_width();
        const int height = depth_frame.as<video_frame>().get_height();

        // Convert RealSense color frame to OpenCV Mat
        Mat frame(Size(width, height), CV_8UC3, (void*)color_frame.get_data(), Mat::AUTO_STEP);
       
        SlicePart(frame, images, slices);
        cv::Mat resultImage = RepackImages(images);
        cv::imshow("Result", resultImage);
        cv::imshow("Mask", White_mask);
        cv::imshow("Thresh", thresh);
        cv::imshow("Thresh2", thresh2);
        
        char c = cv::waitKey(30);
        if (c == 27) // ESC key to exit
            break;
    }
}

int main() {
     configs.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    configs.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    pipekan.start(configs);
    auto device = pipekan.get_active_profile().get_device();
    auto device_product_line = device.get_info(RS2_CAMERA_INFO_PRODUCT_LINE);
    auto video_stream_profile = pipekan.get_active_profile().get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
    ProcessImages();
    return 0;
}