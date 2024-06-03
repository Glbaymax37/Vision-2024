#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

Mat thresh;
Mat thresh2;
int thresh_value = 200;

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
        cv::threshold(imgray, thresh, thresh_value, 255, cv::THRESH_BINARY_INV);
        threshold(imgray, thresh2, thresh_value, 255, cv::THRESH_BINARY_INV);
        cv::findContours(thresh, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

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
    cv::VideoCapture cap(2); // Open the default camera
    if (!cap.isOpened()) {
        std::cerr << "Error: Couldn't access the webcam!" << std::endl;
        return;
    }
    namedWindow("Result", WINDOW_AUTOSIZE);
    createTrackbar("Threshold", "Result", &thresh_value, 255);

    cv::Mat frame;
    std::vector<Image> images(3); // Change 10 to the number of slices you want
    int slices = images.size();

    while (true) {
        cap >> frame;

    if (frame.empty()) {
            std::cerr << "Error: Blank frame grabbed\n";
            break;
        }
       
        SlicePart(frame, images, slices);
        cv::Mat resultImage = RepackImages(images);
        cv::imshow("Result", resultImage);
        cv::imshow("Thresh", frame);
        cv::imshow("Thresh2", thresh2);
        

        char c = cv::waitKey(30);
        if (c == 27) // ESC key to exit
            break;
    }
}

int main() {
    ProcessImages();
    return 0;
}