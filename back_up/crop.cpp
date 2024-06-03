#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera" << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (true) {
        cap >> frame;

        if (frame.empty()) {
            std::cerr << "Error: Could not capture frame" << std::endl;
            break;
        }

        // Create a blank frame of size 640x480
        cv::Mat displayFrame = cv::Mat::zeros(480, 640, CV_8UC3);

        // Calculate the region of interest
        int x1 = (640 - 320) / 2;  // Top-left corner X coordinate
        int y1 = (480 - 240) / 2;  // Top-left corner Y coordinate
        int x2 = x1 + 320;         // Bottom-right corner X coordinate
        int y2 = y1 + 240;         // Bottom-right corner Y coordinate

        // Copy the region of interest from the captured frame to the display frame
        frame(cv::Rect(x1, y1, 320, 240)).copyTo(displayFrame(cv::Rect(x1, y1, 320, 240)));

        // Display the resulting frame
        cv::imshow("Frame", displayFrame);

        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
