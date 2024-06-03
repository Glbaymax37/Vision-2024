#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#define COLOR_ROWS 80
#define COLOR_COLS 250

using namespace std;
using namespace cv;

cv::Rect roi; // Deklarasikan variabel roi di luar fungsi main

int vmin = 20;
int vmax = 255;

float hue,saturation;

int b, g, r;

void setCaptureFPS(VideoCapture& cap, double fps) {
    cap.set(CAP_PROP_FPS, fps);
}


void on_mouse_click(int event, int x, int y, int flags, void* ptr) {
    if (event == cv::EVENT_LBUTTONDOWN) {
        roi.x = x;
        roi.y = y;
    }
    else if (event == cv::EVENT_LBUTTONUP) {
        roi.width = x - roi.x;
        roi.height = y - roi.y;

        // Gambar kotak batasan di sekitar area yang dipilih
        cv::Mat* snapshot = (cv::Mat*)ptr;
        cv::Mat snapshot_with_rect = snapshot->clone();
        rectangle(snapshot_with_rect, roi, cv::Scalar(0, 255, 0), 2); // Gambar kotak batasan hijau dengan ketebalan 2 pixel
        imshow("Snapshot", snapshot_with_rect);

        // Ambil region of interest (ROI) dari gambar snapshot berdasarkan kotak batasan
        Mat region_of_interest = (*snapshot)(roi);

        // Buat salinan ROI dalam format HSV
        Mat hsv_roi;
        cvtColor(region_of_interest, hsv_roi, cv::COLOR_BGR2HSV);

        // Hitung nilai rata-rata Hue dan Saturation di dalam ROI
        Scalar mean_hsv = cv::mean(hsv_roi);
        hue = mean_hsv[0];
        saturation = mean_hsv[1];
        cout << "Rataan Nilai Hue: " << hue << endl;
        cout << "Rataan Nilai Saturation: " << saturation << endl;

        // Hitung nilai rata-rata RGB di dalam ROI
        cv::Scalar mean_rgb = mean(region_of_interest);
         r = mean_rgb[2];
         g = mean_rgb[1];
         b = mean_rgb[0];
        cout << "Rataan Nilai Rgb: [" << r << ", " << g << ", " << b << "]" << endl;
        string rgbText = "[" + to_string(r) + ", " + to_string(g)
			+ ", " + to_string(b) + "]";


        float luminance = 1 - (0.299*r + 0.587*g + 0.114*b) / 255;
		Scalar textColor;
		if (luminance < 0.5) {
			textColor = cv::Scalar(0,0,0);
		} else {
			textColor = cv::Scalar(255,255,255);
		}

		Mat colorArray;
		colorArray = cv::Mat(COLOR_ROWS, COLOR_COLS, CV_8UC3, cv::Scalar(b,g,r));
		putText(colorArray, rgbText, cv::Point2d(20, COLOR_ROWS - 20),
		FONT_HERSHEY_SIMPLEX, 0.8, textColor);
		imshow("Color", colorArray);


    }
}

int main(int argc, char** argv) {
    VideoCapture capture(2);
    double desiredFPS = 30.0;
    setCaptureFPS(capture, desiredFPS);
    b, g, r;
    if (!capture.isOpened()) {
        std::cout << "Error opening VideoCapture." << std::endl;
        return -1;
    }

    Mat frame, snapshot, colorArray;

    capture.read(frame);
    resize(frame, frame, cv::Size(500, 500));

    snapshot = Mat(frame.size(), CV_8UC3, Scalar(0, 0, 0));
    imshow("Snapshot", snapshot);

    colorArray = Mat(COLOR_ROWS, COLOR_COLS, CV_8UC3,Scalar(0, 0, 0));
    imshow("Color", colorArray);
    setMouseCallback("Snapshot", on_mouse_click, &snapshot);

    namedWindow("Trackbars", (640, 200));
	    createTrackbar("Val Min", "Trackbars", &vmin, 255);
	    createTrackbar("Val Max", "Trackbars", &vmax, 255);

    int keyVal;
    while (1) {
        if (!capture.read(frame)) {
            break;
        }
        resize(frame, frame, Size(500, 500));
        imshow("Video", frame);


        Mat hsvFrame;
        cvtColor(frame, hsvFrame, COLOR_BGR2HSV);

        
        Scalar lower(hue - 10, 50, vmin);
        Scalar upper(hue + 10, 255, vmax);

        Mat mask;
        inRange(hsvFrame, lower, upper, mask);

        Mat maskpembanding;
        inRange(hsvFrame,lower,upper,maskpembanding);
        
        erode(mask, mask, getStructuringElement(MORPH_RECT, Size(8, 8)));
        dilate(mask, mask, getStructuringElement(MORPH_RECT, Size(8, 8)));

        erode(mask, mask, getStructuringElement(MORPH_RECT, Size(8, 8)));
        dilate(mask, mask, getStructuringElement(MORPH_RECT, Size(8, 8)));

        keyVal = waitKey(1) & 0xFF;
        if (keyVal == 113) {
            break;
        }
        else if (keyVal == 116) {
            snapshot = frame.clone();
            imshow("Snapshot", snapshot);
        }

        imshow("mask",mask);
        imshow("maskpembanding",maskpembanding);


    }
    return 0;
}