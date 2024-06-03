#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    // Buka webcam (kamera default biasanya adalah 0)
    VideoCapture cap(2);
    if (!cap.isOpened()) {
        cout << "Error: Could not open camera." << endl;
        return -1;
    }

    // Parameter lebar minimum dan maksimum
    int min_width = 5;
    int max_width = 50;

    // Loop untuk menangkap frame dari webcam
    while (true) {
        Mat frame;
        cap >> frame; // Tangkap frame baru
        if (frame.empty()) {
            cout << "Error: Blank frame grabbed." << endl;
            break;
        }

        // Ubah frame menjadi grayscale
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // Terapkan thresholding
        Mat thresh;
        threshold(gray, thresh, 127, 255, THRESH_BINARY);

        //erode(thresh, thresh, getStructuringElement(MORPH_RECT, Size(8, 8)));
        dilate(thresh, thresh, getStructuringElement(MORPH_RECT, Size(2, 2)));

        // Temukan kontur
        vector<vector<Point>> contours;
        findContours(thresh, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
         

        // Iterasi melalui setiap kontur
        for (size_t i = 0; i < contours.size(); i++) {
            Rect bounding_box = boundingRect(contours[i]);
            int x = bounding_box.x;
            int y = bounding_box.y;
            int w = bounding_box.width;
            int h = bounding_box.height;

            // Cek jika objek vertikal dan memiliki lebar yang diinginkan
            if (h > w && min_width <= w && w <= max_width) {
                // Gambar bounding box di frame asli
                rectangle(frame, Point(x, y), Point(x + w, y + h), Scalar(0, 255, 0), 2);
            }
        }

        // Tampilkan frame yang telah diproses
        imshow("Result", frame);
        imshow("thres", thresh);

        // Hentikan jika tombol 'q' ditekan
        if (waitKey(30) >= 0) {
            break;
        }
    }

    return 0;
}
