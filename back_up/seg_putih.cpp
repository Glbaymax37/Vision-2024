#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// Fungsi callback untuk trackbar
void onTrackbarChange(int, void*) {}

int main() {
    // Membuat objek VideoCapture untuk webcam
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "Tidak dapat membuka webcam." << endl;
        return -1;
    }

    // Membuat jendela
    namedWindow("Trackbars", WINDOW_NORMAL);
    namedWindow("Mask", WINDOW_NORMAL);
    namedWindow("Original", WINDOW_NORMAL);

    // Nilai awal trackbar
    int h_min = 0, h_max = 179;
    int s_min = 0, s_max = 255;
    int v_min = 0, v_max = 255;

    // Membuat trackbar
    createTrackbar("H_min", "Trackbars", &h_min, 179, onTrackbarChange);
    createTrackbar("H_max", "Trackbars", &h_max, 179, onTrackbarChange);
    createTrackbar("S_min", "Trackbars", &s_min, 255, onTrackbarChange);
    createTrackbar("S_max", "Trackbars", &s_max, 255, onTrackbarChange);
    createTrackbar("V_min", "Trackbars", &v_min, 255, onTrackbarChange);
    createTrackbar("V_max", "Trackbars", &v_max, 255, onTrackbarChange);

    while (true) {
        Mat frame;
        // Mengambil frame dari webcam
        cap >> frame;
        
        if (frame.empty()) {
            cout << "Tidak dapat membaca frame dari webcam." << endl;
            break;
        }

        // Konversi frame ke HSV
        Mat hsv_image;
        cvtColor(frame, hsv_image, COLOR_BGR2HSV);

        // Buat batas atas dan bawah untuk rentang warna dalam HSV
        Scalar lower_bound = Scalar(h_min, s_min, v_min);
        Scalar upper_bound = Scalar(h_max, s_max, v_max);

        // Buat mask menggunakan batas warna yang ditentukan
        Mat mask;
        inRange(hsv_image, lower_bound, upper_bound, mask);

        // Menampilkan mask
        imshow("Mask", mask);

        // Menampilkan frame asli
        imshow("Original", frame);

        // Tombol keyboard 'q' untuk keluar dari loop
        if (waitKey(1) == 'q') {
            break;
        }
    }

    // Menutup semua jendela
    destroyAllWindows();

    return 0;
}
