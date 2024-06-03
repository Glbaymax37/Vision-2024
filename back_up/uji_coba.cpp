#include <opencv2/opencv.hpp>

using namespace cv;

int main() {
    // Buka kamera
    VideoCapture cap(2);
    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to open camera." << std::endl;
        return -1;
    }

    // Loop untuk membaca dari kamera
    while (true) {
        Mat frame;
        cap >> frame; // Baca frame dari kamera

        // Konversi ke grayscale
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // Deteksi tepi menggunakan Canny edge detection
        Mat thresh;
        threshold(gray, thresh, 100, 255, cv::THRESH_BINARY_INV);

        // Temukan kontur
        std::vector<std::vector<Point>> contours;
        findContours(thresh, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        // Loop melalui setiap kontur
        for (size_t i = 0; i < contours.size(); i++) {
            // Hitung panjang kontur
            double contourLength = arcLength(contours[i], true);
            if (contourLength > 30) {
                // Gambar kontur
                drawContours(frame, contours, static_cast<int>(i), Scalar(0, 255, 0), 2);

                // Jika panjang kontur lebih besar dari 30 pixel, tambahkan garis tambahan
                if (contourLength > 30) {
                    for (int j = 0; j < contourLength; j += 10) {
                        Point start, end;
                        start = contours[i][(j % static_cast<int>(contourLength))];
                        end = contours[i][((j + 10) % static_cast<int>(contourLength))];
                        line(frame, start, end, Scalar(0, 0, 255), 2);
                    }
                }
            }
        }

        // Tampilkan hasil
        imshow("Deteksi dan Pemisahan Objek", frame);

        // Keluar dari loop jika tombol 'q' ditekan
        if (waitKey(1) == 'q') {
            break;
        }
    }

    // Tutup kamera dan tutup program
    cap.release();
    destroyAllWindows();
    return 0;
}
