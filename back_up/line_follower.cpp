#include <opencv2/opencv.hpp>

using namespace cv;

int main() {
    // Buka kamera
    VideoCapture cap(2);
    if (!cap.isOpened()) {
        std::cerr << "Error: Gagal membuka kamera" << std::endl;
        return -1;
    }

    while (true) {
        Mat frame;
        cap >> frame; // Ambil frame dari kamera

        if (frame.empty()) {
            std::cerr << "Error: Frame kosong" << std::endl;
            break;
        }

        // Ubah ke ruang warna HSV
        Mat hsv;
        cvtColor(frame, hsv, COLOR_BGR2HSV);

        // Tentukan rentang warna putih di HSV
        Scalar lower_white = Scalar(0, 0, 200);
        Scalar upper_white = Scalar(180, 30, 255);

        // Segmentasi warna putih
        Mat mask;
        inRange(hsv, lower_white, upper_white, mask);

        // Temukan kontur dari objek yang di-segmentasi
        std::vector<std::vector<Point>> contours;
        std::vector<Vec4i> hierarchy;
        findContours(mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        // Gambar kontur pada frame asli
        Mat drawing = Mat::zeros(frame.size(), CV_8UC3);
        for (size_t i = 0; i < contours.size(); i++) {
            drawContours(frame, contours, (int)i, Scalar(0, 255, 0), 2, LINE_8, hierarchy, 0);
        }

        // Tampilkan hasil segmentasi dan kontur
        imshow("Original", frame);
        imshow("Segmentasi Warna Putih", mask);
        imshow("Kontur", drawing);

        // Tombol untuk keluar (tekan 'q')
        if (waitKey(1) == 'q') {
            break;
        }
    }

    cap.release();
    destroyAllWindows();

    return 0;
}
