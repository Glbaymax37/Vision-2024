#include <opencv2/opencv.hpp>

using namespace cv;

// Variabel global untuk menyimpan nilai trackbar
int low_h = 30, low_s = 50, low_v = 50;
int high_h = 130, high_s = 255, high_v = 255;
int thres_hold = 200;


int dialedx = 6;
int dialedy = 6;

int erodex = 4;
int erodey = 4;

// Fungsi callback untuk trackbar
void onTrackbar(int, void*) {
    // Tidak melakukan apa-apa di sini karena nilai trackbar akan digunakan di dalam loop utama
}

int main() {
    // Buka video atau webcam
    VideoCapture cap(2); // 0 untuk webcam utama, bisa diganti dengan path video

    if (!cap.isOpened()) {
        std::cout << "Error: Webcam tidak bisa diakses\n";
        return -1;
    }

    // Buat jendela untuk menampung trackbar
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

    // Loop utama
    while (true) {
        Mat frame;
        cap >> frame;

        if (frame.empty()) {
            std::cout << "Error: Tidak ada frame yang didapat\n";
            break;
        }

        // Ubah warna ke HSV


        Mat gray;

       Mat thres = cv::Mat::zeros(frame.size(), frame.type());
        cvtColor(frame, gray, COLOR_BGR2GRAY); // Ubah citra ke grayscale
        threshold(gray, thres, thres_hold, 255, THRESH_BINARY_INV); // Lakukan thresholding pada citra grayscale


        // Lakukan operasi morfologi (opsional)
        // Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
        // morphologyEx(mask, mask, MORPH_OPEN, kernel);
        // morphologyEx(mask, mask, MORPH_CLOSE, kernel);

        Mat threskan;
        threshold(gray, threskan, thres_hold, 255, THRESH_BINARY);

        cv::Mat frame2 = cv::Mat::zeros(frame.size(), frame.type());
        frame.copyTo(frame2);

        cv::Mat thres2 = cv::Mat::zeros(frame.size(), frame.type());
        thres.copyTo(thres2);

        cvtColor(thres2,thres2,COLOR_GRAY2BGR);
        erode(thres2,thres2, getStructuringElement(MORPH_RECT, Size(2, 2)));
        dilate(thres2, thres2, getStructuringElement(MORPH_RECT, Size(8, 8)));


        Mat gabung;
        bitwise_and(frame2,thres2,gabung);

        Mat putih;
        threshold(gabung, putih, thres_hold, 255, THRESH_BINARY);
        cvtColor(gabung,putih,COLOR_BGR2GRAY);

        cv::Mat frame3 = cv::Mat::zeros(frame.size(), frame.type());
        gabung.copyTo(frame3);

        Mat graykhus;
        cvtColor(frame3,graykhus,COLOR_BGR2GRAY);

        
        Mat hsv;
        cvtColor(frame3, hsv, COLOR_BGR2HSV);

        // Tentukan range warna yang ingin di-segmentasi
        Scalar lower_bound(low_h, low_s, low_v);
        Scalar upper_bound(high_h, high_s, high_v);

        // Buat mask untuk segmentasi warna
        Mat mask;
        inRange(hsv, lower_bound, upper_bound, mask);

        erode(mask, mask, getStructuringElement(MORPH_RECT, Size(erodex, erodey)));
        dilate(mask, mask, getStructuringElement(MORPH_RECT, Size(dialedx, dialedy)));





        // Temukan kontur di dalam mask
        std::vector<std::vector<Point>> contours;
        std::vector<Vec4i> hierarchy;
        findContours(mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        // Gambar bounding box untuk setiap kontur
        for (size_t i = 0; i < contours.size(); ++i) {
            Rect bounding_rect = boundingRect(contours[i]);
            rectangle(frame, bounding_rect, Scalar(0, 255, 0), 2);
        }

        // Tampilkan hasil segmentasi
        imshow("Frame", frame);
        imshow("hasil", threskan);
        imshow("thres",thres);
        imshow("gabung",gabung);
        imshow("Putih",putih);
        imshow("Puh",frame3);
        imshow("KING",mask);
       
       
    
        // Tunggu tombol 'q' ditekan untuk keluar
        if (waitKey(1) == 'q') {
            break;
        }
    }

    // Tutup webcam dan destroy windows
    cap.release();
    destroyAllWindows();

    return 0;
}