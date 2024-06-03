#include <opencv2/opencv.hpp>
#include <iostream>
#include <librealsense2/rs.hpp>

using namespace cv;
using namespace std;
using namespace rs2;

// Variabel global untuk menyimpan nilai trackbar
int low_h = 30, low_s = 50, low_v = 50;
int high_h = 130, high_s = 255, high_v = 255;
int thres_hold = 152;


int dialedx = 1;
int dialedy = 1;

int erodex = 1;
int erodey = 1;

int min_width = 1;
int max_width = 10000;

pipeline pipe;
rs2::config configs;

// Fungsi callback untuk trackbar
void onTrackbar(int, void*) {
    // Tidak melakukan apa-apa di sini karena nilai trackbar akan digunakan di dalam loop utama
}

int main() {
    configs.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    configs.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    pipe.start(configs);
    auto device = pipe.get_active_profile().get_device();
    auto device_product_line = device.get_info(RS2_CAMERA_INFO_PRODUCT_LINE);
    auto video_stream_profile = pipe.get_active_profile().get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
    double fps = video_stream_profile.fps();
    

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
        rs2::frameset frames = pipe.wait_for_frames();
        rs2::frame color_frame = frames.get_color_frame();
        rs2::frame depth_frame = frames.get_depth_frame();
        Mat frame(cv::Size(640, 480), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
        //resize(frame,frame,Size(500,500));

        // Ubah warna ke HSV
        Mat displayFrame = Mat::zeros(480, 640, CV_8UC3);
        int x1 =  560 /2;//(640 - 320) / 2;  // Top-left corner X coordinate
        int y1 = 120;//(480 - 240) / 2;  // Top-left corner Y coordinate
        int x2 = x1 + 390;         // Bottom-right corner X coordinate
        int y2 = y1 + 0;    
        // Ubah warna ke HSV
    frame(cv::Rect(x1, y1, 70, 150)).copyTo(displayFrame(cv::Rect(x1, y1, 70, 150)));


        Mat gray;

       Mat thres = cv::Mat::zeros(displayFrame.size(), displayFrame.type());
        cvtColor(displayFrame, gray, COLOR_BGR2GRAY); // Ubah citra ke grayscale
        threshold(gray, thres, thres_hold, 255, THRESH_BINARY_INV); // Lakukan thresholding pada citra grayscale



        Mat threskan;
        threshold(gray, threskan, thres_hold, 255, THRESH_BINARY);

        cv::Mat frame2 = cv::Mat::zeros(displayFrame.size(), displayFrame.type());
        displayFrame.copyTo(frame2);

        cv::Mat thres2 = cv::Mat::zeros(displayFrame.size(), displayFrame.type());
        thres.copyTo(thres2);

        cvtColor(thres2,thres2,COLOR_GRAY2BGR);
        erode(thres2,thres2, getStructuringElement(MORPH_RECT, Size(2, 2)));
        dilate(thres2, thres2, getStructuringElement(MORPH_RECT, Size(8, 8)));


        Mat gabung;
        bitwise_and(frame2,thres2,gabung);

        Mat putih;
        threshold(gabung, putih, thres_hold, 255, THRESH_BINARY);
        cvtColor(gabung,putih,COLOR_BGR2GRAY);

        cv::Mat frame3 = cv::Mat::zeros(displayFrame.size(), displayFrame.type());
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
        Mat graylah;

        Mat last_thres;
        cvtColor(frame3,graylah, COLOR_BGR2GRAY);
        threshold(graylah,last_thres,thres_hold,255,THRESH_BINARY);

        Mat pembanding;
        cvtColor(frame3,graylah, COLOR_BGR2GRAY);
        threshold(graylah,pembanding,thres_hold,255,THRESH_BINARY);


        erode(last_thres, last_thres, getStructuringElement(MORPH_RECT, Size(erodex, erodey)));
        dilate(last_thres, last_thres, getStructuringElement(MORPH_RECT, Size(dialedx, dialedy)));





        // Temukan kontur di dalam mask
        std::vector<std::vector<Point>> contours;
        std::vector<Vec4i> hierarchy;
        findContours(last_thres, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        
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

       

        // Tampilkan hasil segmentasi
        imshow("Frame", frame);
        imshow("hasil", pembanding);
        //imshow("thres",thres);
        // imshow("gabung",gabung);
        // imshow("Putih",putih);
        imshow("Puh",frame3);
        imshow("KING",mask);
        imshow("lam",last_thres);
       
       
    
        // Tunggu tombol 'q' ditekan untuk keluar
        if (waitKey(1) == 'q') {
            break;
        }
    }

    // Tutup webcam dan destroy windows
    pipe.stop(); 
    destroyAllWindows();

    return 0;
}