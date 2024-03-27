#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <opencv2/opencv.hpp>  // Include OpenCV API
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <iostream>
#include <vector>
#include <time.h>

using namespace cv;
using namespace std;
using namespace rs2;

#define COLOR_ROWS 80
#define COLOR_COLS 250

cv::Rect roi; 



vector<Point> titik;
Ptr<BackgroundSubtractor> pBackSub = createBackgroundSubtractorKNN();
Mat frames,fgMask,snapshot;
Mat hsvArray1, hsvArray2;

string warna;
string hsvWindowName1 = "hsvObject1";
string hsvWindowName2 = "hsvObject2";
Scalar hsvObject1;

float hue,saturation;

double fpsLive;
int num_frames = 1;
double angle();
double slope(Point p1, Point p2);
struct ObjectData {
    int b, g, r;
};
vector<ObjectData> objectDataList;
void resetObjects() {
    objectDataList.clear();
    cout << "Objects reset." << endl;
}
void applyColormap(const Mat& depth, Mat& colorizedDepth) {
    applyColorMap(depth, colorizedDepth, COLORMAP_JET);
}
double angle() {
    Point a = titik[0];
    Point b = titik[1];
    Point c = titik[2];

    double m1 = slope(b, a);
    double m2 = slope(b, c);

    double angle = atan((m2 - m1) / (1 + m1 * m2));
    angle = round(angle * 180 / CV_PI);

    if (angle < 0) {
        angle = 180 + angle;
    }

    putText(frames, to_string(angle), Point(b.x - 40, b.y + 40), FONT_HERSHEY_DUPLEX, 2, Scalar(0, 0, 255), 2, LINE_AA);
    return angle;
}
double slope(Point p1, Point p2) {
    return static_cast<double>(p2.y - p1.y) / (p2.x - p1.x);
}


pipeline pipe;
rs2::config configs;
int keyVal;

int b, g, r;

int smin = 0, vmin = 0;
int smax = 255, vmax = 255;

int konstanta1 = 7;
int konstanta2 = 10;


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

void padi(){
 while (true) {
        // kamera RealSense
        frameset frames = pipe.wait_for_frames();
        frame color_frame = frames.get_color_frame();
        depth_frame depth_frame = frames.get_depth_frame();
        const int width = depth_frame.as<video_frame>().get_width();
        const int height = depth_frame.as<video_frame>().get_height();

        Mat depth(Size(width, height), CV_16U, (void*)depth_frame.get_data(), Mat::AUTO_STEP);
        Mat frame(Size(640, 480), CV_8UC3, (void*)color_frame.get_data(), Mat::AUTO_STEP);
        const void* depth_data = depth_frame.get_data();
        
        Mat depth_float;
        depth.convertTo(depth_float, CV_8U, 255.0 / 2000.0);

        Mat colorizedDepth;
        applyColormap(depth_float, colorizedDepth);

        Mat hsv;
        cvtColor(frame,hsv,COLOR_BGR2HSV);

        Scalar lower(hue - konstanta1, smin, vmin);
		Scalar upper(hue + konstanta2, smax, vmax);
     

        int minDistance = 500;
        // int maxDistance = 1000;
        int maxDistance = 1500;

        Mat mask;
        inRange(hsv,lower,upper,mask);

        Mat object_mask = depth > minDistance;//depth

        Mat object_mask2 = depth < maxDistance;

        Mat segmented_image = Mat::zeros(frame.size(), frame.type());
        frame.copyTo(segmented_image, object_mask);


        Mat segmented_image2 = Mat::zeros(frame.size(), frame.type());
        frame.copyTo(segmented_image2, object_mask2);

        Mat combined_mask2;
        bitwise_and(segmented_image, segmented_image2, combined_mask2);

        Mat combined_mask;
        bitwise_and(mask, object_mask2, combined_mask);


        erode(combined_mask, combined_mask, getStructuringElement(MORPH_RECT, Size(2, 2)));
        dilate(combined_mask, combined_mask, getStructuringElement(MORPH_RECT, Size(2, 2)));

        //morpologi close
        erode(combined_mask, combined_mask, getStructuringElement(MORPH_RECT, Size(2, 2)));
        dilate(combined_mask, combined_mask, getStructuringElement(MORPH_RECT, Size(2, 2)));

        vector<vector<Point>> contours;
        findContours(combined_mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        int contours_count = 0;

        for (size_t i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        double epsilon = 0.01 * arcLength(contours[i], true);
        vector<Point> approx;
        approxPolyDP(contours[i], approx, epsilon, true);

        if (200 < area && area < 10000) {
            //if (approx.size() > 5 && approx.size() < 15){
                
                // cout << "Ellipse" << endl;
            drawContours(frame, contours, (int)i, Scalar(0, 255, 0), 2);
            contours_count++;
            cout << contours_count << endl;

            Moments m = moments(contours[i]);
            Point center(m.m10 / m.m00, m.m01 / m.m00);
            circle(frame, center, 3, Scalar(255, 255, 255), -1);
            putText(frame, to_string(contours_count), center - Point(25, 25), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 2);
            string jumlah_con = "\njumlah (" + to_string(contours_count)+")";
            string messageData = "\nBarelang v" + jumlah_con;
            }
        }
        imshow("gabungan_Mask_Padi", combined_mask);
        //imshow("Segmented Image_Padi", segmented_image);
        //imshow("Segmented Image2_Padi", segmented_image2);
        imshow("Gabungan2_Padi", combined_mask2);
        imshow("frame_Padi", frame);
        imshow("mask_Padi", mask);
    
        keyVal = cv::waitKey(1) & 0xFF;

        if (keyVal == 113) { // 'Esc' u
            break;
        }
        else if (keyVal == 116) { // 't' key for snapshot
            snapshot = frame.clone();
            imshow("Snapshot_Padi", snapshot);
        }
        else if (keyVal == 117){
            destroyWindow("mask_Padi");
            destroyWindow("frame_Padi");
            destroyWindow("gabungan_Mask_Padi");
            destroyWindow("Gabungan2_Padi");
            
        }
       
    }
}

class MyKalmanFilter {
private:
    KalmanFilter kf;

public:
    MyKalmanFilter() {
        kf = KalmanFilter(4, 2);
        kf.measurementMatrix = (Mat_<float>(2, 4) << 1, 0, 0, 0, 0, 1, 0, 0);
        kf.transitionMatrix = (Mat_<float>(4, 4) << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1);
    }

    pair<int, int> predict(int coordX, int coordY) {
        Mat measured = (Mat_<float>(2, 1) << static_cast<float>(coordX), static_cast<float>(coordY));
        kf.correct(measured);
        Mat predicted = kf.predict();
        int x = static_cast<int>(predicted.at<float>(0));
        int y = static_cast<int>(predicted.at<float>(1));
        return make_pair(x, y);
    }
};

MyKalmanFilter kfWrapper;

std::clock_t startTime, endTime;

void bola(){
    while (true) {
        startTime = clock();

        rs2::frameset frames = pipe.wait_for_frames();
        rs2::frame color_frame = frames.get_color_frame();
        rs2::frame depth_frame = frames.get_depth_frame();
        Mat frame(cv::Size(640, 480), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
        //resize(frame,frame,Size(500,500));


        Mat depth_mat(Size(640, 480), CV_16UC1, (void*)depth_frame.get_data(), Mat::AUTO_STEP);
        const void* depth_data = depth_frame.get_data();

        pBackSub->apply(frame, fgMask);

        erode(fgMask, fgMask, getStructuringElement(MORPH_RECT, Size(3, 3)));
        dilate(fgMask, fgMask, getStructuringElement(MORPH_RECT, Size(3, 3)));

        // End Time
        endTime = clock();


        double seconds =  (double(endTime) - double(startTime) / double(CLOCKS_PER_SEC));
        cout << "Time taken : " << seconds << " seconds" << endl;


        fpsLive = double(num_frames) / seconds;
        cout << "Estimated frames per second : " << fpsLive << endl;

        //putText(frame, "FPS: " + to_string(int(fpsLive)), { 50, 50 }, FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 0), 2);

        keyVal = cv::waitKey(1) & 0xFF;
        if (keyVal == 113) { // 'q' key for exit
            break;
        }
        else if (keyVal == 114) { // 'r' key for reset
            resetObjects();
            destroyWindow("Color Object 1");
            destroyWindow("Color Object 2");
        }
        else if (keyVal == 116) { // 't' key for snapshot
            snapshot = frame.clone();
            imshow("Snapshot", snapshot);
        }
         
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        Mat hsvFrame2;
        cvtColor(frame, hsvFrame2, COLOR_BGR2HSV);

        erode(hsvFrame2, hsvFrame2, getStructuringElement(MORPH_RECT, Size(8, 8)));
        dilate(hsvFrame2, hsvFrame2, getStructuringElement(MORPH_RECT, Size(8, 8)));

        //morpologi close
        erode(hsvFrame2, hsvFrame2, getStructuringElement(MORPH_RECT, Size(8, 8)));
        dilate(hsvFrame2, hsvFrame2, getStructuringElement(MORPH_RECT, Size(8, 8)));


        Mat imghsv;
        cvtColor(frame, imghsv, COLOR_BGR2HSV);

        Scalar lower(hue - konstanta1, smin, vmin);
        Scalar upper(hue + konstanta2, smax, vmax);

        Mat maskobject1;
        inRange(hsvFrame2, lower, upper, maskobject1);

        vector<vector<Point>> contoursobject1;
        findContours(maskobject1, contoursobject1, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        Mat maskpembanding;
        inRange(imghsv, lower, upper, maskpembanding);

        double maxArea = 0;
        int maxAreaIdx = -1;

        for (size_t i = 0; i < contoursobject1.size(); ++i) {
            double area = contourArea(contoursobject1[i]);
            if (area > maxArea) {
                maxArea = area;
                maxAreaIdx = static_cast<int>(i);
            }
        }

        // Check if a contour with sufficient area is found
        if (maxAreaIdx != -1 && maxArea > 100) {
            // Process the largest contour only
            const auto& contour = contoursobject1[maxAreaIdx];

            // Your existing code for processing the contour goes here
            Moments oMoments = moments(contour);
            double dM01 = oMoments.m01;
            double dM10 = oMoments.m10;
            double dArea = oMoments.m00;

            //Rectangle Object 1
            Rect boundingBox = boundingRect(contour);
            rectangle(frame, boundingBox, Scalar(255, 0, 0), 2);

            // Circle Object1
            Point circleCenter(boundingBox.x + boundingBox.width / 2, boundingBox.y + boundingBox.height / 2);
            int circleRadius = (boundingBox.width + boundingBox.height) / 4;
            circle(frame, circleCenter, circleRadius, Scalar(0, 0, 255), 2);

            // Point Object1
            Point objectCenter(boundingBox.x + boundingBox.width / 2, boundingBox.y + boundingBox.height / 2);
            Point frameCenter(frame.cols / 2, frame.rows);

            //Rectangle background object 1
            Size textSize = getTextSize("Object1", FONT_HERSHEY_SIMPLEX, 0.5, 2, nullptr);
            rectangle(frame, Point(boundingBox.x, boundingBox.y - textSize.height - 10), Point(boundingBox.x + textSize.width, boundingBox.y), Scalar(255, 0, 0), FILLED);

            //garis dan keterangan
            putText(frame, "Object1", Point(boundingBox.x, boundingBox.y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 2);
            line(frame, frameCenter, objectCenter, Scalar(0, 255, 0), 2);
            circle(frame, Point(objectCenter), 5, Scalar(100, 255, 100), FILLED);

            // Koordinat Object 1
            string coordinateText1 = "(" + to_string(objectCenter.x) + ", " + to_string(objectCenter.y) + ")";
            putText(frame, coordinateText1, Point(objectCenter.x - 20, objectCenter.y - 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(250, 50, 255), 2);

            //Rectangle background Depth
            rectangle(frame, Point(500, 430), Point(640, 470), Scalar(200, 0, 0), FILLED);
            //Rectangle Keterangan
            rectangle(frame, Point(550, 410), Point(640, 430), Scalar(0, 0, 0), FILLED);

            //Depth
            uint16_t distance = depth_mat.at<uint16_t>(circleCenter);
            putText(frame, to_string(distance) + "mm", Point(500, 460), FONT_HERSHEY_PLAIN, 2, Scalar(255, 255, 255), 2);
            cout << "jarak Object 1 = " << distance << "mm" << endl;

            //Keterangan Depth
            putText(frame, "Object_1", Point(560, 425), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 2);

            // Prediksi Kalman Filter
            auto predicted = kfWrapper.predict(objectCenter.x, objectCenter.y);
            circle(frame, Point(predicted.first, predicted.second), 20, Scalar(255, 0, 0), 5);
            cout << "Object 1 - Actual: (" << objectCenter.x << ", " << objectCenter.y << ")  Predicted: (" << predicted.first << ", " << predicted.second << ")" << endl;


            //Perhitungan SUDUT
            titik.push_back(Point(0,480));
            circle(frame,Point(0,480), 5, Scalar(255, 0, 0), -1);
   
            titik.push_back(Point(320, 480));
            circle(frame, Point(320, 480), 5, Scalar(255, 0, 0), -1);
            arrowedLine(frame, titik[1], titik[0], Scalar(255, 0, 0), 3);

            titik.push_back(objectCenter);
            circle(frame, objectCenter, 5, Scalar(255, 0, 0), -1);
            arrowedLine(frame, titik[1], objectCenter, Scalar(255, 0, 0), 3);

            double degrees = angle();
            cout << "Angle: " << abs(degrees) << endl;

            Point a = titik[0];
            Point b = titik[1];
            Point c = objectCenter;

            double m1 = slope(b, a);
            double m2 = slope(b, c);

            double angle = atan((m2 - m1) / (1 + m1 * m2));
            angle = round(angle * 180 / CV_PI);

            string posisi;

            if (angle < 0) {
            angle = 180 + angle;
            }

            if(angle > 90){
                posisi = "\nPosisi Object = Kiri";
            }
            if(angle < 90){
                posisi = "\nPosisi Object = Kanan";
            }
            if(angle == 90){
                posisi = "\nPosisi Object = Tengah";
            }
            fpsLive = double(num_frames) / seconds;
            cout << "Estimated frames per second : " << fpsLive << endl;
            //putText(frame, "FPS: " + to_string(int(fpsLive)), { 50, 50 }, FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 0), 2);



            ostringstream angleText;
            angleText << fixed << setprecision(2) << angle;
            //string angleString = angleText.str();
            string angleString = "\nSudut Object (" + to_string(angle)+")";
            string fps = "\nFPS ("+ to_string(fpsLive)+")";


            // Ubah objek cv::Point objectCenter menjadi string dengan format yang sesuai
            string objectCenterString = "(" + to_string(objectCenter.x) + ", " + to_string(objectCenter.y) + ")";
            string distancel = "\nJarak Object (" + to_string(distance) + ")";

            // Gabungkan kedua string tersebut menjadi satu pesan
            string messageData = "\nBarelang V" + fps + posisi + angleString + distancel +" \nKoordinat Titik Tengah: ("+ objectCenterString + ")";

            // Masukkan pesan ke dalam message
            putText(frame, to_string(angle), Point(500,30), FONT_HERSHEY_DUPLEX, 0.8, Scalar(0, 0, 255), 2, LINE_AA);

        }
        imshow("Frame_Bola", frame);
        imshow("Mask1_Bola", maskobject1);
        imshow("pembanding_Bola", maskpembanding);


    }

}   

int main(int argc, char** argv) {
    configs.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    configs.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    pipe.start(configs);
    auto device = pipe.get_active_profile().get_device();
    auto device_product_line = device.get_info(RS2_CAMERA_INFO_PRODUCT_LINE);
    auto video_stream_profile = pipe.get_active_profile().get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
    double fps = video_stream_profile.fps();
    cout << "Frames per second camera : " << fps << endl;

    //int num_frames = 1;
   
    cout << "Capturing " << num_frames << " frames" << endl;
   
    Mat frame, colorArray,fgMask;
    snapshot = Mat(Size(640, 480), CV_8UC3, Scalar(0, 0, 0));
    resize(snapshot,snapshot,Size(500,500));
    imshow("Snapshot", snapshot);

    colorArray = Mat(COLOR_ROWS, COLOR_COLS, CV_8UC3, Scalar(0, 0, 0));
    setMouseCallback("Snapshot", on_mouse_click, &snapshot);

    hsvArray1 = Mat(COLOR_ROWS, COLOR_COLS, CV_8UC3, Scalar(0, 0, 0));
    setMouseCallback("Snapshot", on_mouse_click, &snapshot);

    namedWindow("Trackbars", (640, 200));
	    createTrackbar("Sat Min", "Trackbars", &smin, 255);
	    createTrackbar("Sat Max", "Trackbars", &smax, 255);
	    createTrackbar("Val Min", "Trackbars", &vmin, 255);
	    createTrackbar("Val Max", "Trackbars", &vmax, 255);
        createTrackbar("Konsanta 1", "Trackbars", &konstanta1, 255);
	    createTrackbar("Konstanta 2", "Trackbars", &konstanta2, 255);

  bool isPadiRunning = false;

while (true) {
    keyVal = cv::waitKey(1) & 0xFF;
    
    if (!isPadiRunning) {
        switch (keyVal) {
            case 98: // Ketika keyVal sama dengan 115
                bola();
                destroyWindow("mask_Padi");
                destroyWindow("frame_Padi");
                break; 
            case 112:
                padi();
                destroyWindow("Frame_Bola");
                destroyWindow("Mask1_Bola");
                isPadiRunning = true;
                break; 
        }
    }
    
    if (isPadiRunning && keyVal == 113) { 
        isPadiRunning = false;
        //break;
    }

    if(keyVal == 100){
        break;
    }
}

    // Tutup pipeline RealSense
    pipe.stop(); 
    destroyAllWindows();

    return 0;
}

