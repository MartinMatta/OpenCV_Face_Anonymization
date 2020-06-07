#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

#include <iostream>
#include <vector>
#include <string>

//./main ../models/opencv_face_detector_uint8.pb ../models/opencv_face_detector.pbtxt

using namespace cv::dnn;
using namespace std;
using namespace cv;

const Scalar meanVal(104.0, 177.0, 123.0);
const float confidenceThreshold = 0.7;
const double inScaleFactor = 1.0;
const size_t inHeight = 300;
const size_t inWidth = 300;

Mat anonymization(Mat src, Rect rect);
Mat forward(Net net, Mat src);
Mat pixelate(Mat src);


int main(int argc, const char* argv[]) {

    Net net = readNetFromTensorflow(argv[1], argv[2]);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    VideoCapture cap(0, CAP_V4L2);
    cap.set(CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(CAP_PROP_FRAME_HEIGHT, 720);

    if (cap.isOpened()!=true) {
        return -1;
    }

    Mat frame, face;

    while (1) {
        cap >> frame;

        if (frame.empty()) {
            break;
        }

        Mat detection = forward(net, frame);
        Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

        for(int i = 0; i < detectionMat.rows; i++) {

            float confidence = detectionMat.at<float>(i, 2);
            
            if(confidence > confidenceThreshold) {
                int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
                int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
                int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
                int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);


                Rect rect = Rect(x1, y1, x2-x1, y2-y1);
            
                if (frame.rows>=y2 && frame.cols>=x2 && y1>=0) {
                    frame = anonymization(frame, rect);
                }
                //rectangle(frame, Point(x1, y1), Point(x2, y2), Scalar(0, 255, 0),2, 4);
            }
        }
        
        imshow("image", frame);

        if (waitKey(10)==27) {
            break;
        }
    }

    return 0;
}

Mat anonymization(Mat src, Rect rect) {
    Mat face = src(rect);
    //medianBlur(face, face, 21);
    face = pixelate(face);
    src.copyTo(face);
    return src;
}

Mat forward(Net net, Mat src) {
    Mat blob = blobFromImage(src,
                             inScaleFactor,
                             Size(inWidth, inHeight),
                             meanVal, false, false);

    net.setInput(blob, "data");
    return net.forward("detection_out");
}

Mat pixelate(Mat src) {

    int size = 11;//only odd!
    int max_step = (size - 1) / 2;

    for (int i = max_step; i < src.rows- max_step; i+=size) {
        for (int j = max_step; j < src.cols- max_step; j+=size) {

            Vec3b colour = src.at<Vec3b>(Point(j, i));

            for (int k = -max_step; k <= max_step; k++) {
                for (int l = -max_step; l <= max_step; l++) {
                    src.at<Vec3b>(Point(j - k, i - l)) = colour;
                }
            }
        }
    }
    return src;
}