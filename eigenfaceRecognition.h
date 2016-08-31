#pragma once
#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"
#include "source1.h"

using namespace cv;
using namespace cv::face;
using namespace std;

int eigen(Mat img, CascadeClassifier face_cascade);
Mat copyFace(Mat img, int leftWidth, int bottomHeight, int rightWidth, int topHeight);