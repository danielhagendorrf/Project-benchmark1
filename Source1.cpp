#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "source1.h"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

Mat copyFace(Mat img,int leftWidth,int bottomHeight,int rightWidth,int topHeight) {
	Mat copy;
	copy.create(img.size(), img.type());
	copy.setTo(Scalar(0, 0, 0));
	for (int i = bottomHeight; i < topHeight; i++) {
		for (int j = leftWidth; j < rightWidth; j++) {
			copy.at<Vec3b>(i, j) = img.at<Vec3b>(i, j);
		}
	}
	//namedWindow("copy", WINDOW_AUTOSIZE);
	//imshow("copy", copy);
	return copy;
}