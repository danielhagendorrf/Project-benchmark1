/*
* Copyright (c) 2011. Philipp Wagner <bytefish[at]gmx[dot]de>.
* Released to public domain under terms of the BSD Simplified license.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above copyright
*     notice, this list of conditions and the following disclaimer in the
*     documentation and/or other materials provided with the distribution.
*   * Neither the name of the organization nor the names of its contributors
*     may be used to endorse or promote products derived from this software
*     without specific prior written permission.
*
*   See <http://www.opensource.org/licenses/bsd-license>
*/

#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace cv::face;
using namespace std;


int fisher(Mat img, CascadeClassifier face_cascade, vector<Mat>& images, vector<int>& labels) {
	int test = 0;
	Ptr<BasicFaceRecognizer> model = createFisherFaceRecognizer(0);
	model->train(images, labels);
	int predicted_label = -1;
	double predicted_confidence = 0.0;
	model->predict(img, predicted_label, predicted_confidence);
	string result_message;
	if (predicted_label != 0) {
		result_message = "unknown";
		cout << predicted_confidence << endl;
	}
	else {
		result_message = format("Predicted class = %d / Actual class = %d. confidence= %d", predicted_label, test, predicted_confidence);
	}
	cout << result_message << endl;
	waitKey(0);
	return 0;
	
}
