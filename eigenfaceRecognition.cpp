
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
#include "C:/Users/Daniel Hagendorf/Documents/Visual Studio 2015/Projects/Project6/Project6/eigenfaceRecognition.h"
#include "source1.h"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace cv::face;
using namespace std;


static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator , CascadeClassifier face_cascade) {
	std::ifstream file(filename.c_str(), ifstream::in);
	if (!file) {
		string error_message = "No valid input file was given, please check the given filename.";
		CV_Error(Error::StsBadArg, error_message);
	}
	string line, path, classlabel;
	while (getline(file, line)) {
		stringstream liness(line);
		getline(liness, path, separator);
		getline(liness, classlabel);
		if (!path.empty() && !classlabel.empty()) {
			std::vector<Rect> faces;
			Mat img = imread(path);
			face_cascade.detectMultiScale(img, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
			if (faces.size()>=1) {
				Mat face = copyFace(img, faces[0].x, faces[0].y, faces[0].x + faces[0].width, faces[0].y + faces[0].height);
				cvtColor(face, face, COLOR_BGR2GRAY);
				images.push_back(face);
				labels.push_back(atoi(classlabel.c_str()));
			}
			else {
				cout << "could not load image" << endl;
				cout << path << endl;
			}
		}
	}
}

int eigen(Mat img, CascadeClassifier face_cascade) {
	string csv = string("c:/csv.csv");
    vector<Mat> images;
	vector<int> labels;
	try {
		read_csv(csv, images, labels, ';', face_cascade);
	}
	catch (cv::Exception& e) {
		cerr << "Error opening file \"" << csv << "\". Reason: " << e.msg << endl;
		exit(1);
	}
	if (images.size() <= 1) {
		string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
		CV_Error(Error::StsError, error_message);
	}
	int test = 0;
	Ptr<BasicFaceRecognizer> model = createEigenFaceRecognizer(0);
	model->train(images, labels);
	int predicted_label = -1;
	double predicted_confidence = 0.0;
	model->predict(img, predicted_label, predicted_confidence);
	//int predictedLabel = model->predict(img);
	string result_message = format("Predicted class = %d / Actual class = %d. confidencd= %d", predicted_label, test,predicted_confidence);
	cout << result_message << endl;
	waitKey(0);
	return 0;
}
