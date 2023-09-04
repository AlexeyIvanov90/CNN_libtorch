#include "utils.h"


std::vector<double> parameter_img(cv::Mat img) {
	std::vector<double> out;
	cv::Mat mask;

	cv::cvtColor(img, mask, cv::COLOR_BGR2GRAY);
	cv::threshold(mask, mask, 0, 255, cv::ThresholdTypes::THRESH_BINARY);

	std::vector<std::vector<cv::Point> > contours;
	cv::findContours(mask, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
	cv::Rect boundRect = cv::boundingRect(contours[0]);

	out.push_back(boundRect.height / 200.0);
	out.push_back(boundRect.width / 100.0);
	//out.push_back(cv::contourArea(contours[0]));
	out.push_back((cv::sum(mask / 255) / 20000.0)[0]);

	return out;
}


std::vector<double> parameter_img(std::string path) {
	cv::Mat img = cv::imread(path);
	return parameter_img(img);
}
