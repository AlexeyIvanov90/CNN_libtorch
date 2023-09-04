#pragma once
#include <string>
#include <opencv2/opencv.hpp>
#include <vector>

std::vector<double> parameter_img(cv::Mat img);
std::vector<double> parameter_img(std::string path);
