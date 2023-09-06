#pragma once
#include <string>
#include <opencv2/opencv.hpp>
#include <vector>

std::vector<float> parameter_img(cv::Mat img);
std::vector<float> parameter_img(std::string path);
