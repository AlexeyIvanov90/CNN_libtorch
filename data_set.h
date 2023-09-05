#pragma once

#include <vector>
#include <tuple>
#include <string>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>


struct Element
{
	Element() {};
	Element(std::string img, int label) :img{ img }, label{ label } {};

	std::string img;
	int label;
};


struct Element_data
{
	Element_data(torch::Tensor img, torch::Tensor parametr, torch::Tensor label) :img{ img }, parametr{ parametr }, label{ label } {};

	torch::Tensor img;
	torch::Tensor parametr;
	torch::Tensor label;
};


class Data_set
{
private:
	std::vector<Element> _data;
	bool data_in_ram = false;

public:
	Data_set(std::string paths_csv);
	Element_data get(size_t index);

	Element get_element(size_t index);
	size_t size();
};


torch::Tensor img_to_tensor(cv::Mat scr);
torch::Tensor img_to_tensor(std::string file_location);