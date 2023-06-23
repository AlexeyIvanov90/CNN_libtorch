#pragma once

#include <vector>
#include <tuple>
#include <string>

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

struct Element
{
	Element() {};
	Element(std::string img , int label) :img{ img }, label{ label } {};

	void print() {
		std::cout << "Element:" << std::endl;
		cv::Mat img_show = cv::imread(img);

		std::cout << img << std::endl;
		cv::imshow("Img", img_show);

		std::cout << label << std::endl;

		cv::waitKey();
	}

	std::string img;
	int label;
};

struct Element_data
{
	Element_data(torch::Tensor img, torch::Tensor label) :img{ img }, label{ label } {};

	void print() {
		std::cout << "Element:" << std::endl;
		std::cout << img.sizes() << std::endl;
		std::cout << label << std::endl;
	}

	torch::Tensor img;
	torch::Tensor label;
};

class Data_set
{
private:
	bool data_in_ram = false;
	std::vector<Element> _data;
	std::vector<Element_data> _data_mem;

public:	
	Data_set(std::string paths_csv);
	Element_data get(size_t index);

	void get_img(size_t index);
	size_t size();

	Element_data get_mem(size_t index);
	void load_to_mem();
};

torch::Tensor img_to_tensor(std::string file_location);