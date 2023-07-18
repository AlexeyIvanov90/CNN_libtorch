#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>


struct Element
{
	Element() {};
	Element(std::string img, int label) :img{ img }, label{ label } {};

	std::string img;
	int label;
};


class CustomDataset : public torch::data::Dataset<CustomDataset>
{
    private:
		std::vector<Element> _csv;

    public:
		CustomDataset(std::string& file_names_csv);
		torch::data::Example<> get(size_t index) override;
		Element get_element(size_t index);

		torch::optional<size_t> size() const override;
};


std::vector<Element> read_csv(std::string& location);
torch::Tensor img_to_tensor(cv::Mat scr);
torch::Tensor img_to_tensor(std::string path);