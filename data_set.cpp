#include "data_set.h"

torch::Tensor img_to_tensor(cv::Mat scr) {
	cv::cvtColor(scr, scr, cv::COLOR_BGR2RGB); // camera out - RGB, openCV - BGR

	torch::Tensor img_tensor = torch::from_blob(scr.data, { scr.rows, scr.cols, 3 }, torch::kByte).clone();

	img_tensor = img_tensor.toType(torch::kFloat);
	img_tensor = img_tensor.div(255);
	img_tensor = img_tensor.permute({ 2,0,1 });
	return img_tensor;
}


torch::Tensor img_to_tensor(std::string path) {
	cv::Mat img = cv::imread(path);
	return img_to_tensor(img);
}


std::vector<Element> read_csv(std::string& location) {
	std::fstream in(location, std::ios::in);
	std::string line;
	std::string name;
	std::string label;
	std::vector<Element> csv;

	while (getline(in, line))
	{
		std::stringstream s(line);
		getline(s, name, ',');
		getline(s, label, ',');

		csv.push_back(Element(name, stoi(label)));
	}

	return csv;
}


CustomDataset::CustomDataset(std::string& file_names_csv) {
	_csv = read_csv(file_names_csv);
}


torch::data::Example<> CustomDataset::get(size_t index) {

	std::string file_location = _csv[index].img;
	int64_t label = _csv[index].label;

	torch::Tensor img_tensor = img_to_tensor(file_location);

	torch::Tensor label_tensor = torch::full({ 1 }, label);
	label_tensor.to(torch::kInt64);

	return { img_tensor, label_tensor };
}


Element CustomDataset::get_element(size_t index) {
	return _csv[index];
}


torch::optional<size_t> CustomDataset::size() const{
	return _csv.size();
};