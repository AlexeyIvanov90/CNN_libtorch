#include "data_set.h"
#include "utils.h"


auto read_csv(const std::string& location) -> std::vector<Element> {
	std::fstream in(location, std::ios::in);
	std::string line;
	std::vector<Element> csv;
	std::string label;

	while (getline(in, line))
	{
		Element buf;
		std::stringstream s(line);
		getline(s, buf.img, ',');
		getline(s, label, ',');

		buf.label = std::stoi(label);

		csv.push_back(buf);
	}
	return csv;
}


Data_set::Data_set(std::string paths_csv)
{
	_data = read_csv(paths_csv);
}


Element Data_set::get_element(size_t index) {
	return _data.at(index);
}


Element_data Data_set::get(size_t index) {
	auto obj = _data.at(index);

	torch::Tensor img = img_to_tensor(obj.img);

	std::vector<float> buf = parameter_img(obj.img);
	
	auto opts = torch::TensorOptions().dtype(torch::kFloat);
	torch::Tensor parameter = torch::from_blob(buf.data(), { 3 }, opts).to(torch::kFloat).unsqueeze(0);
	torch::Tensor label = torch::full({ 1 }, obj.label);

	label.to(torch::kInt64);

	//std::cout << parameter << std::endl;

	return Element_data(img, parameter.clone(), label);
}


size_t Data_set::size() {
	return _data.size();
}

torch::Tensor img_to_tensor(cv::Mat scr) {
	cv::cvtColor(scr, scr, CV_BGR2RGB);
	torch::Tensor img_tensor = torch::from_blob(scr.data, { scr.rows, scr.cols, 3 }, torch::kByte).clone();
	img_tensor = img_tensor.toType(torch::kFloat);
	img_tensor = img_tensor.div(255);
	img_tensor = img_tensor.permute({ 2,0,1 });
	img_tensor = img_tensor.unsqueeze(0);
	return img_tensor;
}


torch::Tensor img_to_tensor(std::string file_location) {
	cv::Mat img = cv::imread(file_location);
	return img_to_tensor(img);
}