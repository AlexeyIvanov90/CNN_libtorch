#include "data_set.h"

auto ReadCsv(const std::string& location) -> std::vector<Element> {
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

torch::Tensor img_to_tensor(std::string file_location) {
	cv::Mat img = cv::imread(file_location);

	torch::Tensor img_tensor = torch::from_blob(img.data, { img.rows, img.cols, 3 }, torch::kByte).clone();
	img_tensor = img_tensor.toType(torch::kFloat);
	img_tensor = img_tensor.div(255);
	img_tensor = img_tensor.permute({ 2,0,1 });
	img_tensor = img_tensor.unsqueeze(0);
	return img_tensor;
}

Data_set::Data_set(std::string paths_csv)
{
	_data = ReadCsv(paths_csv);
}

void Data_set::get_img(size_t index) {
	Element obj = _data.at(index);
	obj.print();
}

Element_data Data_set::get(size_t index) {
	if (data_in_ram)
		return get_mem(index);

	auto obj = _data.at(index);

	torch::Tensor img = img_to_tensor(obj.img);
	torch::Tensor label = torch::full({ 1 }, obj.label);
	label.to(torch::kInt64);

	return Element_data(img, label);
}

size_t Data_set::size() {
	return _data.size();
}

Element_data Data_set::get_mem(size_t index) {
	return _data_mem.at(index);
}
void Data_set::load_to_mem() {
	for(int i = 0; i < _data.size(); i++)
		_data_mem.push_back(get(i));
	data_in_ram = true;
}

