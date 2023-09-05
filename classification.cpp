#include "model.h"

torch::Tensor classification(torch::Tensor img_tensor, ConvNet model)
{
	model->eval();
	model->to(torch::kCPU);
	img_tensor.to(torch::kCPU);
	img_tensor = img_tensor.unsqueeze(0);

	torch::Tensor log_prob = model(img_tensor);
	torch::Tensor prob = torch::exp(log_prob);

	return torch::argmax(prob);
}


double classification_accuracy(Data_set &scr, ConvNet model, bool save_error)
{
	int error = 0;
	std::ofstream out;
	out.open("../error_CNN/error_CNN.csv", std::ios::out);
	for (int i = 0; i < scr.size(); i++) {
		auto obj = scr.get(i);

		torch::Tensor result = classification(obj.img, model);

		if (result.item<int>() != obj.label.item<int>()) {
			error++;
			if (save_error) {
				Element elem = scr.get_element(i);
				cv::Mat img = cv::imread(elem.img);
				std::string path_img = "../error_CNN/" + elem.img.substr(elem.img.rfind("/") + 1);
				cv::imwrite(path_img, img);

				if (out.is_open())
					out << elem.img + "," +
					std::to_string(elem.label) + "," +
					std::to_string(result.item<int>()) +
					"\n";
			}
		}
		else {
			Element elem = scr.get_element(i);
			cv::Mat img = cv::imread(elem.img);
			std::string path_img = "../new_data/" + elem.img.substr(elem.img.rfind("/") + 1);
			cv::imwrite(path_img, img);
		}
	}
	out.close();

	return (double)error / scr.size();
}