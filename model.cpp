#include "model.h"
#include <chrono>


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


double classification_accuracy(CustomDataset &scr, ConvNet model, bool save_error)
{
	int error = 0;
	std::ofstream out;
	out.open("../error_CNN/error_CNN.csv", std::ios::out);
	for (int i = 0; i < scr.size().value(); i++) {
		auto obj = scr.get(i);

		torch::Tensor result = classification(obj.data, model);

		if (result.item<int>() != obj.target.item<int>()) {
			error++;
			if (save_error) {
				Element elem = scr.get_element(i);

				if (out.is_open())
					out << elem.img + "," +
					std::to_string(elem.label) + "," +
					std::to_string(result.item<int>()) +
					"\n";
			}
		}
	}
	out.close();


	return (double)error / scr.size().value();
}


void train(CustomDataset &train_data_set, CustomDataset &val_data_set, ConvNet &model, int epochs, torch::data::DataLoaderOptions OptionsData, torch::Device device)
{
	if (device == torch::kCPU)
		std::cout << "Training on CPU" << std::endl;
	else
		std::cout << "Training on GPU" << std::endl;

	model->to(device);

	auto train_data_set_ = train_data_set.map(torch::data::transforms::Stack<>());

	auto train_data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
		train_data_set_,
		OptionsData);


	torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-3));

	int dataset_size = train_data_set.size().value();
	float best_mse = std::numeric_limits<float>::max();

	model->train();

	for (int epoch = 1; epoch <= epochs; epoch++) {
		auto begin = std::chrono::steady_clock::now();

		size_t batch_idx = 0;
		double train_mse = 0.;

		double val_accuracy = DBL_MAX;


		for (auto& batch : *train_data_loader) {
			auto stat = "\r" + std::to_string(int((double(batch_idx * OptionsData.batch_size()) / dataset_size) * 100)) + "%";
			std::cout << stat;

			auto imgs = batch.data;
			auto labels = batch.target.squeeze();

			imgs = imgs.to(device);
			labels = labels.to(device);

			optimizer.zero_grad();
			auto output = model(imgs);

			auto loss = torch::nll_loss(output, labels);

			loss.backward();
			optimizer.step();

			train_mse += loss.template item<float>();

			batch_idx++;
		}

		train_mse /= (float)batch_idx;

		auto end = std::chrono::steady_clock::now();
		auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
		std::cout << "\rTime for epoch: " << elapsed_ms.count() << " ms\n";

		model->eval();
		model->to(torch::kCPU);
		val_accuracy = classification_accuracy(val_data_set, model);

		std::string stat = "\rEpoch [" + std::to_string(epoch) + "/" +
			std::to_string(epochs) + "] Train MSE: " + std::to_string(train_mse) +
			" Val error: " + std::to_string(val_accuracy * 100.) + " %";

		std::string model_file_name = "../models/epoch_" + std::to_string(epoch);

		if (val_accuracy < best_mse)
		{
			stat += "\nbest_model";
			model_file_name += "_best_model";
			torch::save(model, "../best_model.pt");
			best_mse = val_accuracy;
		}

		std::ofstream out;
		out.open("../models/stat.txt", std::ios::app);
		if (out.is_open())
			out << stat;
		out.close();

		std::cout << stat << std::endl;

		torch::save(model, model_file_name + ".pt");

		if (epoch != epochs) {
			model->to(device);
			model->train();
		}
	}
}