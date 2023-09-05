#include "model.h"

#include <chrono>
#include <filesystem>

void train(Data_loader &train_data_loader, Data_set &val_data_set, ConvNet &model, int epochs, torch::Device device)
{
	if (device == torch::kCPU)
		std::cout << "Training on CPU" << std::endl;
	else
		std::cout << "Training on GPU" << std::endl;

	model->to(device);

	torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-3));

	//int dataset_size = train_data_set.size().value();
	float best_mse = std::numeric_limits<float>::max();

	model->train();

	for (int epoch = 1; epoch <= epochs; epoch++) {
		auto begin = std::chrono::steady_clock::now();

		size_t batch_idx = 0;
		double train_mse = 0.;

		double val_accuracy = DBL_MAX;


		for (; !train_data_loader.epoch_end();) {
			//auto stat = "\r" + std::to_string(int((double(batch_idx * OptionsData.batch_size()) / dataset_size) * 100)) + "%";
			std::string consol_text = "\r" + std::to_string((int)(((train_data_loader.num_batch()) / ((float)train_data_loader.size() / train_data_loader.size_batch())) * 100)) + "%";

			std::cout << stat;

			Batch data = train_data_loader.get_batch();

			auto img = data.img;
			auto parameter = data.parameter;
			auto labels = data.label.squeeze();


			img = img.to(device);
			parameter = parameter.to(device);
			labels = labels.to(device);

			optimizer.zero_grad();
			auto output = model(img);

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