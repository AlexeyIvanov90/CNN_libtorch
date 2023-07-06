#include "model.h"
#include  "data_set.h"
#include <chrono>


torch::Tensor classification(torch::Tensor img_tensor, ConvNet model)
{
	torch::Tensor log_prob = model(img_tensor);
	torch::Tensor prob = torch::exp(log_prob);

	return torch::argmax(prob);
}


double classification_accuracy(std::string file_csv, ConvNet model)
{
	int error = 0;

	auto data_set = CustomDataset(file_csv);

	for (int i = 0; i < data_set.size().value(); i++) {
		auto obj = data_set.get(i);
		torch::Tensor result = classification(obj.data, model);

		if (result.item<int>() != obj.target.item<int>())
			error++;
	}

	return (double)error / data_set.size().value();
}


void train(std::string train_file_csv, std::string val_file_csv, ConvNet model, int epochs, torch::Device device)
{
	if (device == torch::kCPU)
		std::cout << "Training on CPU" << std::endl;
	else
		std::cout << "Training on GPU" << std::endl;

	model->to(device);
	
	torch::data::DataLoaderOptions OptionsData;
	OptionsData.batch_size(100).workers(4);

	auto train_data_set = CustomDataset(train_file_csv).map(torch::data::transforms::Stack<>());
	auto val_data_set = CustomDataset(val_file_csv).map(torch::data::transforms::Stack<>());


	auto train_data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
		train_data_set,
		OptionsData);

	auto val_data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
		val_data_set,
		OptionsData);

	torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-3));

	int64_t log_interval = 10;
	int dataset_size = train_data_set.size().value();

	float best_mse = std::numeric_limits<float>::max();

	model->train();
	
	for (int epoch = 1; epoch <= epochs; epoch++) {
		auto begin = std::chrono::steady_clock::now();

		size_t batch_idx = 0;
		float train_mse = 0.;
		float val_mse = 0.;
		int train_count = 0;
		int val_count = 0;


		for (auto& batch : *train_data_loader) {
			std::string consol_text = "\r[" + std::to_string(batch_idx * batch.data.size(0)) + "/" +
			std::to_string(dataset_size) + "]";
			std::cout << consol_text;

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
			train_count++;
		}

		train_mse /= (float)train_count;
		
		model->eval();
		for (auto& batch : *val_data_loader) {
			auto imgs = batch.data;
			auto labels = batch.target.squeeze();

			imgs = imgs.to(device);
			labels = labels.to(device);

			auto output = model(imgs);
			auto loss = torch::nll_loss(output, labels);

			val_mse += loss.template item<float>();
			val_count++;
		}

		val_mse /= (float)val_count;
		model->train();


		auto end = std::chrono::steady_clock::now();
		auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
		std::cout << "\rTime for epoch: " << elapsed_ms.count() << " ms\n";

		std::string consol_text = "\rEpoch [" + std::to_string(epoch) + "/" +
			std::to_string(epochs) + "] Train error: " + std::to_string(train_mse) + 
			" Val error: " + std::to_string(val_mse) + "\n";
		std::cout << consol_text;


		if (val_mse < best_mse)
		{
			model->to(torch::kCPU);
			model->eval();
			torch::save(model, "../best_model.pt");
			best_mse = val_mse;
			std::cout << "model save" << std::endl;
			if (epoch != epochs) {
				model->to(device);
				model->train();
			}
		}
	}
}