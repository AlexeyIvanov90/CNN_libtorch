#include "model.h"
#include  "data_set.h"
#include <chrono>


void classification(std::string path, ConvNet model)
{
	cv::Mat img = cv::imread(path);

	torch::Tensor img_tensor = img_to_tensor(img);

	torch::Tensor log_prob = model(img_tensor);
	torch::Tensor prob = torch::exp(log_prob);

	printf("Probability of being\n\
    1 category = %.2f percent\n\
    2 category  = %.2f percent\n", prob[0][0].item<float>()*100., prob[0][1].item<float>()*100.);
}


void train(std::string file_names_csv, ConvNet model, int epochs, torch::Device device)
{
	if (device == torch::kCPU)
		std::cout << "Training on CPU" << std::endl;
	else
		std::cout << "Training on GPU" << std::endl;

	model->to(device);
	
	torch::data::DataLoaderOptions OptionsData;
	OptionsData.batch_size(100).workers(4);

	auto data_set = CustomDataset(file_names_csv).map(torch::data::transforms::Stack<>());

	auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
		data_set,
		OptionsData);

	torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-3));

	int64_t log_interval = 10;
	int dataset_size = data_set.size().value();

	float best_mse = std::numeric_limits<float>::max();

	model->train();
	
	for (int epoch = 1; epoch <= epochs; epoch++) {
		auto begin = std::chrono::steady_clock::now();

		size_t batch_idx = 0;
		float mse = 0.; // mean squared error
		int count = 0;

		for (auto& batch : *data_loader) {
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

			mse += loss.template item<float>();

			batch_idx++;
			count++;
		}

		mse /= (float)count;

		auto end = std::chrono::steady_clock::now();
		auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
		std::cout << "\rTime for epoch: " << elapsed_ms.count() << " ms\n";

		std::string consol_text = "\rEpoch [" + std::to_string(epoch) + "/" +
			std::to_string(epochs) + "] Mean squared error: " + std::to_string(mse) + "\n";
		std::cout << consol_text;


		if (mse < best_mse)
		{
			model->to(torch::kCPU);
			model->eval();
			torch::save(model, "../best_model");
			best_mse = mse;
			std::cout << "model save" << std::endl;
			if (epoch != epochs) {
				model->to(device);
				model->train();
			}
		}
	}
}

