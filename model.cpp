#include "model.h"


void train(Data_loader data_train, Data_set data_val, std::string path_save_NN, int epochs, torch::Device device)
{
	if (device == torch::kCPU)
		std::cout << "Training on CPU" << std::endl;
	else
		std::cout << "Training on GPU" << std::endl;

	ConvNet model(3, 64, 64);
	model->to(device);

	torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-3));

	int64_t log_interval = 10;
	int dataset_size = data_train.size();

	float best_mse = std::numeric_limits<float>::max();

	model->train();

	for (int epoch = 1; epoch <= epochs; epoch++) {

		size_t batch_idx = 0;
		float mse = 0.; // mean squared error
		int count = 0;

		for (; !data_train.epoch_end();) {
		
			std::cout << "\r" + std::to_string((int)(((data_train.num_batch()) / ((float)dataset_size/ data_train.size_batch()))*100)) + "%";

			Batch data = data_train.get_batch();
			
			auto img = data.img;
			auto labels = data.label.squeeze();

			img = img.to(device);
			labels = labels.to(device);

			optimizer.zero_grad();

			auto output = model->forward(img);

			auto loss = torch::nll_loss(output, labels);

			loss.backward();
			optimizer.step();

			mse += loss.template item<float>();

			batch_idx++;
			count++;
		}
		std::cout << "\r";


		mse /= (float)count;

		std::cout << "Train Epoch: " << epoch << "/" << epochs <<
		" Mean squared error: " << mse << " Validation data ";

		model->eval();
		test(data_val, model);
		model->train();

		if (mse < best_mse)
		{
			model->to(torch::kCPU);
			model->eval();
			torch::save(model, path_save_NN);
			best_mse = mse;
			std::cout << "model save" << std::endl;
			if (epoch != epochs) {
				model->to(device);
				model->train();
			}
		}
	}
}

void classification(std::string path_img, std::string path_NN) {
	auto img = img_to_tensor(path_img);

	cv::imshow("img", cv::imread(path_img));
	cv::waitKey();

	ConvNet model(3, 64, 64);
	torch::load(model, path_NN);

	model->eval();

	auto out_model = model->forward(img);
	torch::Tensor prob = torch::exp(out_model);

	std::cout << "zerno " << prob[0][0].item<float>()*100. << "%\n";
	std::cout << "ne zerno " << prob[0][1].item<float>()*100. << "%\n";
}

void test(Data_set data_test, ConvNet model){
	int error = 0;

	for (int i = 0; i < data_test.size(); i++) {
		auto data = data_test.get(i);
		auto out_model = model->forward(data.img);
		torch::Tensor prob = torch::exp(out_model);

		if ((prob[0][0].item<float>()*100. > prob[0][1].item<float>()*100. && data.label.template item<int>() != 0)||
			(prob[0][0].item<float>()*100. < prob[0][1].item<float>()*100. && data.label.template item<int>() != 1)) {
			error++;
		}
	}

	std::cout << "error: " << error << " " << (float)error/data_test.size() <<std::endl;
}



