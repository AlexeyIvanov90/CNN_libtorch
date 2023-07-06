#include "model.h"
#include  "data_set.h"


int main()
{
	std::string train_file_csv = "../data_train_clop.csv";
	std::string val_file_csv = "../data_val_clop.csv";
	std::string test_file_csv = "../data_test_clop.csv";

	std::string path_img_1 = "../00000.png";
	std::string path_img_2 = "../clop.png";
	std::string path_img_3 = "../al.png";

	std::string path_NN = "../best_model.pt";

	auto epochs = 10;
	auto device = torch::kCPU;

	if (torch::cuda::is_available()) {
		std::cout << "CUDA is available!" << std::endl;
		device = torch::kCUDA;
	}

	device = torch::kCPU;

	ConvNet model(3,100,200);
	torch::load(model, path_NN);

	auto train_data_set = CustomDataset(train_file_csv).map(torch::data::transforms::Stack<>());
	auto val_data_set = CustomDataset(val_file_csv).map(torch::data::transforms::Stack<>());
	auto test_data_set = CustomDataset(test_file_csv).map(torch::data::transforms::Stack<>());

	torch::data::DataLoaderOptions OptionsData;
	OptionsData.batch_size(100).workers(4);

	auto train_data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
		train_data_set, OptionsData);
	auto val_data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
		val_data_set, OptionsData);

   	 
	train(train_file_csv, model, epochs, device);

	classification(path_img_1, model);

	return 0;
}