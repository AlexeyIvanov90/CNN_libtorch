#include "model.h"
#include  "data_set.h"
#include "utils.h"


int main()
{
	std::string train_file_csv = "../data_train.csv";
	std::string val_file_csv = "../data_val.csv";
	std::string test_file_csv = "../data_test.csv";

	std::string error_file_csv = "../error/error.csv";


	std::string path_NN = "../best_model.pt";

	auto epochs = 1000000;
	auto device = torch::kCPU;

	if (torch::cuda::is_available()) {
		std::cout << "CUDA is available!" << std::endl;
		device = torch::kCUDA;
	}

	device = torch::kCPU;

	CustomDataset train_data_set(train_file_csv);
	CustomDataset val_data_set(val_file_csv);
	CustomDataset test_data_set(test_file_csv);

	ConvNet model(3, 100, 200);

	torch::data::DataLoaderOptions OptionsData;
	OptionsData.batch_size(64).workers(12);

	//train(train_data_set, val_data_set, model, epochs, OptionsData);

	torch::load(model, path_NN);
	std::cout << "Model load" << std::endl;

	std::cout << "Test error: " << classification_accuracy(test_data_set, model, true) << std::endl;
	//std::cout << "Val error: " << classification_accuracy(val_data_set, model) << std::endl;
	//std::cout << "Train error: " << classification_accuracy(train_data_set, model) << std::endl;

	examination_error_img(error_file_csv);

	return 0;
}