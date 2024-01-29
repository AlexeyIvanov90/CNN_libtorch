#include "model.h"
#include "data_set.h"
#include "utils.h"


int main()
{
	std::string train_file_csv = "../data_set/data_train.csv";
	std::string val_file_csv = "../data_set/data_val.csv";
	std::string test_file_csv = "../data_set/data_test.csv";

	std::string path_NN = "../best_model.pt";

	auto epochs = 2;
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

	torch::load(model, path_NN);

	train(train_data_set, val_data_set, model, epochs, OptionsData);

	torch::load(model, path_NN);

	std::cout << "Test accuracy: " << classification_accuracy(test_data_set, model) << " %\n" ;
	std::cout << "Val accuracy: " << classification_accuracy(val_data_set, model) << " %\n" ;
	std::cout << "Train accuracy: " << classification_accuracy(train_data_set, model) << " %" << std::endl;

	return 0;
}