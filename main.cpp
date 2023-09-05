#include "model.h"
#include "data_set.h"
#include "utils.h"


int main()
{
	//parameter_img("../test.png");

	std::string file_csv = "../data_set/data.csv";

	std::string train_file_csv = "../data_set/data_train.csv";
	std::string val_file_csv = "../data_set/data_val.csv";
	std::string test_file_csv = "../data_set/data_test.csv";
	std::string error_file_csv = "../error_CNN/error_CNN.csv";
	std::string single_file_csv = "../single_data/single_data.csv";

	std::string path_NN = "../best_model.pt";

	auto epochs = 1000000;
	auto device = torch::kCPU;

	if (torch::cuda::is_available()) {
		std::cout << "CUDA is available!" << std::endl;
		device = torch::kCUDA;
	}

	device = torch::kCPU;

	Data_set train_data_set(train_file_csv);
	Data_set val_data_set(val_file_csv);
	Data_set test_data_set(test_file_csv);

	ConvNet model(3, 100, 200);

	Data_loader train_data_loader(train_data_set, 64);

	train(train_data_loader, val_data_set, model, epochs);

	torch::load(model, path_NN);
	std::cout << "Model load" << std::endl;

	std::cout << "Test error: " << classification_accuracy(test_data_set, model) << std::endl;
	std::cout << "Val error: " << classification_accuracy(val_data_set, model) << std::endl;
	std::cout << "Train error: " << classification_accuracy(train_data_set, model) << std::endl;

	return 0;
}