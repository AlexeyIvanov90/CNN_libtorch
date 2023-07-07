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

	auto epochs = 100;
	auto device = torch::kCPU;

	if (torch::cuda::is_available()) {
		std::cout << "CUDA is available!" << std::endl;
		device = torch::kCUDA;
	}

	device = torch::kCPU;

	ConvNet model(3,100,200);
	
	train(train_file_csv, val_file_csv, model, epochs, device);

	torch::load(model, path_NN);
	std::cout << "Model load" << std::endl;
	
	std::cout << "Test error: " << classification_accuracy(test_file_csv, model) << std::endl;
	std::cout << "Val error: " << classification_accuracy(val_file_csv, model) << std::endl;
	std::cout << "Train error: " << classification_accuracy(train_file_csv, model) << std::endl;

	return 0;
}