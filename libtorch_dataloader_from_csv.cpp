#include "model.h"
#include "data_loader.h"


int main()
{
	std::string train_csv = "../data_train.csv"; //path csv file
	std::string val_csv = "../data_val.csv";
	std::string test_csv = "../data_test.csv";

	std::string path_NN = "../best_model.pt"; //path model NN

	auto epochs = 2;
	auto batch_size = 64;
	auto device = torch::kCPU;

	if (torch::cuda::is_available()) {
		std::cout << "CUDA is available!" << std::endl;
		device = torch::kCUDA;
	}

	Data_set Data_set_train(train_csv);
	Data_set_train.load_to_mem();

	Data_set Data_set_val(val_csv);
	Data_set Data_set_test(test_csv);

	Data_loader train_loader(Data_set_train, batch_size);

	train(train_loader, Data_set_val, path_NN, epochs);

	ConvNet model(3, 64, 64);
	torch::load(model, path_NN);
	model->eval();

	std::cout << "Test data ";
	test(Data_set_test, model);


	std::string img_1 = "../../libtorch/data/category_1/10367.png";
	classification(img_1, path_NN);

	return 0;
}

