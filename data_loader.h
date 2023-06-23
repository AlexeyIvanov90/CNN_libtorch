#pragma once

#include "data_set.h"
#include <random> 

struct Batch
{
	Batch(torch::Tensor img, torch::Tensor  label) :img(img), label(label) {};
	torch::Tensor img;
	torch::Tensor label;
};

class Data_loader
{
private:
	torch::Tensor batch_img;
	torch::Tensor batch_label;

	size_t data_size;
	size_t batch_size;

	size_t count_batch = 0;
	Data_set data;

	std::vector<size_t> random_index;
public:
	Data_loader(Data_set data, size_t batch_size);

	Batch get_batch();
	size_t num_batch();
	size_t size();
	size_t size_batch();
	bool epoch_end();

	void random_data();
};
