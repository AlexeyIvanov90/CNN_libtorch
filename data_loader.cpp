#include "data_loader.h"

Data_loader::Data_loader(Data_set &data, size_t batch_size) :data(data), batch_size(batch_size), data_size(data.size()) {
	random_index.resize(data_size);
	for (size_t index = 0; index < random_index.size(); index++)
		random_index.at(index) = index;

	random_data();
}


void Data_loader::random_data() {
	auto rng = std::default_random_engine{ rd() };
	std::shuffle(random_index.begin(), random_index.end(), rng);
}


Batch Data_loader::get_batch() {
	bool flag = false;

	for (; count_batch < data_size; ) {
		if (count_batch%batch_size == 0) {
			Element_data x = data.get(random_index.at(count_batch));

			batch_img = x.img;
			batch_parameter = x.parameter;
			batch_label = x.label;
		}
		else {
			Element_data x = data.get(random_index.at(count_batch));

			batch_img = torch::cat({ batch_img, x.img }, 0);
			batch_parameter = torch::cat({ batch_parameter, x.parameter }, 0);
			batch_label = torch::cat({ batch_label, x.label }, 0);

		}
		count_batch++;
		if (count_batch%batch_size == 0) {
			break;
		}

	//std::cout << batch_parameter << std::endl;

	}


	return Batch(batch_img, batch_parameter, batch_label);
}


bool Data_loader::epoch_end() {
	if (count_batch == data_size) {
		count_batch = 0;
		random_data();
		return true;
	}
	else {
		return false;
	}
}


size_t Data_loader::size_batch() {
	return batch_size;
}


size_t Data_loader::num_batch() {
	return count_batch / batch_size;
}


size_t Data_loader::size() {
	return data_size;
}