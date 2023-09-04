#pragma once

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include "data_set.h"


struct ConvNetImpl : public torch::nn::Module 
{
	ConvNetImpl(int64_t channels, int64_t height, int64_t width);
	torch::Tensor forward(torch::Tensor x);
	int64_t GetConvOutput(int64_t channels, int64_t height, int64_t width);

	torch::nn::Conv2d conv1, conv2;
	int64_t n;
	torch::nn::Linear lin1, lin2;
};


TORCH_MODULE(ConvNet);

torch::Tensor classification(torch::Tensor img_tensor, ConvNet model);
void classification_data(CustomDataset &scr, ConvNet model);
double classification_accuracy(CustomDataset &scr, ConvNet model, bool save_error = false);
void train(CustomDataset &train_data_set, CustomDataset &val_data_set, ConvNet &model, int epochs, torch::data::DataLoaderOptions OptionsData, torch::Device device = torch::kCPU);
