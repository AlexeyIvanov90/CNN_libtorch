#pragma once

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include "data_set.h"
#include "data_loader.h"


struct ConvNetImpl : public torch::nn::Module 
{
	ConvNetImpl(int64_t channels, int64_t height, int64_t width);
	torch::Tensor forward(torch::Tensor x, torch::Tensor parameter);
	int64_t GetConvOutput(int64_t channels, int64_t height, int64_t width);

	torch::nn::Conv2d conv1, conv2;
	int64_t n;
	torch::nn::Linear lin1, lin2;
};


TORCH_MODULE(ConvNet);

torch::Tensor classification(torch::Tensor img_tensor, torch::Tensor parameter, ConvNet model);
double classification_accuracy(Data_set &scr, ConvNet model, bool save_error = false);
void train(Data_loader &train_data_set, Data_set &val_data_set, ConvNet &model, int epochs, torch::Device device = torch::kCPU);
