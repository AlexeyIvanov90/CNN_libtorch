#pragma once

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

struct ConvNetImpl : public torch::nn::Module 
{
    ConvNetImpl(int64_t channels, int64_t height, int64_t width) 
        : conv1(torch::nn::Conv2dOptions(3 /*input channels*/, 8 /*output channels*/, 5 /*kernel size*/).stride(2)),
          conv2(torch::nn::Conv2dOptions(8, 16, 3).stride(2)),
          
          n(GetConvOutput(channels, height, width)),
          lin1(n, 32),
          lin2(32, 2) {

        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("lin1", lin1);
        register_module("lin2", lin2);
    };

    torch::Tensor forward(torch::Tensor x) {

        x = torch::relu(torch::max_pool2d(conv1(x), 2));
        x = torch::relu(torch::max_pool2d(conv2(x), 2));

        x = x.view({-1, n});
        x = torch::relu(lin1(x));
        x = torch::log_softmax(lin2(x), 1/*dim*/);

        return x;
    };


	int64_t GetConvOutput(int64_t channels, int64_t height, int64_t width) {

        torch::Tensor x = torch::zeros({1, channels, height, width});
        x = torch::max_pool2d(conv1(x), 2);
        x = torch::max_pool2d(conv2(x), 2);
        return x.numel();
    }

    torch::nn::Conv2d conv1, conv2;
    int64_t n;
    torch::nn::Linear lin1, lin2;
};

TORCH_MODULE(ConvNet);

void classification(std::string path, ConvNet model);
void train(std::string file_names_csv, ConvNet model, int epochs, torch::Device device = torch::kCPU);