#include "model.h"


ConvNetImpl::ConvNetImpl(int64_t channels, int64_t height, int64_t width)
	: conv1(torch::nn::Conv2dOptions(3, 8, 5).stride(1)),
	conv2(torch::nn::Conv2dOptions(8, 16, 3).stride(1)),

	n(GetConvOutput(channels, height, width)),
	lin1(n, 1024),
	lin2(1024, 2)
{
	register_module("conv1", conv1);
	register_module("conv2", conv2);

	register_module("lin1", lin1);
	register_module("lin2", lin2);
};

torch::Tensor ConvNetImpl::forward(torch::Tensor x)
{
	x = torch::relu(torch::max_pool2d(conv1(x), 2));
	x = torch::relu(torch::max_pool2d(conv2(x), 2));

	x = x.view({ -1, n });
	x = torch::relu(lin1(x));

	x = torch::log_softmax(lin2(x), 1/*dim*/);

	return x;
};

int64_t ConvNetImpl::GetConvOutput(int64_t channels, int64_t height, int64_t width) {

	torch::Tensor x = torch::zeros({ 1, channels, height, width });
	x = torch::max_pool2d(conv1(x), 2);
	x = torch::max_pool2d(conv2(x), 2);
	return x.numel();
}


