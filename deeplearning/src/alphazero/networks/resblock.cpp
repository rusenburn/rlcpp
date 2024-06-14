#include <deeplearning/alphazero/networks/resblock.hpp>

namespace rl::deeplearning::alphazero
{
    ResBlockSEImpl::ResBlockSEImpl(int n_channels)
        : block_{torch::nn::Sequential{
              torch::nn::Conv2d{torch::nn::Conv2dOptions{n_channels, n_channels, 3}.padding(1).stride(1)},
              torch::nn::Conv2d{torch::nn::Conv2dOptions{n_channels, n_channels, 3}.padding(1).stride(1)}}},
          se_{SqueezeAndExcite(n_channels, 4)}
    {
        register_module("block_", block_);
        register_module("se_", se_);
    }
    ResBlockSEImpl::~ResBlockSEImpl()=default;

    torch::Tensor ResBlockSEImpl::forward(torch::Tensor state)
    {
        torch::Tensor output = block_->forward(state);
        output = se_->forward(output, state);
        output = output + state;
        output = output.relu();
        return output;
    }
} // namespace rl::deeplearning::alphazero
