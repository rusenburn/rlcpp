#include <deeplearning/alphazero/networks/smallnn.hpp>

namespace rl::deeplearning::alphazero
{
    SmallAlphaImpl::SmallAlphaImpl(
        std::array<int, 3> &observation_shape, int n_actions, int filters, int fc_dims)
        : shared_(torch::nn::Sequential(
              torch::nn::Conv2d(torch::nn::Conv2dOptions(observation_shape[0], filters, 3).stride(1).padding(1)),
              torch::nn::ReLU(),
              torch::nn::Conv2d(torch::nn::Conv2dOptions(filters, filters, 3).stride(1).padding(1)),
              torch::nn::ReLU(),
              torch::nn::Flatten(),
              torch::nn::Linear(torch::nn::LinearOptions(filters * (observation_shape[1]) * (observation_shape[2]), fc_dims)),
              torch::nn::ReLU())),
          value_head_(torch::nn::Sequential(
              torch::nn::Linear(torch::nn::LinearOptions(fc_dims, fc_dims)),
              torch::nn::ReLU(),
              torch::nn::Linear(torch::nn::LinearOptions(fc_dims, 3)),
              torch::nn::Softmax(torch::nn::SoftmaxOptions(-1)))),
          probs_head_(torch::nn::Sequential(
              torch::nn::Linear(torch::nn::LinearOptions(fc_dims, fc_dims)),
              torch::nn::ReLU(),
              torch::nn::Linear(torch::nn::LinearOptions(fc_dims, n_actions)),
              torch::nn::Softmax(torch::nn::SoftmaxOptions(-1))))

    {
        register_module("shared_", shared_);
        register_module("value_head_", value_head_);
        register_module("probs_head_", probs_head_);
    }

    std::pair<torch::Tensor, torch::Tensor> SmallAlphaImpl::forward(torch::Tensor state)
    {
        torch::Tensor shared = shared_->forward(state);
        torch::Tensor wdl = value_head_->forward(shared);
        torch::Tensor probs = probs_head_->forward(shared);
        return std::make_pair(probs, wdl);
    }

    SmallAlphaImpl::~SmallAlphaImpl() = default;
}

namespace rl::deeplearning::alphazero
{
    SmallAlphaNetwork::SmallAlphaNetwork(std::array<int, 3> shape, int n_actions, int filters, int fc_dims)
        : AlphazeroNetwork(SmallAlpha(shape, n_actions, filters, fc_dims), torch::kCPU),
         observation_shape_(shape),
          n_actions_{n_actions},
          filters_{filters},
          fc_dims_{fc_dims}
    
    {
        mod_ = SmallAlpha(shape, n_actions, filters, fc_dims);
        dev_ = torch::kCPU;
    }
    
   
    std::unique_ptr<IAlphazeroNetwork> SmallAlphaNetwork::deepcopy()
    {
        auto other_network = std::make_unique<SmallAlphaNetwork>(observation_shape_, n_actions_, filters_, fc_dims_);
        deepcopyto(other_network);
        return other_network;
    }

    std::unique_ptr<IAlphazeroNetwork> SmallAlphaNetwork::copy()
    {
        return std::make_unique<SmallAlphaNetwork>(*this);
    }
    SmallAlphaNetwork::~SmallAlphaNetwork() = default;

}