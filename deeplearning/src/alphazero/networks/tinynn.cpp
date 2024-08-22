#include <deeplearning/alphazero/networks/tinynn.hpp>

namespace rl::deeplearning::alphazero
{
TinyImpl::TinyImpl(std::array<int, 3>& observation_shape, int n_actions)
    : shared_{ torch::nn::Sequential{
          torch::nn::Flatten(),
          torch::nn::Linear(torch::nn::LinearOptions(observation_shape[0] * observation_shape[1] * observation_shape[2], 1024)),
          torch::nn::ReLU()} },
          probs_head_{ torch::nn::Sequential{
              torch::nn::Linear(torch::nn::LinearOptions(512, 256)),
              torch::nn::ReLU(),
              torch::nn::Linear(torch::nn::LinearOptions(256, n_actions))} },
              value_head_{ torch::nn::Sequential{
                  torch::nn::Linear(torch::nn::LinearOptions(512, 32)),
                  torch::nn::ReLU(),
                  torch::nn::Linear(torch::nn::LinearOptions(32, 3))} }
{
    register_module("shared_", shared_);
    register_module("value_head_", value_head_);
    register_module("probs_head_", probs_head_);
}

TinyImpl::~TinyImpl() = default;

std::pair<torch::Tensor, torch::Tensor> TinyImpl::forward(torch::Tensor state)
{
    torch::Tensor shared = shared_->forward(state);
    auto split = shared.split(512, -1);
    torch::Tensor& split_0 = split[0];
    torch::Tensor& split_1 = split[1];
    torch::Tensor probs = probs_head_->forward(split_0);
    probs = probs - probs.logsumexp(-1, true);
    probs = probs.softmax(-1);
    torch::Tensor wdls = value_head_->forward(split_1);
    wdls = wdls - wdls.logsumexp(-1, true);
    wdls = wdls.softmax(-1);
    return std::make_pair(probs, wdls);
}
} // namespace rl::deeplearning::alphazero

namespace rl::deeplearning::alphazero
{
TinyNetwork::TinyNetwork(std::array<int, 3> observation_shape, int n_actions)
    : AlphazeroNetwork(Tiny(observation_shape, n_actions), torch::kCPU),
    observation_shape_(observation_shape),
    n_actions_(n_actions)
{
}
TinyNetwork::~TinyNetwork() = default;
std::unique_ptr<IAlphazeroNetwork> TinyNetwork::deepcopy()
{
    auto other_network = std::make_unique<TinyNetwork>(observation_shape_, n_actions_);
    deepcopyto(other_network);
    return other_network;
}

std::unique_ptr<IAlphazeroNetwork> TinyNetwork::copy()
{
    return std::make_unique<TinyNetwork>(*this);
}
} // namespace rl::deeplearning::alphazero
