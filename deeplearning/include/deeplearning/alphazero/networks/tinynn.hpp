#ifndef RL_DEEPLEARNING_ALPHAZERO_NETWORKS_TINYNN_HPP_
#define RL_DEEPLEARNING_ALPHAZERO_NETWORKS_TINYNN_HPP_

#include <torch/torch.h>
#include <array>
#include "az.hpp"

namespace rl::deeplearning::alphazero
{

class TinyImpl : public torch::nn::Module
{
private:

    torch::nn::Sequential shared_;
    torch::nn::Sequential value_head_;
    torch::nn::Sequential probs_head_;
public:
    TinyImpl(std::array<int, 3>& observation_shape, int n_actions);
    ~TinyImpl();
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor state);
};

TORCH_MODULE(Tiny);

class TinyNetwork : public AlphazeroNetwork<TinyNetwork, Tiny>
{
private:
    std::array<int, 3> observation_shape_;
    int n_actions_;

public:
    TinyNetwork(std::array<int, 3> observation_shape, int n_actions);
    ~TinyNetwork()override;
    std::unique_ptr<IAlphazeroNetwork> deepcopy()override;
    std::unique_ptr<IAlphazeroNetwork> copy()override;
};

} // namespace rl::deeplearning::alphazero


#endif