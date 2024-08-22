#ifndef RL_DEEPLEARNING_ALPHAZERO_NETWORKS_SHARED_RES_NETWORK_HPP_
#define RL_DEEPLEARNING_ALPHAZERO_NETWORKS_SHARED_RES_NETWORK_HPP_

#include <array>
#include <utility>
#include "resblock.hpp"
#include "az.hpp"

namespace rl::deeplearning::alphazero
{
class SharedResImpl : public torch::nn::Module
{
private:
    std::array<int, 3> shape_;
    int n_actions_;
    int filters_;
    int fc_dims_;
    int n_blocks_;
    torch::nn::Sequential shared_;
    torch::nn::Sequential probs_head_;
    torch::nn::Sequential wdls_head_;

public:
    SharedResImpl(std::array<int, 3> observation_shape, int n_actions, int filters, int fc_dims, int n_blocks);
    ~SharedResImpl();
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor state);
};

TORCH_MODULE(SharedRes);

class SharedResNetwork : public AlphazeroNetwork<SharedResNetwork, SharedRes>
{
private:
    std::array<int, 3> observation_shape_;
    int n_actions_, filters_, fc_dims_, n_blocks_;

public:
    SharedResNetwork(std::array<int, 3> observation_shape, int n_actions, int filters = 128, int fc_dims = 512, int n_blocks = 5);
    ~SharedResNetwork() override;
    std::unique_ptr<IAlphazeroNetwork> deepcopy() override;
    std::unique_ptr<IAlphazeroNetwork> copy() override;
};
}

#endif