#ifndef RL_DEEPLEARNING_ALPHAZERO_NETWORKS_SMALLNN_HPP_
#define RL_DEEPLEARNING_ALPHAZERO_NETWORKS_SMALLNN_HPP_

#include <torch/torch.h>
#include <array>
#include "az.hpp"
namespace rl::deeplearning::alphazero
{

    class SmallAlphaImpl : public torch::nn::Module
    {
        torch::nn::Sequential shared_;
        torch::nn::Sequential value_head_;
        torch::nn::Sequential probs_head_;
    public:
        SmallAlphaImpl(std::array<int,3> &observation_shape, int n_actions, int filters, int fc_dims);
        ~SmallAlphaImpl();
        std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor state);
    };

    TORCH_MODULE(SmallAlpha);

    class SmallAlphaNetwork : public AlphazeroNetwork<SmallAlphaNetwork,SmallAlpha>
    {
    private:
        std::array<int,3> observation_shape_;
        int n_actions_, filters_, fc_dims_;

    public:
        SmallAlphaNetwork(std::array<int,3> observation_shape, int n_actions, int filters, int fc_dims);
        ~SmallAlphaNetwork()override;
        std::unique_ptr<IAlphazeroNetwork> deepcopy()override;
        std::unique_ptr<IAlphazeroNetwork> copy()override;
    };
}
#endif