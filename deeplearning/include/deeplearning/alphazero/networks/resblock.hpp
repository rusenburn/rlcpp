#ifndef RL_DEEPLEARNING_ALPHAZERO_NETWORKS_RESBLOCK_HPP_
#define RL_DEEPLEARNING_ALPHAZERO_NETWORKS_RESBLOCK_HPP_

#include "squeeze_and_excite.hpp"

namespace rl::deeplearning::alphazero
{

    class ResBlockSEImpl : public torch::nn::Module
    {
    private:
        torch::nn::Sequential block_;
        SqueezeAndExcite se_;

    public:
        ResBlockSEImpl(int n_channels);
        ~ResBlockSEImpl();
        torch::Tensor forward(torch::Tensor state);
    };
    TORCH_MODULE(ResBlockSE);

} // namespace rl::deeplearning::alphazero

#endif