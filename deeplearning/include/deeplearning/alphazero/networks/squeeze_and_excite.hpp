#ifndef RL_DEEPLEARNING_ALPHAZERO_NETWORKS_SQUEEZE_AND_EXCITE_HPP_
#define RL_DEEPLEARNING_ALPHAZERO_NETWORKS_SQUEEZE_AND_EXCITE_HPP_

#include <torch/torch.h>

namespace rl::deeplearning::alphazero
{
class SqueezeAndExciteImpl : public torch::nn::Module
{
private:
    int channels_;
    torch::nn::Sequential prepare_;
    torch::nn::Sequential fcs_;

public:
    SqueezeAndExciteImpl(int channels, int squeeze_rate);
    ~SqueezeAndExciteImpl();
    torch::Tensor forward(torch::Tensor state, torch::Tensor input_);
};
TORCH_MODULE(SqueezeAndExcite);

} // namespace rl::deeplearning::alphazero

#endif