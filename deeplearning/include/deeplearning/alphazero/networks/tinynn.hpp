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
    bool normalize_outputs_;
public:
    TinyImpl(std::array<int, 3>& observation_shape, int n_actions, bool normalize_outputs);
    ~TinyImpl();
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor state);
};

TORCH_MODULE(Tiny);

class TinyNetwork : public AlphazeroNetwork<TinyNetwork, Tiny>
{
private:
    std::array<int, 3> observation_shape_;
    int n_actions_;
    bool normalize_outputs_;
protected:
    static const std::string OBSERVATION_SHAPE_KEY;
    static const std::string ACTIONS_KEY;
    static const std::string NORMALIZE_OUTPUTS_KEY;

public:
    TinyNetwork(std::array<int, 3> observation_shape, int n_actions, bool normalize_outputs);
    ~TinyNetwork()override;
    std::unique_ptr<IAlphazeroNetwork> deepcopy()override;
    std::unique_ptr<IAlphazeroNetwork> copy()override;
    void save_full(const std::string& file_path) override;
    void load_mod(torch::serialize::InputArchive& mod_iarcive);
    static std::unique_ptr<TinyNetwork> create_full(torch::serialize::InputArchive& iarchive);
};

} // namespace rl::deeplearning::alphazero


#endif