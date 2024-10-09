#ifndef RL_DEEPLEARNING_ALPHAZERO_NETWORKS_SMALLNN_HPP_
#define RL_DEEPLEARNING_ALPHAZERO_NETWORKS_SMALLNN_HPP_

#include <torch/torch.h>
#include <array>
#include <string>
#include "az.hpp"
namespace rl::deeplearning::alphazero
{

class SmallAlphaImpl : public torch::nn::Module
{
    torch::nn::Sequential shared_;
    torch::nn::Sequential value_head_;
    torch::nn::Sequential probs_head_;
    bool normalize_outputs_;
public:
    SmallAlphaImpl(std::array<int, 3>& observation_shape, int n_actions, int filters, int fc_dims,bool normalize_outputs=true);
    ~SmallAlphaImpl();
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor state);
};

TORCH_MODULE(SmallAlpha);

class SmallAlphaNetwork : public AlphazeroNetwork<SmallAlphaNetwork, SmallAlpha>
{
private:
    std::array<int, 3> observation_shape_;
    int n_actions_, filters_, fc_dims_;
    bool normalize_outputs_;
protected:
    static const std::string OBSERVATION_SHAPE_KEY;
    static const std::string ACTIONS_KEY;
    static const std::string FILTERS_KEY;
    static const std::string FC_DIMS_KEY;
    static const std::string NORMALIZE_OUTPUTS_KEY;

public:
    SmallAlphaNetwork(std::array<int, 3> observation_shape, int n_actions, int filters, int fc_dims,bool normalize_outputs);
    ~SmallAlphaNetwork()override;
    std::unique_ptr<IAlphazeroNetwork> deepcopy()override;
    std::unique_ptr<IAlphazeroNetwork> copy()override;
    void save_full(const std::string& file_path) override;
    void load_mod(torch::serialize::InputArchive& mod_iarcive);
    static std::unique_ptr<SmallAlphaNetwork> create_full(torch::serialize::InputArchive& iarchive);
};
}
#endif