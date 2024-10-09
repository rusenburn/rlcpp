#ifndef RL_DEEPLEARNING_ALPHAZERO_NETWORKS_SHARED_RES_NETWORK_HPP_
#define RL_DEEPLEARNING_ALPHAZERO_NETWORKS_SHARED_RES_NETWORK_HPP_

#include <array>
#include <utility>
#include <string>
#include <string_view>
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
    bool normalize_outputs_;
    torch::nn::Sequential shared_;
    torch::nn::Sequential probs_head_;
    torch::nn::Sequential wdls_head_;

public:
    SharedResImpl(std::array<int, 3> observation_shape, int n_actions, int filters, int fc_dims, int n_blocks,bool normalize_outputs);
    ~SharedResImpl();
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor state);
};

TORCH_MODULE(SharedRes);

class SharedResNetwork : public AlphazeroNetwork<SharedResNetwork, SharedRes>
{
    
private:
    std::array<int, 3> observation_shape_;
    int n_actions_, filters_, fc_dims_, n_blocks_;
    bool normalize_outputs_;
protected:
    static const std::string OBSERVATION_SHAPE_KEY;
    static const std::string FILTERS_KEY;
    static const std::string ACTIONS_KEY;
    static const std::string FC_DIMS_KEY;
    static const std::string BLOCKS_KEY;
    static const std::string NORMALIZE_OUTPUTS_KEY;

public:
    SharedResNetwork(std::array<int, 3> observation_shape, int n_actions, int filters = 128, int fc_dims = 512, int n_blocks = 5,bool normalize_outputs=true);
    ~SharedResNetwork() override;
    std::unique_ptr<IAlphazeroNetwork> deepcopy() override;
    std::unique_ptr<IAlphazeroNetwork> copy() override;
    void save_full(const std::string& file_path) override;
    void load_mod(torch::serialize::InputArchive& mod_iarcive);
    static std::unique_ptr<SharedResNetwork> create_full(torch::serialize::InputArchive& iarchive);
};
}

#endif