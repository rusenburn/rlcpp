#include <deeplearning/alphazero/networks/shared_res_nn.hpp>
#include <memory>
#include <torch/script.h>

namespace rl::deeplearning::alphazero
{

    SharedResImpl::SharedResImpl(std::array<int, 3> observation_shape, int n_actions, int filters, int fc_dims, int n_blocks)
        : shape_{observation_shape},
          n_actions_{n_actions},
          filters_{filters},
          fc_dims_{fc_dims},
          n_blocks_{n_blocks},
          shared_{torch::nn::Sequential{
              torch::nn::Conv2d{torch::nn::Conv2dOptions{observation_shape.at(0), filters, 3}.stride(1).padding(1)}}},
          probs_head_{torch::nn::Sequential{
              torch::nn::Conv2d{torch::nn::Conv2dOptions{filters, filters, 3}.stride(1).padding(1)},
              torch::nn::Flatten(),
              torch::nn::Linear(torch::nn::LinearOptions{observation_shape.at(1) * observation_shape.at(2) * filters, fc_dims}),
              torch::nn::ReLU(),
              torch::nn::Linear(torch::nn::LinearOptions(fc_dims, n_actions))
            //   ,torch::nn::Softmax(torch::nn::SoftmaxOptions{-1})
              }},
          wdls_head_{torch::nn::Sequential{
              torch::nn::Conv2d{torch::nn::Conv2dOptions{filters, filters, 3}.stride(1).padding(1)},
              torch::nn::Flatten(),
              torch::nn::Linear(torch::nn::LinearOptions{observation_shape.at(1) * observation_shape.at(2) * filters, fc_dims}),
              torch::nn::ReLU(),
              torch::nn::Linear(torch::nn::LinearOptions(fc_dims, 3))
              //   ,torch::nn::Softmax(torch::nn::SoftmaxOptions{-1})
          }}
    {
        for (int block{0}; block < n_blocks; block++)
        {
            shared_->push_back(ResBlockSE(filters));
        }
        register_module("shared_", shared_);
        register_module("probs_head_", probs_head_);
        register_module("wdls_head_", wdls_head_);
    }

    SharedResImpl::~SharedResImpl() = default;

    std::pair<torch::Tensor, torch::Tensor> SharedResImpl::forward(torch::Tensor state)
    {
        torch::Tensor shared = shared_->forward(state);
        torch::Tensor probs = probs_head_->forward(shared);
        probs = probs - probs.logsumexp(-1,true);
        probs = probs.softmax(-1);
        torch::Tensor wdls = wdls_head_->forward(shared);
        wdls = wdls - wdls.logsumexp(-1,true);
        wdls = wdls.softmax(-1);
        return std::make_pair(probs, wdls);
    }

    SharedResNetwork::SharedResNetwork(std::array<int, 3> observation_shape, int n_actions, int filters, int fc_dims, int n_blocks)
        : AlphazeroNetwork(SharedRes(observation_shape, n_actions, filters, fc_dims, n_blocks), torch::kCPU),
          observation_shape_{observation_shape},
          n_actions_{n_actions},
          filters_{filters},
          fc_dims_{fc_dims},
          n_blocks_{n_blocks}
    {
    }

    SharedResNetwork::~SharedResNetwork() = default;

    std::unique_ptr<IAlphazeroNetwork> SharedResNetwork::deepcopy()
    {
        auto other_network = std::make_unique<SharedResNetwork>(observation_shape_, n_actions_, filters_, fc_dims_, n_blocks_);
        deepcopyto(other_network);
        return other_network;
    }

    std::unique_ptr<IAlphazeroNetwork> SharedResNetwork::copy()
    {
        return std::make_unique<SharedResNetwork>(*this);
    }

} // namespace rl::deeplearning::alphazero