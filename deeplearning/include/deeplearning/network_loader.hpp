#ifndef RL_DEEPLEARNING_NETWORK_LOADER_HPP_
#define RL_DEEPLEARNING_NETWORK_LOADER_HPP_

#include <memory>
#include <deeplearning/alphazero/networks/az.hpp>

namespace rl::deeplearning::alphazero
{
    std::unique_ptr<IAlphazeroNetwork> load_network(torch::serialize::InputArchive & iarchive);
    std::unique_ptr<IAlphazeroNetwork> load_network(const std::string& file_path);
}

#endif