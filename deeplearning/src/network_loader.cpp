#include <deeplearning/network_loader.hpp>
#include <iostream>
#include <deeplearning/alphazero/networks/shared_res_nn.hpp>
#include <deeplearning/alphazero/networks/smallnn.hpp>
#include <deeplearning/alphazero/networks/tinynn.hpp>
namespace rl::deeplearning::alphazero
{
std::unique_ptr<IAlphazeroNetwork> load_network(torch::serialize::InputArchive& iarchive)
{
    torch::Tensor a{};
    bool is_able_to_read = iarchive.try_read(NETWORK_TYPE_KEY, a, true);
    auto b = a.item<int>();
    std::cout << "Network type is " << b << std::endl;
    NetworkType n{ b };
    if (n == NetworkType::SmallAlpha)
    {
        return SmallAlphaNetwork::create_full(iarchive);
    }
    else if (n == NetworkType::SharedResidualNetwork)
    {
        return SharedResNetwork::create_full(iarchive);
    }
    else if (n == NetworkType::TinyNetwork)
    {
        return TinyNetwork::create_full(iarchive);
    }
    return nullptr;
}

std::unique_ptr<IAlphazeroNetwork> load_network(const std::string& file_path)
{
    torch::serialize::InputArchive iarchive{};
    iarchive.load_from(file_path);
    return load_network(iarchive);
}
}
