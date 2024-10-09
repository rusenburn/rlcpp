#include <deeplearning/alphazero/networks/shared_res_nn.hpp>
#include <memory>
#include <torch/script.h>
#include <deeplearning/alphazero/networks/network_type.hpp>

namespace rl::deeplearning::alphazero
{

SharedResImpl::SharedResImpl(std::array<int, 3> observation_shape, int n_actions, int filters, int fc_dims, int n_blocks, bool normalize_outputs)
    : shape_{ observation_shape },
    n_actions_{ n_actions },
    filters_{ filters },
    fc_dims_{ fc_dims },
    n_blocks_{ n_blocks },
    normalize_outputs_{ normalize_outputs },
    shared_{ torch::nn::Sequential{
        torch::nn::Conv2d{torch::nn::Conv2dOptions{observation_shape.at(0), filters, 3}.stride(1).padding(1)}} },
        probs_head_{ torch::nn::Sequential{
            torch::nn::Conv2d{torch::nn::Conv2dOptions{filters, filters, 3}.stride(1).padding(1)},
            torch::nn::Flatten(),
            torch::nn::Linear(torch::nn::LinearOptions{observation_shape.at(1) * observation_shape.at(2) * filters, fc_dims}),
            torch::nn::ReLU(),
            torch::nn::Linear(torch::nn::LinearOptions(fc_dims, n_actions))
              } },
    wdls_head_{ torch::nn::Sequential{
        torch::nn::Conv2d{torch::nn::Conv2dOptions{filters, filters, 3}.stride(1).padding(1)},
        torch::nn::Flatten(),
        torch::nn::Linear(torch::nn::LinearOptions{observation_shape.at(1) * observation_shape.at(2) * filters, fc_dims}),
        torch::nn::ReLU(),
        torch::nn::Linear(torch::nn::LinearOptions(fc_dims, 3))
    } }
{
    for (int block{ 0 }; block < n_blocks; block++)
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
    torch::Tensor wdls = wdls_head_->forward(shared);
    if (normalize_outputs_)
    {
        probs = probs - probs.logsumexp(-1, true);
        wdls = wdls - wdls.logsumexp(-1, true);
    }
    probs = probs.softmax(-1);
    wdls = wdls.softmax(-1);
    return std::make_pair(probs, wdls);
}

SharedResNetwork::SharedResNetwork(std::array<int, 3> observation_shape, int n_actions, int filters, int fc_dims, int n_blocks, bool normalize_outputs)
    : AlphazeroNetwork(SharedRes(observation_shape, n_actions, filters, fc_dims, n_blocks, normalize_outputs), torch::kCPU, NetworkType::SharedResidualNetwork),
    observation_shape_{ observation_shape },
    n_actions_{ n_actions },
    filters_{ filters },
    fc_dims_{ fc_dims },
    n_blocks_{ n_blocks },
    normalize_outputs_{ normalize_outputs }
{
}

SharedResNetwork::~SharedResNetwork() = default;

std::unique_ptr<IAlphazeroNetwork> SharedResNetwork::deepcopy()
{
    auto other_network = std::make_unique<SharedResNetwork>(observation_shape_, n_actions_, filters_, fc_dims_, n_blocks_, normalize_outputs_);
    deepcopyto(other_network);
    return other_network;
}

std::unique_ptr<IAlphazeroNetwork> SharedResNetwork::copy()
{
    return std::make_unique<SharedResNetwork>(*this);
}

void SharedResNetwork::save_full(const std::string& file_path)
{
    torch::serialize::OutputArchive full_archive{};
    torch::serialize::OutputArchive mod_archive{};
    torch::Tensor type_tensor = torch::tensor({ static_cast<int>(NetworkType::SharedResidualNetwork) }, torch::kInt);
    full_archive.write(NETWORK_TYPE_KEY, type_tensor);
    mod_->save(mod_archive);
    full_archive.write(MODULE_KEY, mod_archive);
    torch::Tensor shape_tensor = torch::tensor({ observation_shape_[0],observation_shape_[1],observation_shape_[2] }, torch::kInt);
    full_archive.write(OBSERVATION_SHAPE_KEY, shape_tensor);
    torch::Tensor n_actions_tensor = torch::tensor({ n_actions_ }, torch::kInt);
    full_archive.write(ACTIONS_KEY, n_actions_tensor);
    torch::Tensor filters_tensor = torch::tensor({ filters_ }, torch::kInt);
    full_archive.write(FILTERS_KEY, filters_tensor);
    torch::Tensor blocks_tensor = torch::tensor({ n_blocks_ }, torch::kInt);
    full_archive.write(BLOCKS_KEY, blocks_tensor);
    torch::Tensor fc_dims_tensor = torch::tensor({ fc_dims_ }, torch::kInt);
    full_archive.write(FC_DIMS_KEY, fc_dims_tensor);
    torch::Tensor normalize_outputs_tensor = torch::tensor({ normalize_outputs_ }, torch::kInt);
    full_archive.write(NORMALIZE_OUTPUTS_KEY, normalize_outputs_tensor);
    full_archive.save_to(file_path);
}

void SharedResNetwork::load_mod(torch::serialize::InputArchive& mod_iarcive)
{
    mod_->load(mod_iarcive);
}

std::unique_ptr<SharedResNetwork> SharedResNetwork::create_full(torch::serialize::InputArchive& iarchive)
{
    torch::serialize::InputArchive mod_archive{};
    iarchive.read(MODULE_KEY, mod_archive);

    torch::Tensor n_actions_tensor{};
    iarchive.read(ACTIONS_KEY, n_actions_tensor);
    int n_actions = n_actions_tensor.accessor<int, 1>()[0];
    torch::Tensor filters_tensor{};
    iarchive.read(FILTERS_KEY, filters_tensor);
    int filters = filters_tensor.accessor<int, 1>()[0];
    torch::Tensor blocks_tensor{};
    iarchive.read(BLOCKS_KEY, blocks_tensor);
    int n_blocks = blocks_tensor.accessor<int, 1>()[0];
    torch::Tensor fc_dims_tensor{};
    iarchive.read(FC_DIMS_KEY, fc_dims_tensor);
    int fc_dims = fc_dims_tensor.accessor<int, 1>()[0];
    torch::Tensor shape_tensor{};
    iarchive.read(OBSERVATION_SHAPE_KEY, shape_tensor);
    auto shape_accessor = shape_tensor.accessor<int, 1>();
    std::array<int, 3> shape_array{ {shape_accessor[0],shape_accessor[1],shape_accessor[2] } };
    torch::Tensor normalize_outputs_tensor{};
    iarchive.read(NORMALIZE_OUTPUTS_KEY, normalize_outputs_tensor);
    bool normalize_outputs = static_cast<bool>(normalize_outputs_tensor.accessor<int, 1>()[0]);
    auto network_ptr = std::make_unique<SharedResNetwork>(shape_array, n_actions, filters, fc_dims, n_blocks, normalize_outputs);
    network_ptr->load_mod(mod_archive);
    std::cout << "Loading Shared Res network with \n";
    std::cout << "NActions " << n_actions << '\n';
    std::cout << "Filters " << filters << '\n';
    std::cout << "FcDims " << fc_dims << '\n';
    std::cout << "blocks " << n_blocks << '\n';
    std::cout << "shape " << shape_array[0] << 'x' << shape_array[1] << 'x' << shape_array[2] << '\n';
    return network_ptr;
}

const std::string SharedResNetwork::OBSERVATION_SHAPE_KEY = "ObservationShape";
const std::string SharedResNetwork::FILTERS_KEY = "Filters";
const std::string SharedResNetwork::ACTIONS_KEY = "NActions";
const std::string SharedResNetwork::FC_DIMS_KEY = "FcDims";
const  std::string SharedResNetwork::BLOCKS_KEY = "Blocks";
const  std::string SharedResNetwork::NORMALIZE_OUTPUTS_KEY = "NormalizeOutputs";

} // namespace rl::deeplearning::alphazero


