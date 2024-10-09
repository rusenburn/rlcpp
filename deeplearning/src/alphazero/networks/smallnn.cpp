#include <deeplearning/alphazero/networks/smallnn.hpp>

namespace rl::deeplearning::alphazero
{
SmallAlphaImpl::SmallAlphaImpl(
    std::array<int, 3>& observation_shape, int n_actions, int filters, int fc_dims,bool normalize_outputs)
    : normalize_outputs_{normalize_outputs},
    shared_(torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(observation_shape[0], filters, 3).stride(1).padding(1)),
        torch::nn::ReLU(),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(filters, filters, 3).stride(1).padding(1)),
        torch::nn::ReLU(),
        torch::nn::Flatten(),
        torch::nn::Linear(torch::nn::LinearOptions(filters* (observation_shape[1])* (observation_shape[2]), fc_dims)),
        torch::nn::ReLU())),
    value_head_(torch::nn::Sequential(
        torch::nn::Linear(torch::nn::LinearOptions(fc_dims, fc_dims)),
        torch::nn::ReLU(),
        torch::nn::Linear(torch::nn::LinearOptions(fc_dims, 3)),
        torch::nn::Softmax(torch::nn::SoftmaxOptions(-1)))),
    probs_head_(torch::nn::Sequential(
        torch::nn::Linear(torch::nn::LinearOptions(fc_dims, fc_dims)),
        torch::nn::ReLU(),
        torch::nn::Linear(torch::nn::LinearOptions(fc_dims, n_actions)),
        torch::nn::Softmax(torch::nn::SoftmaxOptions(-1))))

{
    // TODO check if normalize outputs is possible
    register_module("shared_", shared_);
    register_module("value_head_", value_head_);
    register_module("probs_head_", probs_head_);
}

std::pair<torch::Tensor, torch::Tensor> SmallAlphaImpl::forward(torch::Tensor state)
{
    torch::Tensor shared = shared_->forward(state);
    torch::Tensor wdl = value_head_->forward(shared);
    torch::Tensor probs = probs_head_->forward(shared);
    return std::make_pair(probs, wdl);
}

SmallAlphaImpl::~SmallAlphaImpl() = default;
}

namespace rl::deeplearning::alphazero
{
SmallAlphaNetwork::SmallAlphaNetwork(std::array<int, 3> shape, int n_actions, int filters, int fc_dims,bool normalize_outputs)
    : AlphazeroNetwork(SmallAlpha(shape, n_actions, filters, fc_dims,normalize_outputs), torch::kCPU, NetworkType::SmallAlpha),
    observation_shape_(shape),
    n_actions_{ n_actions },
    filters_{ filters },
    fc_dims_{ fc_dims },
    normalize_outputs_{normalize_outputs}
{
}


std::unique_ptr<IAlphazeroNetwork> SmallAlphaNetwork::deepcopy()
{
    auto other_network = std::make_unique<SmallAlphaNetwork>(observation_shape_, n_actions_, filters_, fc_dims_,normalize_outputs_);
    deepcopyto(other_network);
    return other_network;
}

std::unique_ptr<IAlphazeroNetwork> SmallAlphaNetwork::copy()
{
    return std::make_unique<SmallAlphaNetwork>(*this);
}

void SmallAlphaNetwork::save_full(const std::string& file_path)
{
    
    torch::serialize::OutputArchive full_archive{};
    torch::serialize::OutputArchive mod_archive{};
    torch::Tensor network_type_tensor = torch::tensor({static_cast<int>(NetworkType::SmallAlpha)},torch::kInt); 
    full_archive.write(NETWORK_TYPE_KEY,network_type_tensor);
    mod_->save(mod_archive);
    full_archive.write(MODULE_KEY, mod_archive);
    torch::Tensor shape_tensor = torch::tensor({ observation_shape_[0],observation_shape_[1],observation_shape_[2] }, torch::kInt);
    full_archive.write(OBSERVATION_SHAPE_KEY, shape_tensor);
    torch::Tensor n_actions_tensor = torch::tensor({n_actions_},torch::kInt);
    full_archive.write(ACTIONS_KEY, n_actions_tensor);
    torch::Tensor filters_tensor = torch::tensor({filters_},torch::kInt);
    full_archive.write(FILTERS_KEY, filters_tensor);
    torch::Tensor fc_dims_tensor = torch::tensor({fc_dims_},torch::kInt);
    full_archive.write(FC_DIMS_KEY, fc_dims_tensor);
    full_archive.save_to(file_path);
}

void SmallAlphaNetwork::load_mod(torch::serialize::InputArchive& mod_iarcive)
{
    mod_->load(mod_iarcive);
}

std::unique_ptr<SmallAlphaNetwork> SmallAlphaNetwork::create_full(torch::serialize::InputArchive& iarchive)
{
    torch::serialize::InputArchive mod_archive{};
    iarchive.read(MODULE_KEY, mod_archive);

    torch::Tensor n_actions_tensor{};
    iarchive.read(ACTIONS_KEY, n_actions_tensor);
    int n_actions = n_actions_tensor.accessor<int, 1>()[0];

    torch::Tensor filters_tensor{};
    iarchive.read(FILTERS_KEY, filters_tensor);
    int filters = filters_tensor.accessor<int, 1>()[0];

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

    auto network_ptr = std::make_unique<SmallAlphaNetwork>(shape_array, n_actions, filters, fc_dims, normalize_outputs);
    network_ptr->load_mod(mod_archive);

    std::cout << "Loading Small network with \n";
    std::cout << "NActions " << n_actions << '\n';
    std::cout << "Filters " << filters << '\n';
    std::cout << "FcDims " << fc_dims << '\n';
    std::cout << "Shape " << shape_array[0] << 'x' << shape_array[1] << 'x' << shape_array[2] << '\n';
    return network_ptr;
}

SmallAlphaNetwork::~SmallAlphaNetwork() = default;
const std::string SmallAlphaNetwork::OBSERVATION_SHAPE_KEY = "ObservationShape";
const std::string SmallAlphaNetwork::ACTIONS_KEY = "NActions";
const std::string SmallAlphaNetwork::FILTERS_KEY = "Filters";
const std::string SmallAlphaNetwork::FC_DIMS_KEY = "FcDims";
const std::string SmallAlphaNetwork::NORMALIZE_OUTPUTS_KEY = "NormalizeOuputs";
}