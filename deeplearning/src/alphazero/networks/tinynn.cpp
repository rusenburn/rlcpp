#include <deeplearning/alphazero/networks/tinynn.hpp>
#include <deeplearning/alphazero/networks/network_type.hpp>

namespace rl::deeplearning::alphazero
{
TinyImpl::TinyImpl(std::array<int, 3>& observation_shape, int n_actions, bool normalize_outputs)
    : normalize_outputs_{ normalize_outputs },
    shared_{ torch::nn::Sequential{
      torch::nn::Flatten(),
      torch::nn::Linear(torch::nn::LinearOptions(observation_shape[0] * observation_shape[1] * observation_shape[2], 1024)),
      torch::nn::ReLU()} },
      probs_head_{ torch::nn::Sequential{
                torch::nn::Linear(torch::nn::LinearOptions(512, 256)),
                torch::nn::ReLU(),
                torch::nn::Linear(torch::nn::LinearOptions(256, n_actions))} },
                value_head_{ torch::nn::Sequential{
                              torch::nn::Linear(torch::nn::LinearOptions(512, 32)),
                              torch::nn::ReLU(),
                              torch::nn::Linear(torch::nn::LinearOptions(32, 3))} }
{
    register_module("shared_", shared_);
    register_module("value_head_", value_head_);
    register_module("probs_head_", probs_head_);
}

TinyImpl::~TinyImpl() = default;

std::pair<torch::Tensor, torch::Tensor> TinyImpl::forward(torch::Tensor state)
{
    torch::Tensor shared = shared_->forward(state);
    auto split = shared.split(512, -1);
    torch::Tensor& split_0 = split[0];
    torch::Tensor& split_1 = split[1];
    torch::Tensor probs = probs_head_->forward(split_0);
    torch::Tensor wdls = value_head_->forward(split_1);

    if (normalize_outputs_)
    {
        probs = probs - probs.logsumexp(-1, true);
        wdls = wdls - wdls.logsumexp(-1, true);
    }


    probs = probs.softmax(-1);
    wdls = wdls.softmax(-1);
    return std::make_pair(probs, wdls);
}
} // namespace rl::deeplearning::alphazero

namespace rl::deeplearning::alphazero
{
TinyNetwork::TinyNetwork(std::array<int, 3> observation_shape, int n_actions, bool normalize_outputs)
    : AlphazeroNetwork(Tiny(observation_shape, n_actions, normalize_outputs), torch::kCPU, NetworkType::TinyNetwork),
    observation_shape_(observation_shape),
    n_actions_(n_actions),
    normalize_outputs_{ normalize_outputs }
{
}
TinyNetwork::~TinyNetwork() = default;
std::unique_ptr<IAlphazeroNetwork> TinyNetwork::deepcopy()
{
    auto other_network = std::make_unique<TinyNetwork>(observation_shape_, n_actions_,normalize_outputs_);
    deepcopyto(other_network);
    return other_network;
}

std::unique_ptr<IAlphazeroNetwork> TinyNetwork::copy()
{
    return std::make_unique<TinyNetwork>(*this);
}


void TinyNetwork::save_full(const std::string& file_path)
{
    torch::serialize::OutputArchive full_archive{};
    torch::serialize::OutputArchive mod_archive{};
    torch::Tensor network_type_tensor = torch::tensor({ static_cast<int>(NetworkType::TinyNetwork) }, torch::kInt);
    full_archive.write(NETWORK_TYPE_KEY, network_type_tensor);

    mod_->save(mod_archive);
    full_archive.write(MODULE_KEY, mod_archive);

    torch::Tensor shape_tensor = torch::tensor({ observation_shape_[0],observation_shape_[1],observation_shape_[2] }, torch::kInt);
    full_archive.write(OBSERVATION_SHAPE_KEY, shape_tensor);

    torch::Tensor n_actions_tensor = torch::tensor({ n_actions_ }, torch::kInt);
    full_archive.write(ACTIONS_KEY, n_actions_tensor);

    full_archive.save_to(file_path);
}

void TinyNetwork::load_mod(torch::serialize::InputArchive& mod_iarcive)
{
    mod_->load(mod_iarcive);
}
std::unique_ptr<TinyNetwork> TinyNetwork::create_full(torch::serialize::InputArchive& iarchive)
{
    torch::serialize::InputArchive mod_archive{};
    iarchive.read(MODULE_KEY, mod_archive);

    torch::Tensor n_actions_tensor{};
    iarchive.read(ACTIONS_KEY, n_actions_tensor);
    int n_actions = n_actions_tensor.accessor<int, 1>()[0];


    torch::Tensor shape_tensor{};
    iarchive.read(OBSERVATION_SHAPE_KEY, shape_tensor);
    auto shape_accessor = shape_tensor.accessor<int, 1>();
    std::array<int, 3> shape_array{ {shape_accessor[0],shape_accessor[1],shape_accessor[2] } };

    torch::Tensor normalize_outputs_tensor{};
    iarchive.read(NORMALIZE_OUTPUTS_KEY, normalize_outputs_tensor);
    bool normalize_outputs = static_cast<bool>(normalize_outputs_tensor.accessor<int, 1>()[0]);
    auto network_ptr = std::make_unique<TinyNetwork>(shape_array, n_actions, normalize_outputs);
    network_ptr->load_mod(mod_archive);
    std::cout << "Loading Tiny network with \n";
    std::cout << "NActions " << n_actions << '\n';
    std::cout << "shape " << shape_array[0] << 'x' << shape_array[1] << 'x' << shape_array[2] << '\n';
    return network_ptr;
}

const std::string TinyNetwork::OBSERVATION_SHAPE_KEY = "ObservationShape";
const std::string TinyNetwork::ACTIONS_KEY = "NActions";
const std::string TinyNetwork::NORMALIZE_OUTPUTS_KEY = "NormalizeOutputs";
} // namespace rl::deeplearning::alphazero


