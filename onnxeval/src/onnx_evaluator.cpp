#include <stdexcept>
#include <onnxeval/onnx_evaluator.hpp>

namespace rl::onnxeval
{

OnnxEvaluator::OnnxEvaluator(std::shared_ptr<IOnnxSession> session_ptr,
    int n_actions,
    const std::array<int, 3>& observation_shape)
    : session_ptr_{ std::move(session_ptr) },
    n_actions_{ n_actions },
    observation_shape_{ observation_shape }
{
    if (session_ptr_ == nullptr)
    {
        throw std::runtime_error("OnnxEvaluator was constructed with a null session");
    }
}

OnnxEvaluator::~OnnxEvaluator() = default;

std::tuple<std::vector<float>, std::vector<float>> OnnxEvaluator::evaluate(const std::vector<const rl::common::IState*>& state_ptrs)
{
    std::vector<float> probs;
    std::vector<float> values;
    evaluate(state_ptrs, probs, values);
    return std::make_tuple(probs, values);
}

std::tuple<std::vector<float>, std::vector<float>> OnnxEvaluator::evaluate(const rl::common::IState* state_ptrs)
{
    std::vector<const rl::common::IState*> ptr_vec = { state_ptrs };
    std::vector<float> probs;
    std::vector<float> values;
    evaluate(ptr_vec, probs, values);
    return std::make_tuple(probs, values);
}

std::tuple<std::vector<float>, std::vector<float>> OnnxEvaluator::evaluate(const std::unique_ptr<rl::common::IState>& state_ptrs)
{
    std::vector<const rl::common::IState*> ptr_vec = { state_ptrs.get() };
    std::vector<float> probs;
    std::vector<float> values;
    evaluate(ptr_vec, probs, values);
    return std::make_tuple(probs, values);
}

void OnnxEvaluator::evaluate(const std::vector<const rl::common::IState*>& state_ptrs_vec,
    std::vector<float>& probs_out,
    std::vector<float>& values_out)
{
    const int n_states = static_cast<int>(state_ptrs_vec.size());
    if (n_states == 0)
    {
        return;
    }
    const int observation_size = observation_shape_.at(0) * observation_shape_.at(1) * observation_shape_.at(2);

    observations_.clear();
    actions_mask_.clear();
    observations_.reserve(static_cast<size_t>(n_states) * observation_size);
    actions_mask_.reserve(static_cast<size_t>(n_states) * n_actions_);

    for (auto& state_ptr : state_ptrs_vec)
    {
        for (auto& cell_value : state_ptr->get_observation())
        {
            observations_.emplace_back(cell_value);
        }
        for (auto cell_value : state_ptr->actions_mask())
        {
            actions_mask_.emplace_back(float(cell_value));
        }
    }

    if (observations_.size() != static_cast<size_t>(n_states) * observation_size)
    {
        throw std::runtime_error("while evaluating a state , got an error on its observation size");
    }

    raw_probs_.resize(static_cast<size_t>(n_states) * n_actions_);
    wdls_.resize(static_cast<size_t>(n_states) * 3);
    session_ptr_->run(observations_.data(), n_states, raw_probs_.data(), wdls_.data());

    probs_out.assign(static_cast<size_t>(n_states) * n_actions_, 0.0f);
    values_out.resize(n_states);

    for (int i = 0; i < n_states; i++)
    {
        const int probs_start = i * n_actions_;

        // Zero the illegal actions, then divide by what is left. No epsilon on
        // the denominator: NetworkEvaluator does not use one either, and a state
        // with no legal actions is a terminal position that never reaches an
        // evaluator.
        float sum = 0.0f;
        for (int j = 0; j < n_actions_; j++)
        {
            const float p = raw_probs_.at(probs_start + j) * actions_mask_.at(probs_start + j);
            probs_out.at(probs_start + j) = p;
            sum += p;
        }
        for (int j = 0; j < n_actions_; j++)
        {
            probs_out.at(probs_start + j) /= sum;
        }

        // wdl -> scalar, the same v = w - l the torch path uses.
        values_out.at(i) = wdls_.at(i * 3) - wdls_.at(i * 3 + 2);
    }
}

std::unique_ptr<rl::players::IEvaluator> OnnxEvaluator::clone() const
{
    return std::make_unique<OnnxEvaluator>(session_ptr_, n_actions_, observation_shape_);
}

std::unique_ptr<rl::players::IEvaluator> OnnxEvaluator::copy() const
{
    return std::make_unique<OnnxEvaluator>(session_ptr_, n_actions_, observation_shape_);
}

} // namespace rl::onnxeval
