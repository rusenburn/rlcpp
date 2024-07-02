#include <deeplearning/network_evaluator.hpp>

namespace rl::deeplearning
{

    NetworkEvaluator::NetworkEvaluator(std::unique_ptr<rl::deeplearning::alphazero::IAlphazeroNetwork> network_ptr,
                                       int n_actions,
                                       const std::array<int,3> &observation_shape)
        : network_ptr_{std::move(network_ptr)},
          n_actions_{n_actions},
          observation_shape_{observation_shape}
    {
    }

    NetworkEvaluator::~NetworkEvaluator() = default;

    std::tuple<std::vector<float>, std::vector<float>> NetworkEvaluator::evaluate(const std::vector<const rl::common::IState *> &state_ptrs)
    {
        std::vector<float> probs;
        std::vector<float> values;
        evaluate(state_ptrs, probs, values);
        return std::make_tuple(probs, values);
    }

    std::tuple<std::vector<float>, std::vector<float>> NetworkEvaluator::evaluate(const rl::common::IState *state_ptrs)
    {
        std::vector<const rl::common::IState *> ptr_vec = {state_ptrs};
        std::vector<float> probs;
        std::vector<float> values;
        evaluate(ptr_vec, probs, values);
        return std::make_tuple(probs, values);
    }

    std::tuple<std::vector<float>, std::vector<float>> NetworkEvaluator::evaluate(const std::unique_ptr<rl::common::IState> &state_ptrs)
    {
        std::vector<const rl::common::IState *> ptr_vec = {state_ptrs.get()};
        std::vector<float> probs;
        std::vector<float> values;
        evaluate(ptr_vec, probs, values);
        return std::make_tuple(probs, values);
    }

    void NetworkEvaluator::evaluate(const std::vector<const rl::common::IState *> &state_ptrs_vec, std::vector<float> &probs_out, std::vector<float> &values_out)
    {

        int n_states = state_ptrs_vec.size();
        if (n_states == 0)
        {
            return;
        }
        int observation_size = observation_shape_.at(0) * observation_shape_.at(1) * observation_shape_.at(2);
        std::vector<float> observations;
        std::vector<float> actions_mask;
        observations.reserve(n_states * observation_size);
        actions_mask.reserve(n_states * n_actions_);
        probs_out.reserve(n_states * n_actions_);
        values_out.reserve(n_states);

        for (auto &state_ptr : state_ptrs_vec)
        {
            for (auto &cell_value : state_ptr->get_observation())
            {
                observations.emplace_back(cell_value);
            }
            for (auto &cell_value : state_ptr->actions_mask())
            {
                actions_mask.emplace_back(float(cell_value));
            }
        }

        if (observations.size() != n_states * observation_size)
        {
            throw std::runtime_error("while evaluating a state , got an error on its observation size");
        }

        torch::NoGradGuard nograd;
        torch::Tensor observations_tensor = torch::tensor(observations, torch::kFloat32)
                                                .to(network_ptr_->device())
                                                .reshape({n_states, observation_shape_.at(0), observation_shape_.at(1), observation_shape_.at(2)});

        torch::Tensor actions_mask_tensor = torch::tensor(actions_mask, torch::kFloat32).to(network_ptr_->device()).reshape({n_states, n_actions_});
        auto rs = network_ptr_->forward(observations_tensor);
        auto probs_tensor = std::get<0>(rs);
        auto legal_probs_tensor = probs_tensor * actions_mask_tensor; // assign illegal probs as zeroes
        legal_probs_tensor /= legal_probs_tensor.sum(-1, true);       // normalize legal_probs_tensor
        auto wdl_tensor = std::get<1>(rs);

        // convert wdl to values  simply by v = w - l for all states
        auto values_tensor = wdl_tensor.index({torch::indexing::Ellipsis, 0}) - wdl_tensor.index({torch::indexing::Ellipsis, 2});
        values_tensor = values_tensor.squeeze(-1);
        legal_probs_tensor = legal_probs_tensor.cpu();
        values_tensor = values_tensor.cpu();

        probs_out = std::vector<float>(legal_probs_tensor.data_ptr<float>(), legal_probs_tensor.data_ptr<float>() + legal_probs_tensor.numel());
        values_out = std::vector<float>(values_tensor.data_ptr<float>(), values_tensor.data_ptr<float>() + values_tensor.numel());
        // wdl_tensor = wdl_tensor.cpu();
        // auto wdls_vec = std::vector<float>(wdl_tensor.data_ptr<float>(), wdl_tensor.data_ptr<float>() + wdl_tensor.numel());
    }

    std::unique_ptr<rl::players::IEvaluator> NetworkEvaluator::clone() const
    {
        return std::make_unique<NetworkEvaluator>(network_ptr_->deepcopy(), n_actions_, observation_shape_);
    }
    std::unique_ptr<rl::players::IEvaluator> NetworkEvaluator::copy() const
    {
        return std::make_unique<NetworkEvaluator>(network_ptr_->copy(), n_actions_, observation_shape_);
    }
} // namespace rl::evaluators