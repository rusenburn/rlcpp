#ifndef RL_DEEPLEARNING_NETWORK_EVALUATOR_HPP_
#define RL_DEEPLEARNING_NETWORK_EVALUATOR_HPP_

#include <memory>
#include <vector>
#include <torch/torch.h>
#include <players/evaluator.hpp>
#include <deeplearning/alphazero/networks/az.hpp>
namespace rl::deeplearning
{
class NetworkEvaluator : public rl::players::IEvaluator
{
private:
    std::unique_ptr<rl::deeplearning::alphazero::IAlphazeroNetwork> network_ptr_;
    int n_actions_;
    std::array<int, 3> observation_shape_;
    void evaluate(const std::vector<const rl::common::IState*>& state_ptrs_vec, std::vector<float>& probs_out, std::vector<float>& values_out);

public:
    NetworkEvaluator(std::unique_ptr<rl::deeplearning::alphazero::IAlphazeroNetwork> network_ptr,
        int n_actions,
        const std::array<int, 3>& observation_shape);
    ~NetworkEvaluator()override;
    std::tuple<std::vector<float>, std::vector<float>> evaluate(const std::vector<const rl::common::IState*>& state_ptrs) override;
    std::tuple<std::vector<float>, std::vector<float>> evaluate(const rl::common::IState* state_ptrs) override;

    std::tuple<std::vector<float>, std::vector<float>> evaluate(const std::unique_ptr<rl::common::IState>& state_ptrs) override;
    std::unique_ptr<rl::players::IEvaluator> clone() const override;
    std::unique_ptr<rl::players::IEvaluator> copy() const override;
};

} // namespace rl::evaluators

#endif