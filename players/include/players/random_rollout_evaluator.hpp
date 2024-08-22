#ifndef RL_PLAYERS_RANDOM_ROLLOUT_EVALUATOR_HPP_
#define RL_PLAYERS_RANDOM_ROLLOUT_EVALUATOR_HPP_

#include "evaluator.hpp"

namespace rl::players
{
class RandomRolloutEvaluator : public IEvaluator
{
private:
    int n_game_actions_{};
    void evaluate(const rl::common::IState* state_ptr, std::vector<float>& probs, std::vector<float>& values) const;
    int choose_action(const std::vector<bool>& masks)const;

public:
    RandomRolloutEvaluator(int n_game_actions);
    ~RandomRolloutEvaluator() override;

    std::tuple<std::vector<float>, std::vector<float>> evaluate(const std::vector<const rl::common::IState*>& state_ptrs) override;
    std::tuple<std::vector<float>, std::vector<float>> evaluate(const rl::common::IState* state_ptrs) override;
    std::tuple<std::vector<float>, std::vector<float>> evaluate(const std::unique_ptr<rl::common::IState>& state_ptrs) override;
    std::unique_ptr<IEvaluator> clone() const override;
    std::unique_ptr<IEvaluator> copy() const override;
};

} // namespace rl::evaluators

#endif