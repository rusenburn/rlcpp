#ifndef RL_PLAYERS_LM_MCTS_NODE_HPP_
#define RL_PLAYERS_LM_MCTS_NODE_HPP_

#include <memory>
#include <vector>
#include <optional>
#include <chrono>
#include <common/state.hpp>
#include <players/evaluator.hpp>
namespace rl::players
{
class LMMctsNode
{
public:
    LMMctsNode(int n_game_actions, float cpuct);
    ~LMMctsNode();
    std::vector<float> search_and_get_probs(std::unique_ptr<rl::common::IState>& state_ptr, std::unique_ptr<IEvaluator>& evaluator_ptr, int n_sims, std::chrono::duration<int, std::milli> minimum_duration, float temperature);

private:
    int n_game_actions_;
    float cpuct_;
    std::vector<int> legal_actions_;
    std::vector<std::unique_ptr<LMMctsNode>> children_;
    std::vector<float> probs_;
    int n_visits_{ 0 };
    std::vector<float> action_visits_;
    std::vector<float> delta_wins_;
    std::optional<bool> is_terminal_;
    std::optional<float> game_result_;
    std::pair<float, int> search(std::unique_ptr<rl::common::IState>& state_ptr, std::unique_ptr<IEvaluator>& evaluator_ptr);
    std::vector<float> get_probs(float temperature);
    std::pair<int, int> get_best_action_and_index();
};
} // namespace rl::players

#endif
