#ifndef RL_PLAYERS_BANDITS_UCT_UCT_NODE_HPP_
#define RL_PLAYERS_BANDITS_UCT_UCT_NODE_HPP_
#include <common/state.hpp>
#include <players/search_tree.hpp>
#include <optional>
#include <memory>
#include <utility>
namespace rl::players
{
class UctNode
{
private:
    const std::unique_ptr<const rl::common::IState> state_ptr_;
    int n_game_actions_;
    float cuct_;
    std::optional<bool> terminal_;
    bool is_game_result_cached_;
    std::optional<float> game_result_;
    std::vector<bool> actions_legality_;
    std::vector<std::unique_ptr<UctNode>> children_;

    int n_;
    // Q(s,a)
    std::vector<float> qsa_;

    // N(n,a)
    std::vector<int> nsa_;

    std::pair<float, int> simulateOne();
    int findBestAction();
    void getFinalProbabilities(float temperature, std::vector<float>& out_actions_probs);
    std::pair<float, int> rollout(std::unique_ptr<rl::common::IState> state_ptr);

public:
    UctNode(const std::unique_ptr<const rl::common::IState> state_ptr, int n_game_actions, float cuct);
    void search(int minimum_simulations, std::chrono::duration<int, std::milli> minimum_duration, float uct, float temperature, std::vector<float>& out_actions_probs);
    ~UctNode();
};
} // namespace players

#endif