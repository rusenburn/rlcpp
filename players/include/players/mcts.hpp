#ifndef RL_PLAYERS_MCTS_HPP_
#define RL_PLAYERS_MCTS_HPP_

#include <optional>
#include <memory>
#include "evaluator.hpp"
#include "search_tree.hpp"
#include <common/state.hpp>

namespace rl::players
{
    class MCTSNode
    {
    private:
        std::unique_ptr<rl::common::IState> state_ptr_;
        int n_game_actions_;
        float cpuct_;
        std::vector<bool> actions_mask_;
        std::vector<std::unique_ptr<MCTSNode>> children_;
        std::vector<float> probs_;
        int n_visits{0};
        std::vector<float> actions_visits_;
        std::vector<float> delta_wins_;
        std::optional<bool> is_terminal_;
        std::optional<float> game_result_;
        std::pair<float, int> search(std::unique_ptr<IEvaluator> &evaluator_ptr);
        std::vector<float> get_probs(float temperature);
        int get_best_action();

    public:
        MCTSNode(std::unique_ptr<rl::common::IState> state_ptr, int n_game_actions, float cpuct);
        ~MCTSNode();
        std::vector<float> search_and_get_probs(std::unique_ptr<IEvaluator> &evaluator_ptr, int n_sims, std::chrono::duration<int, std::milli> minimum_duration, float temperature);
    };
    class MCTS : public ISearchTree
    {

    private:
        std::unique_ptr<IEvaluator> evaluator_ptr_;
        float cpuct_;
        float temperature_;
        std::unique_ptr<MCTSNode> root_node;
        int n_game_actions_;

    public:
        MCTS(std::unique_ptr<IEvaluator> evaluator_ptr, int n_game_actions, float cpuct, float temperature);
        ~MCTS()override;
        std::vector<float> search(const rl::common::IState *state_ptr, int minimum_no_simulations, std::chrono::duration<int, std::milli> minimum_duration)override;
    };

} // namespace rl::searchTrees

#endif
