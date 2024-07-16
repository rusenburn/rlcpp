#ifndef RL_PLAYERS_LM_MCTS_HPP_
#define RL_PLAYERS_LM_MCTS_HPP_

#include <players/search_tree.hpp>
#include <players/evaluator.hpp>
#include "lm_mcts_node.hpp"
namespace rl::players
{
    class LMMcts : public ISearchTree
    {

    private:
        std::unique_ptr<IEvaluator> evaluator_ptr_;
        float cpuct_;
        float temperature_;
        // std::unique_ptr<LMMctsNode> root_node;
        int n_game_actions_;

    public:
        LMMcts(std::unique_ptr<IEvaluator> evaluator_ptr, int n_game_actions, float cpuct, float temperature);
        ~LMMcts() override;
        std::vector<float> search(const rl::common::IState *state_ptr, int minimum_no_simulations, std::chrono::duration<int, std::milli> minimum_duration) override;
    };
} // namespace rl::players

#endif