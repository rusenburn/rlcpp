#ifndef RL_PLAYERS_BANDITS_MCRAVE_NODE_HPP_
#define RL_PLAYERS_BANDITS_MCRAVE_NODE_HPP_

#include <common/state.hpp>
#include <memory>
#include <optional>
#include <utility>
#include <chrono>
namespace rl::players
{
    class McraveNode
    {
        using IState = rl::common::IState;
    private:
        std::unique_ptr<IState> state_ptr_;
        int n_game_actions_;
        const int player_;
        std::optional<bool> terminal_;
        bool is_leaf_node;
        std::optional<float> game_result_;
        std::vector<std::unique_ptr<McraveNode>> children_;
        std::vector<int> nsa_;
        std::vector<float> qsa_;

        std::vector<int> n_sa_;
        std::vector<float> q_sa_;

        std::vector<bool> actions_legality_;

        std::pair<float,int> simulateOne(float b, std::vector<int> &out_our_actions, std::vector<int> &out_their_actions);
        void getFinalProbabilities(std::vector<float> &out_actions_probs);
        int selectMove(float b);
        virtual void heuristic();
        std::pair<float,int> simDefault(std::vector<int> &out_our_actions, std::vector<int> &out_their_actions);
        int player();

    public:
        McraveNode(std::unique_ptr<IState> state_ptr,int n_game_actions);
        ~McraveNode();
        void search(int minimum_simulations_count, std::chrono::duration<int, std::milli> minimum_duration, float b,std::vector<float> &out_actions_probs);
    };
} // namespace rl::players


#endif