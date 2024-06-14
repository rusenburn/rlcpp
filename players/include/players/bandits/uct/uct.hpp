#ifndef RL_PLAYERS_BANDITS_UCT_UCT_HPP_
#define RL_PLAYERS_BANDITS_UCT_UCT_HPP_

#include <players/search_tree.hpp>

namespace rl::players
{
    class UctSearchTree : public rl::players::ISearchTree
    {
    private:
        int n_game_actions_;
        float cuct_;
        float temperature_;

    public:
        UctSearchTree(int n_game_actions, float cuct,float temperature);
        std::vector<float> search(const rl::common::IState *state_ptr, int minimum_no_simulations, std::chrono::duration<int, std::milli> minimum_duration) override;
    };
} // namespace rl::players
#endif
