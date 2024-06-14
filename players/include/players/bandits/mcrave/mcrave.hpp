#ifndef RL_PLAYERS_BANDITS_MCRAVE_MCRAVE_HPP_
#define RL_PLAYERS_BANDITS_MCRAVE_MCRAVE_HPP_

#include <players/search_tree.hpp>
namespace rl::players
{
    class Mcrave : public ISearchTree
    {
    private:
        int n_game_actions_;
        float b_;

    public:
        Mcrave(int n_game_actions, float b = 0.1f);
        std::vector<float> search(const rl::common::IState *state_ptr, int minimum_no_simulations, std::chrono::duration<int, std::milli> minimum_duration) override;
        ~Mcrave() override;
    };
} // namespace rl::players

#endif