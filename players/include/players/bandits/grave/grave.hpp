#ifndef RL_PLAYERS_BANDITS_GRAVE_GRAVE_HPP_
#define RL_PLAYERS_BANDITS_GRAVE_GRAVE_HPP_

#include <players/search_tree.hpp>
namespace rl::players
{
class Grave : public players::ISearchTree
{
    int n_game_actions_;
    int min_ref_count_;
    float b_squared_;
    bool save_illegal_amaf_actions;

public:
    Grave(int n_game_actions, int min_ref_count, float b_squared_, bool save_illegal_amaf_actions);
    ~Grave() override;
    std::vector<float> search(const rl::common::IState* state_ptr, int minimum_no_simulations, std::chrono::duration<int, std::milli> minimum_duration) override;
};
} // namespace rl::players

#endif