#ifndef RL_PLAYERS_GRAVE_PLAYER_HPP_
#define RL_PLAYERS_GRAVE_PLAYER_HPP_

#include <common/player.hpp>
#include <chrono>
namespace rl::players
{
class GravePlayer : public common::IPlayer
{
private:
    int minimum_simulations_;
    std::chrono::duration<int, std::milli> duration_in_millis_;
    int min_ref_count_;
    float b_squared_;
    bool save_illegal_amaf_actions_;

public:
    GravePlayer(int minimum_simulations, std::chrono::duration<int, std::milli> duration_in_millis, int min_ref_count = 15, float b_squared = 0.04f, bool save_illegal_amaf_actions = true);
    ~GravePlayer() override;
    int choose_action(const std::unique_ptr<rl::common::IState>& state_ptr) override;
};

} // namespace rl::players

#endif