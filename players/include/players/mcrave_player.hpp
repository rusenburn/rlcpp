#ifndef RL_PLAYERS_MCRAVE_PLAYER_HPP_
#define RL_PLAYERS_MCRAVE_PLAYER_HPP_

#include <common/player.hpp>
#include "bandits/mcrave/mcrave.hpp"

namespace rl::players
{
    class McravePlayer : public common::IPlayer
    {
    private:
        int minimum_simulations_;
        std::chrono::duration<int, std::milli> duration_in_millis_;
        float b_;

    public:
        McravePlayer(int minimum_simulations, std::chrono::duration<int, std::milli> duration_in_millis, float b = 0.1f);
        ~McravePlayer() override;
        int choose_action(const std::unique_ptr<rl::common::IState> &state_ptr) override;
    };

} // namespace rl::players

#endif