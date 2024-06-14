#ifndef RL_PLAYERS_HUMAN_PLAYER_HPP_
#define RL_PLAYERS_HUMAN_PLAYER_HPP_

#include <common/player.hpp>
namespace rl::players
{
    class HumanPlayer : public rl::common::IPlayer
    {
    public:
        HumanPlayer();
        ~HumanPlayer() override;
        int choose_action(const std::unique_ptr<rl::common::IState> &state_ptr) override;
    };
} // namespace rl::players

#endif