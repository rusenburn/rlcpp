#ifndef RL_PLAYERS_RANDOM_ACTION_PLAYER_HPP_
#define RL_PLAYERS_RANDOM_ACTION_PLAYER_HPP_
#include <common/player.hpp>

namespace rl::players
{
class RandomActionPlayer : public rl::common::IPlayer
{
private:
public:
    RandomActionPlayer();
    ~RandomActionPlayer() override;
    int choose_action(const std::unique_ptr<rl::common::IState>& state_ptr) override;
};

} // namespace rl::players

#endif