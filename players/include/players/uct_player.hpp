#ifndef RL_PLAYERS_UCT_PLAYER_HPP_
#define RL_PLAYERS_UCT_PLAYER_HPP_

#include <common/player.hpp>
#include <players/bandits/uct/uct.hpp>
#include <chrono>
#include <memory>
namespace rl::players
{
using IPlayer = rl::common::IPlayer;

class UctPlayer : public IPlayer
{
private:
    int minimum_simulations_;
    std::chrono::duration<int, std::milli> duration_in_millis_;
    float temperature_;
    float cuct_;

public:
    UctPlayer(int minimum_simulations, std::chrono::duration<int, std::milli> duration_in_millis,
        float temperature, float cuct);
    ~UctPlayer() override;
    int choose_action(const std::unique_ptr<rl::common::IState>& state_ptr) override;
};

} // namespace rl::players

#endif