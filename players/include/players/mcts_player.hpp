#ifndef RL_PLAYERS_MCTS_PLAYER_HPP_
#define RL_PLAYERS_MCTS_PLAYER_HPP_
#include <chrono>
#include <memory>
#include <common/player.hpp>
#include "evaluator.hpp"

namespace rl::players
{
class MctsPlayer : public rl::common::IPlayer
{
private:
    int n_game_actions_;
    std::unique_ptr<IEvaluator> evaluator_ptr_;
    int minimum_simulations_;
    std::chrono::duration<int, std::milli> duration_in_millis_;
    float temperature_;
    float cpuct_;

public:
    MctsPlayer(int n_game_actions,
        std::unique_ptr<IEvaluator> evaluator_ptr,
        int minimum_simulations,
        std::chrono::duration<int, std::milli> duration_in_millis,
        float temperature,
        float cpuct_);
    ~MctsPlayer()override;
    int choose_action(const std::unique_ptr<rl::common::IState>& state_ptr)override;
};

} // namespace rl::players


#endif