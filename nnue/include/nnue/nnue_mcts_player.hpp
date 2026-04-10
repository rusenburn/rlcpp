#ifndef RL_NNUE_NNUE_MCTS_PLAYER_HPP_
#define RL_NNUE_NNUE_MCTS_PLAYER_HPP_


#include <chrono>
#include <memory>
#include <common/player.hpp>
#include <games/migoyugo_light.hpp>
#include "nnue_model.hpp"

namespace rl::players
{
class NNUEMctsPlayer : public rl::common::IPlayer
{
private:
    int n_game_actions_;
    NNUEModel model_;
    int minimum_simulations_;
    std::chrono::duration<int, std::milli> duration_in_millis_;
    float cpuct_;
public:
    NNUEMctsPlayer(
        NNUEModel model,
        int n_game_actions,
        int minimum_simulations,
        std::chrono::duration<int, std::milli> duration_in_millis,
        float cpuct_);
    ~NNUEMctsPlayer()override;
    int choose_action(const std::unique_ptr<rl::common::IState>& state_ptr)override;
};

}

#endif
