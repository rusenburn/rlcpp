#ifndef RL_PLAYERS_AMCTS_PLAYER_HPP_
#define RL_PLAYERS_AMCTS_PLAYER_HPP_

#include <common/player.hpp>

#include "evaluator.hpp"
#include "search_tree.hpp"
#include <memory>
#include <chrono>
namespace rl::players
{
    using IPlayer = rl::common::IPlayer;
    class AmctsPlayer : public IPlayer
    {
    private:
        int n_game_actions_;
        std::unique_ptr<IEvaluator> evaluator_ptr_;
        int minimum_simulations_;
        std::chrono::duration<int, std::milli> duration_in_millis_;
        float temperature_;
        float cpuct_;
        int max_async_simulations_;
        float default_visits_;
        float default_wins_;

    public:
        AmctsPlayer(int n_game_actions,
                    std::unique_ptr<IEvaluator> evaluator_ptr,
                    int minimum_simulations,
                    std::chrono::duration<int, std::milli> duration_in_millis,
                    float temperature,
                    float cpuct_,
                    int max_async_simulations = 4,
                    float default_visits = 1.0f,
                    float default_wins = -1.0f);
        ~AmctsPlayer()override;
        int choose_action(const std::unique_ptr<rl::common::IState> &state_ptr) override;
    };
} // namespace rl::players

#endif