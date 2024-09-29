#ifndef RL_PLAYERS_AMCTS2_PLAYER_HPP_
#define RL_PLAYERS_AMCTS2_PLAYER_HPP_

#include <common/player.hpp>

#include "evaluator.hpp"
#include "search_tree.hpp"
#include <memory>
#include <chrono>
#include <common/concurrent_player.hpp>


namespace rl::players
{
using IPlayer = rl::common::IPlayer;
using IConcurrentPlayer = rl::common::IConcurrentPlayer;
class Amcts2Player : public IPlayer
{
private:
    int n_game_actions_;
    std::unique_ptr<IEvaluator> evaluator_ptr_;
    int minimum_simulations_;
    std::chrono::duration<int, std::milli> duration_in_millis_;
    float temperature_;
    float cpuct_;
    int max_async_simulations_;
    float dirichlet_epsilon_;
    float dirichlet_alpha_;
    float default_visits_;
    float default_wins_;

public:
    Amcts2Player(int n_game_actions,
        std::unique_ptr<IEvaluator> evaluator_ptr,
        int minimum_simulations,
        std::chrono::duration<int, std::milli> duration_in_millis,
        float temperature,
        float cpuct_,
        int max_async_simulations,
        float dirichlet_epsilon,
        float dirichlet_alpha,
        float default_visits = 1.0f,
        float default_wins = -1.0f);
    ~Amcts2Player()override;
    int choose_action(const std::unique_ptr<rl::common::IState>& state_ptr) override;
};

class ConcurrentPlayer : public IConcurrentPlayer
{
private:
    int n_game_actions_;
    std::unique_ptr<IEvaluator> evaluator_ptr_;
    int minimum_simulations_;
    std::chrono::duration<int, std::milli> duration_in_millis_;
    float temperature_;
    float cpuct_;
    int max_async_simulations_;
    float dirichlet_epsilon_;
    float dirichlet_alpha_;
    float default_visits_;
    float default_wins_;
public:
    ConcurrentPlayer::ConcurrentPlayer(
        int n_game_actions,
        std::unique_ptr<IEvaluator> evaluator_ptr,
        int minimum_simulations,
        std::chrono::duration<int, std::milli> duration_in_millis,
        float temperature,
        float cpuct,
        int max_async_simulations,
        float dirichlet_epsilon,
        float dirichlet_alpha,
        float default_visits = 1.0f,
        float default_wins = -1.0f
    );
    ConcurrentPlayer::~ConcurrentPlayer()override;
    std::vector<int> choose_actions(const std::vector<const rl::common::IState*>& states_ptrs_ref)override;
    int choose_action(const std::unique_ptr<rl::common::IState>& state_ptr);
};
} // namespace rl::players

#endif