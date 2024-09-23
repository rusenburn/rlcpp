#ifndef RL_UI_PLAYERS_UTILS_HPP_
#define RL_UI_PLAYERS_UTILS_HPP_

#include <memory>
#include <string>
#include <players/players.hpp>

namespace rl::ui
{

class PlayerInfoFull
{
public:
    PlayerInfoFull::PlayerInfoFull(std::unique_ptr<rl::common::IPlayer> player_ptr, std::string name);
    PlayerInfoFull::~PlayerInfoFull();
    std::unique_ptr<rl::common::IPlayer> player_ptr_;
    std::string name_;
};

std::unique_ptr<PlayerInfoFull> get_default_g_player(rl::common::IState* state_ptr, int n_sims, std::chrono::duration<int, std::milli> minimum_duration);
std::unique_ptr<PlayerInfoFull> get_random_rollout_player_ptr(rl::common::IState* state_ptr, int n_sims, std::chrono::duration<int, std::milli> minimum_duration);
std::unique_ptr<PlayerInfoFull> get_network_amcts_player(rl::common::IState* state_ptr, int n_sims, std::chrono::duration<int, std::milli> minimum_duration, std::string load_name);
std::unique_ptr<PlayerInfoFull> get_network_amcts2_player(rl::common::IState* state_ptr, int n_sims, std::chrono::duration<int, std::milli> minimum_duration, std::string load_name);
std::unique_ptr<PlayerInfoFull> get_long_network_amcts2_player(rl::common::IState* state_ptr, int n_sims, std::chrono::duration<int, std::milli> minimum_duration, std::string load_name);
std::unique_ptr<PlayerInfoFull> get_network_mcts_player(rl::common::IState* state_ptr, int n_sims, std::chrono::duration<int, std::milli> minimum_duration, std::string load_name);
std::unique_ptr<PlayerInfoFull> get_network_lm_mcts_player(rl::common::IState* state_ptr, int n_sims, std::chrono::duration<int, std::milli> minimum_duration, std::string load_name);
std::unique_ptr<PlayerInfoFull> get_network_evaluator_ptr(rl::common::IState* state_ptr, std::string load_name);
std::unique_ptr<PlayerInfoFull> get_tiny_network_mcts_player(rl::common::IState* state_ptr, int n_sims, std::chrono::duration<int, std::milli> minimum_duration, std::string load_name);



} // namespace rl::ui::players_utils

#endif