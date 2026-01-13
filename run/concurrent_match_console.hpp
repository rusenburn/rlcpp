#ifndef RL_RUN_CONCURRENT_MATCH_CONSOLE_HPP_
#define RL_RUN_CONCURRENT_MATCH_CONSOLE_HPP_

#include <chrono>
#include <memory>
#include <common/concurrent_player.hpp>
#include <deeplearning/alphazero/networks/az.hpp>
#include <deeplearning/network_evaluator.hpp>
#include "console.hpp"
#include <common/match.hpp>

namespace rl::run
{
using IState = rl::common::IState;
using IStatePtr = std::unique_ptr<IState>;
using IConcurrentPlayer = rl::common::IConcurrentPlayer;
using IConcurrentPlayerPtr = std::unique_ptr<IConcurrentPlayer>;
using INetwork = rl::deeplearning::alphazero::IAlphazeroNetwork;
using INetworkPtr = std::unique_ptr<INetwork>;
using NetworkEvaluator = rl::deeplearning::NetworkEvaluator;

class ConcurrentMatchConsole : public IConsole
{
public:
    ~ConcurrentMatchConsole() override;
    void run() override;
private:
    static constexpr int TIC_TAC_TOE_GAME = 0;
    static constexpr int OTHELLO_GAME = 1;
    static constexpr int ENGLISH_DRAUGHTS_GAME = 2;
    static constexpr int WALLS_GAME = 3;
    static constexpr int DAMMA_GAME = 4;
    static constexpr int SANTORINI_GAME = 5;
    static constexpr int GOBBLET_GAME = 6;
    static constexpr int MIGOYUGO_GAME = 7;

    int state_index_{ OTHELLO_GAME };
    int n_sets_{ 32 };
    void start_match();
    void print_current_settings();
    void edit_settings();
    void edit_game_settings();
    void edit_all_players_settings();
    void edit_player_1_settings();
    void edit_player_2_settings();

    IStatePtr get_state_ptr();
    INetworkPtr get_network_ptr(int filters, int fc_dims, int blocks, const std::string& load_name);
    std::unique_ptr<rl::players::IEvaluator> get_network_evaluator_ptr(INetworkPtr& network_ptr);
    IConcurrentPlayerPtr get_concurrent_player(std::unique_ptr<rl::players::IEvaluator>& evaluator_ptr, int n_sims, std::chrono::duration<int, std::milli> minimum_duration);

    // Player 1
    int player_1_n_sims{ 100 };
    std::chrono::duration<int, std::milli> player_1_duration{ 0 };
    int player_1_n_filters{ 128 };
    int player_1_fc_dims{ 512 };
    int player_1_blocks{ 5 };
    std::string player_1_load_name{ "temp.pt" };

    // Player 2
    int player_2_n_sims{ 100 };
    std::chrono::duration<int, std::milli> player_2_duration{ 0 };
    int player_2_n_filters{ 128 };
    int player_2_fc_dims{ 512 };
    int player_2_blocks{ 5 };
    std::string player_2_load_name{ "temp.pt" };
};

} // namespace rl::run

#endif
