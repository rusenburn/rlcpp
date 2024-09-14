#ifndef RL_RUN_MATCH_CONSOLE_HPP_
#define RL_RUN_MATCH_CONSOLE_HPP_

#include <chrono>
#include <common/player.hpp>
#include <memory>
#include <deeplearning/alphazero/networks/az.hpp>
#include <deeplearning/network_evaluator.hpp>
#include "console.hpp"
#include <common/match.hpp>
namespace rl::run
{
using IState = rl::common::IState;
using IStatePtr = std::unique_ptr<IState>;
using IPlayer = rl::common::IPlayer;
using IPlayerPtr = std::unique_ptr<IPlayer>;
using INetwork = rl::deeplearning::alphazero::IAlphazeroNetwork;
using INetworkPtr = std::unique_ptr<INetwork>;
using NetworkEvaluator = rl::deeplearning::NetworkEvaluator;

class MatchConsole : public IConsole
{
private:
    static constexpr int NETWORK_AMCTS_PLAYER = 0;
    static constexpr int NETWORK_MCTS_PLAYER = 1;
    static constexpr int CPUCT_RANDOM_ROLLOUT_MCTS = 2;
    static constexpr int UCT_PLAYER = 3;
    static constexpr int G_PLAYER = 4;
    static constexpr int MC_RAVE_PLAYER = 5;
    static constexpr int HUMAN_PLAYER = 6;
    static constexpr int RANDOM_ACTION_PLAYER = 7;
    static constexpr int NETWORK_EVALUATOR_PLAYER = 8;
    static constexpr int NETWORK_CPU_MCTS_PLAYER = 9;
    static constexpr int NETWORK_AMCTS2_PLAYER = 10;
    static constexpr int NETWORK_CONCURRENT_PLAYER = 11;

    static constexpr int TIC_TAC_TOE_GAME = 0;
    static constexpr int OTHELLO_GAME = 1;
    static constexpr int ENGLISH_DRAUGHTS_GAME = 2;
    static constexpr int WALLS_GAME = 3;
    static constexpr int DAMMA_GAME = 4;
    static constexpr int SANTORINI_GAME = 5;

    int state_index_{ OTHELLO_GAME };
    void print_current_settings();
    void edit_settings();
    void start_match();
    void edit_game_settings();
    void edit_player_0_settings();
    void edit_player_1_settings();

    IPlayerPtr get_player(int player_type, int n_sims, std::chrono::duration<int, std::milli> minimum_duration,
        int filters, int fc_dims, int blocks, std::string load_name);
    IStatePtr get_state_ptr();
    INetworkPtr get_network_ptr(int filters, int fc_dims, int blocks, std::string load_name);
    INetworkPtr get_tiny_network_ptr(std::string load_name);
    std::unique_ptr<rl::players::IEvaluator> get_network_evaluator_ptr(INetworkPtr& network_ptr);
    IPlayerPtr get_amcts_player(std::unique_ptr<rl::players::IEvaluator>& evaluator_ptr, int n_sims, std::chrono::duration<int, std::milli> minimum_duration);
    IPlayerPtr get_mcts_player(std::unique_ptr<rl::players::IEvaluator>& evaluator_ptr, int n_sims, std::chrono::duration<int, std::milli> minimum_duration);
    std::unique_ptr<rl::players::IEvaluator> get_random_rollout_evaluator_ptr();
    IPlayerPtr get_default_uct_player(int n_sims, std::chrono::duration<int, std::milli> minimum_duration);
    IPlayerPtr get_default_g_player(int n_sims, std::chrono::duration<int, std::milli> minimum_duration);
    IPlayerPtr get_default_mc_rave_player(int n_sims, std::chrono::duration<int, std::milli> minimum_duration);
    IPlayerPtr get_human_player();
    IPlayerPtr get_amcts2_player(std::unique_ptr<rl::players::IEvaluator>& evaluator_ptr, int n_sims, std::chrono::duration<int, std::milli> minimum_duration);
    IPlayerPtr get_concurrent_player(std::unique_ptr<rl::players::IEvaluator>& evaluator_ptr, int n_sims, std::chrono::duration<int, std::milli> minimum_duration);
    
    int pick_player_type();

    // Player 0
    int player_0_type_{ NETWORK_AMCTS_PLAYER };
    int player_0_n_sims{ 2 };
    std::chrono::duration<int, std::milli> player_0_duration{ 100 };
    int player_0_n_filters{ 128 };
    int player_0_fc_dims{ 512 };
    int player_0_blocks{ 5 };
    std::string player_0_load_name{ "temp.pt" };

    // Player 1
    int player_1_type_{ NETWORK_AMCTS_PLAYER };
    int player_1_n_sims{ 2 };
    std::chrono::duration<int, std::milli> player_1_duration{ 100 };
    int player_1_n_filters{ 128 };
    int player_1_fc_dims{ 512 };
    int player_1_blocks{ 5 };
    std::string player_1_load_name{ "temp.pt" };

    // match
    int n_sets_{ 100 };
    bool render_{ false };
    std::shared_ptr<rl::common::Observer<std::unique_ptr<rl::common::IState>&>> observer_;

public:
    MatchConsole(/* args */);
    ~MatchConsole() override;
    void run();
    void render(std::unique_ptr<rl::common::IState>& state_ptr)
    {
        state_ptr->render();
    }
};
} // namespace rl::run

#endif