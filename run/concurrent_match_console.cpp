#include "concurrent_match_console.hpp"
#include <iostream>
#include <filesystem>
#include <deeplearning/alphazero/networks/shared_res_nn.hpp>
#include <games/tictactoe.hpp>
#include <games/othello.hpp>
#include <games/english_draughts.hpp>
#include <games/walls.hpp>
#include <games/damma.hpp>
#include <games/santorini.hpp>
#include <players/players.hpp>
#include <common/concurrent_match.hpp>
namespace rl::run
{
ConcurrentMatchConsole::~ConcurrentMatchConsole() = default;

void ConcurrentMatchConsole::run()
{
    int choice = 0;
    do
    {
        print_current_settings();
        std::cout << "Do you want to run current settings?\n";
        std::cout << "[0] Yes, run\n";
        std::cout << "[1] No, I want to edit settings\n";
        std::cout << "[2] No, quit match\n";
        std::cin >> choice;
        if (choice == 1)
        {
            edit_settings();
        }
        else if (choice == 2)
        {
            return;
        }
    } while (choice != 0);


    start_match();

    std::cout << "Match has ended" << std::endl;
}

void ConcurrentMatchConsole::start_match()
{
    auto state_ptr = get_state_ptr();
    auto network_1_ptr = get_network_ptr(player_1_n_filters, player_1_fc_dims, player_1_blocks, player_1_load_name);
    auto evaluator_1_ptr = get_network_evaluator_ptr(network_1_ptr);
    auto p1_ptr = get_concurrent_player(evaluator_1_ptr, player_1_n_sims, player_1_duration);

    auto network_2_ptr = get_network_ptr(player_2_n_filters, player_2_fc_dims, player_2_blocks, player_2_load_name);
    auto evaluator_2_ptr = get_network_evaluator_ptr(network_2_ptr);
    auto p2_ptr = get_concurrent_player(evaluator_2_ptr, player_2_n_sims, player_2_duration);

    auto match = rl::common::ConcurrentMatch(state_ptr, p1_ptr.get(), p2_ptr.get(), n_sets_, n_sets_);

    std::cout << "Starting the match ..." << std::endl;
    auto p1_score_average = match.start();
    float ratio = static_cast<float>(p1_score_average + 1.0f) / (2.0f);
    std::cout << "Player 1 win ratio is " << ratio << std::endl;
}

void ConcurrentMatchConsole::print_current_settings()
{
    std::cout << "**** Concurrent Match Settings ****\n";
    std::cout << "[game] ";
    switch (state_index_)
    {
    case TIC_TAC_TOE_GAME:
        std::cout << "Tic Tac Toe\n";
        break;
    case OTHELLO_GAME:
        std::cout << "Othello\n";
        break;
    case ENGLISH_DRAUGHTS_GAME:
        std::cout << "English Draughts\n";
        break;
    case WALLS_GAME:
        std::cout << "Walls\n";
        break;
    case DAMMA_GAME:
        std::cout << "Damma\n";
        break;
    case SANTORINI_GAME:
        std::cout << "Santorini\n";
        break;
    default:
        std::cout << "Default\n";
        break;
    }

    std::cout << "[Number of sets] " << n_sets_ << std::endl;

}

void ConcurrentMatchConsole::edit_settings()
{
    int choice = 1;
    constexpr int NOTHING = 0;
    constexpr int GAME_SETTINGS = 1;
    constexpr int PLAYERS_SETTINGS = 2;
    constexpr int PLAYER_1_SETTINGS = 3;
    constexpr int PLAYER_2_SETTINGS = 4;
    constexpr int SETS_SETTINGS = 5;
    while (choice != 0)
    {
        std::cout << "What do you want to edit?\n";
        std::cout << "[" << NOTHING << "] Nothing , back to the previous menu\n";
        std::cout << "[" << GAME_SETTINGS << "] Game\n";
        std::cout << "[" << PLAYERS_SETTINGS << "] Players settings\n";
        std::cout << "[" << PLAYER_1_SETTINGS << "] Player 1 settings\n";
        std::cout << "[" << PLAYER_2_SETTINGS << "] Player 2 settings\n";
        std::cout << "[" << SETS_SETTINGS << "] Number of sets\n";

        std::cin >> choice;
        switch (choice)
        {
        case NOTHING:
            break;
        case GAME_SETTINGS:
            edit_game_settings();
            break;
        case PLAYERS_SETTINGS:
            edit_all_players_settings();
            break;
        case PLAYER_1_SETTINGS:
            edit_player_1_settings();
            break;
        case PLAYER_2_SETTINGS:
            edit_player_2_settings();
            break;
        case SETS_SETTINGS:
            std::cout << "Number of sets (" << n_sets_ << ") :";
            std::cin >> n_sets_;
            break;
        default:
            break;
        }
    }
}

void ConcurrentMatchConsole::edit_game_settings()
{
    int choice = 1;

    std::cout << "Choose game\n";
    std::cout << "[" << TIC_TAC_TOE_GAME << "] Tic Tac Toe\n";
    std::cout << "[" << OTHELLO_GAME << "] Othello\n";
    std::cout << "[" << ENGLISH_DRAUGHTS_GAME << "] English  Draughts\n";
    std::cout << "[" << WALLS_GAME << "] Walls\n";
    std::cout << "[" << DAMMA_GAME << "] DAMMA\n";
    std::cout << "[" << SANTORINI_GAME << "] Santorini\n";

    std::cout << std::endl;

    std::cin >> choice;
    switch (choice)
    {
    case TIC_TAC_TOE_GAME:
        state_index_ = TIC_TAC_TOE_GAME;
        break;
    case OTHELLO_GAME:
        state_index_ = OTHELLO_GAME;
        break;
    case ENGLISH_DRAUGHTS_GAME:
        state_index_ = ENGLISH_DRAUGHTS_GAME;
        break;
    case WALLS_GAME:
        state_index_ = WALLS_GAME;
        break;
    case DAMMA_GAME:
        state_index_ = DAMMA_GAME;
        break;
    case SANTORINI_GAME:
        state_index_ = SANTORINI_GAME;
        break;
    default:
        break;
    }
}

void ConcurrentMatchConsole::edit_all_players_settings()
{
    int choice = 1;
    constexpr int BACK_TO_MENU = 0;
    constexpr int N_SIMS_CHOICE = 1;
    constexpr int DURATION_CHOICE = 2;
    constexpr int NETWORK_FILTERS_CHOICE = 3;
    constexpr int NETWORK_FC_DIMS_CHOICE = 4;
    constexpr int NETWORK_BLOCKS_CHOICE = 5;
    constexpr int NETWORK_LOAD_NAME_CHOICE = 6;
    while (choice != 0)
    {
        std::cout << "All players settings\n";
        std::cout << "[" << BACK_TO_MENU << "] Nothing, back to previous menu\n";
        std::cout << "[" << N_SIMS_CHOICE << "] Simulations\n";
        std::cout << "[" << DURATION_CHOICE << "] Duration in milli seconds\n";
        std::cout << "[" << NETWORK_FILTERS_CHOICE << "] Network filters [If exits]\n";
        std::cout << "[" << NETWORK_FC_DIMS_CHOICE << "] Network Fc dims [If exits]\n";
        std::cout << "[" << NETWORK_BLOCKS_CHOICE << "] Network Blocks [If exits]\n";
        std::cout << "[" << NETWORK_LOAD_NAME_CHOICE << "] Network Load name [If exits]\n";
        std::cout << std::endl;
        std::cin >> choice;

        switch (choice)
        {
        case BACK_TO_MENU:
            break;
        case N_SIMS_CHOICE:
            std::cout << "Player 1 Simulations (" << player_1_n_sims << ") : " << std::endl;
            std::cout << "Player 2 Simulations (" << player_2_n_sims << ") : " << std::endl;
            std::cin >> player_1_n_sims;
            player_2_n_sims = player_1_n_sims;
            break;
        case DURATION_CHOICE:
            std::cout << "Duration in millis (" << player_1_duration.count() << ") : " << std::endl;
            std::cout << "Duration in millis (" << player_2_duration.count() << ") : " << std::endl;
            int d;
            std::cin >> d;
            player_1_duration = std::chrono::milliseconds(d);
            player_2_duration = std::chrono::milliseconds(d);
            break;

        case NETWORK_FILTERS_CHOICE:
            std::cout << "Network filters (" << player_1_n_filters << ") : " << std::endl;
            std::cout << "Network filters (" << player_2_n_filters << ") : " << std::endl;
            std::cin >> player_1_n_filters;
            player_2_n_filters = player_1_n_filters;
            break;

        case NETWORK_FC_DIMS_CHOICE:
            std::cout << "Network Fc dims (" << player_1_fc_dims << ") : " << std::endl;
            std::cout << "Network Fc dims (" << player_2_fc_dims << ") : " << std::endl;
            std::cin >> player_1_fc_dims;
            player_2_fc_dims = player_1_fc_dims;
            break;
        case NETWORK_BLOCKS_CHOICE:
            std::cout << "Network blocks (" << player_1_blocks << ") : " << std::endl;
            std::cout << "Network blocks (" << player_2_blocks << ") : " << std::endl;
            std::cin >> player_1_blocks;
            player_2_blocks = player_1_blocks;
            break;
        case NETWORK_LOAD_NAME_CHOICE:
            std::cout << "Network load name (" << player_1_load_name << ") : " << std::endl;
            std::cout << "Network load name (" << player_2_load_name << ") : " << std::endl;
            char load[50];
            std::cin >> load;
            player_1_load_name = std::string(load);
            player_2_load_name = std::string(load);
            break;
        default:
            break;
        }
    }
}

void ConcurrentMatchConsole::edit_player_1_settings()
{
    int choice = 1;
    constexpr int BACK_TO_MENU = 0;
    constexpr int N_SIMS_CHOICE = 1;
    constexpr int DURATION_CHOICE = 2;
    constexpr int NETWORK_FILTERS_CHOICE = 3;
    constexpr int NETWORK_FC_DIMS_CHOICE = 4;
    constexpr int NETWORK_BLOCKS_CHOICE = 5;
    constexpr int NETWORK_LOAD_NAME_CHOICE = 6;
    while (choice != 0)
    {
        std::cout << "Player 1 settings\n";
        std::cout << "[" << BACK_TO_MENU << "] Nothing, back to previous menu\n";
        std::cout << "[" << N_SIMS_CHOICE << "] Simulations\n";
        std::cout << "[" << DURATION_CHOICE << "] Duration in milli seconds\n";
        std::cout << "[" << NETWORK_FILTERS_CHOICE << "] Network filters [If exits]\n";
        std::cout << "[" << NETWORK_FC_DIMS_CHOICE << "] Network Fc dims [If exits]\n";
        std::cout << "[" << NETWORK_BLOCKS_CHOICE << "] Network Blocks [If exits]\n";
        std::cout << "[" << NETWORK_LOAD_NAME_CHOICE << "] Network Load name [If exits]\n";
        std::cout << std::endl;
        std::cin >> choice;

        switch (choice)
        {
        case BACK_TO_MENU:
            break;
        case N_SIMS_CHOICE:
            std::cout << "Player 1 Simulations (" << player_1_n_sims << ") : " << std::endl;
            std::cin >> player_1_n_sims;
            break;
        case DURATION_CHOICE:
            std::cout << "Duration in millis (" << player_1_duration.count() << ") : " << std::endl;
            int d;
            std::cin >> d;
            player_1_duration = std::chrono::milliseconds(d);
            break;

        case NETWORK_FILTERS_CHOICE:
            std::cout << "Network filters (" << player_1_n_filters << ") : " << std::endl;
            std::cin >> player_1_n_filters;
            break;

        case NETWORK_FC_DIMS_CHOICE:
            std::cout << "Network Fc dims (" << player_1_fc_dims << ") : " << std::endl;
            std::cin >> player_1_fc_dims;
            break;
        case NETWORK_BLOCKS_CHOICE:
            std::cout << "Network blocks (" << player_1_blocks << ") : " << std::endl;
            std::cin >> player_1_blocks;
            break;
        case NETWORK_LOAD_NAME_CHOICE:
            std::cout << "Network load name (" << player_1_load_name << ") : " << std::endl;
            char load[50];
            std::cin >> load;
            player_1_load_name = std::string(load);
            break;
        default:
            break;
        }
    }
}

void ConcurrentMatchConsole::edit_player_2_settings()
{
    int choice = 1;
    constexpr int BACK_TO_MENU = 0;
    constexpr int N_SIMS_CHOICE = 1;
    constexpr int DURATION_CHOICE = 2;
    constexpr int NETWORK_FILTERS_CHOICE = 3;
    constexpr int NETWORK_FC_DIMS_CHOICE = 4;
    constexpr int NETWORK_BLOCKS_CHOICE = 5;
    constexpr int NETWORK_LOAD_NAME_CHOICE = 6;
    while (choice != 0)
    {
        std::cout << "Player 2 settings\n";
        std::cout << "[" << BACK_TO_MENU << "] Nothing, back to previous menu\n";
        std::cout << "[" << N_SIMS_CHOICE << "] Simulations\n";
        std::cout << "[" << DURATION_CHOICE << "] Duration in milli seconds\n";
        std::cout << "[" << NETWORK_FILTERS_CHOICE << "] Network filters [If exits]\n";
        std::cout << "[" << NETWORK_FC_DIMS_CHOICE << "] Network Fc dims [If exits]\n";
        std::cout << "[" << NETWORK_BLOCKS_CHOICE << "] Network Blocks [If exits]\n";
        std::cout << "[" << NETWORK_LOAD_NAME_CHOICE << "] Network Load name [If exits]\n";
        std::cout << std::endl;
        std::cin >> choice;

        switch (choice)
        {
        case BACK_TO_MENU:
            break;
        case N_SIMS_CHOICE:
            std::cout << "Player 2 Simulations (" << player_2_n_sims << ") : " << std::endl;
            std::cin >> player_2_n_sims;
            break;
        case DURATION_CHOICE:
            std::cout << "Duration in millis (" << player_2_duration.count() << ") : " << std::endl;
            int d;
            std::cin >> d;
            player_2_duration = std::chrono::milliseconds(d);
            break;

        case NETWORK_FILTERS_CHOICE:
            std::cout << "Network filters (" << player_2_n_filters << ") : " << std::endl;
            std::cin >> player_2_n_filters;
            break;

        case NETWORK_FC_DIMS_CHOICE:
            std::cout << "Network Fc dims (" << player_2_fc_dims << ") : " << std::endl;
            std::cin >> player_2_fc_dims;
            break;
        case NETWORK_BLOCKS_CHOICE:
            std::cout << "Network blocks (" << player_2_blocks << ") : " << std::endl;
            std::cin >> player_2_blocks;
            break;
        case NETWORK_LOAD_NAME_CHOICE:
            std::cout << "Network load name (" << player_2_load_name << ") : " << std::endl;
            char load[50];
            std::cin >> load;
            player_2_load_name = std::string(load);
            break;
        default:
            break;
        }
    }
}

IStatePtr ConcurrentMatchConsole::get_state_ptr()
{
    switch (state_index_)
    {
    case TIC_TAC_TOE_GAME:
        return rl::games::TicTacToeState::initialize();
        break;
    case OTHELLO_GAME:
        return rl::games::OthelloState::initialize();
        break;
    case ENGLISH_DRAUGHTS_GAME:
        return rl::games::EnglishDraughtState::initialize();
        break;
    case WALLS_GAME:
        return rl::games::WallsState::initialize();
        break;
    case DAMMA_GAME:
        return rl::games::DammaState::initialize();
        break;
    case SANTORINI_GAME:
        return rl::games::SantoriniState::initialize();
        break;
    default:
        throw "";
        break;
    }
}

INetworkPtr ConcurrentMatchConsole::get_network_ptr(int filters, int fc_dims, int blocks, const std::string& load_name)
{
    auto state_pr = get_state_ptr();
    auto network_ptr = std::make_unique<rl::deeplearning::alphazero::SharedResNetwork>(state_pr->get_observation_shape(), state_pr->get_n_actions(),
        filters, fc_dims, blocks);
    auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    const std::string folder_name = "../checkpoints";
    std::filesystem::path folder(folder_name);
    std::filesystem::path file_path;
    file_path = folder / load_name;
    network_ptr->load(file_path.string());
    network_ptr->to(device);
    int n_examples = 1;
    int n_channels = state_pr->get_observation_shape()[0];
    int n_rows = state_pr->get_observation_shape()[1];
    int n_cols = state_pr->get_observation_shape()[2];

    network_ptr->forward(torch::randn(n_examples * n_channels * n_rows * n_cols).to(device).reshape({ n_examples, n_channels, n_rows, n_cols }));
    return network_ptr;
}

std::unique_ptr<rl::players::IEvaluator>  ConcurrentMatchConsole::get_network_evaluator_ptr(INetworkPtr& network_ptr)
{
    auto state_ptr = get_state_ptr();
    return std::make_unique<rl::deeplearning::NetworkEvaluator>(network_ptr->copy(), state_ptr->get_n_actions(), state_ptr->get_observation_shape());
}

IConcurrentPlayerPtr ConcurrentMatchConsole::get_concurrent_player(std::unique_ptr<rl::players::IEvaluator>& evaluator_ptr, int n_sims, std::chrono::duration<int, std::milli> minimum_duration)
{
    auto state_ptr = get_state_ptr();
    return std::make_unique<rl::players::ConcurrentPlayer>(state_ptr->get_n_actions(), evaluator_ptr->copy(), n_sims, minimum_duration, 0.5f, 2.0f, 8, 0.0f, -1.0f);
}
} // namespace rl::run


