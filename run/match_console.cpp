#include <iostream>
#include <filesystem>
#include <cmath>
#include <common/match.hpp>
#include "match_console.hpp"
#include <players/players.hpp>
#include <players/random_rollout_evaluator.hpp>
#include <deeplearning/alphazero/networks/shared_res_nn.hpp>
#include <games/tictactoe.hpp>
#include <games/othello.hpp>
#include <games/english_draughts.hpp>
#include <games/walls.hpp>

namespace rl::run
{
    MatchConsole::MatchConsole() = default;
    MatchConsole::~MatchConsole() = default;
    void MatchConsole::run()
    {
        int choice{0};
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

    void MatchConsole::print_current_settings()
    {
    }

    void MatchConsole::edit_settings()
    {
        int choice = 1;
        while (choice != 0)
        {
            std::cout << "What do you want to edit?\n";
            std::cout << "[0] Nothing , back to the previous menu\n";
            std::cout << "[1] Game\n";
            std::cout << "[2] Player 0\n";
            std::cout << "[3] Player 1\n";
            std::cout << "[4] Number of sets\n";
            std::cout << "[5] Render \n";

            std::cin >> choice;
            switch (choice)
            {
            case 0:
                break;
            case 1:
                edit_game_settings();
                break;
            case 2:
                edit_player_0_settings();
                break;
            case 3:
                edit_player_1_settings();
                break;
            case 4:
                std::cout << "Number of sets (" << n_sets_ << ") :";
                std::cin >> n_sets_;
                break;
            case 5:
                std::cout << "Render (" << (render_ ? "T" : "F") << ") :";
                char in[50];
                std::cin >> in;
                if (in[0] == 'T' || in[0] == 't')
                {
                    render_ = true;
                }
                else if (in[0] == 'F' || in[0] == 'f')
                {
                    render_ = false;
                }
                break;
            default:
                break;
            }
        }
    }

    void MatchConsole::start_match()
    {
        auto state_ptr = get_state_ptr();

        auto player_0{get_player(
            player_0_type_,
            player_0_n_sims,
            player_0_duration,
            player_0_n_filters, player_0_fc_dims, player_0_blocks, player_0_load_name)};

        auto player_1{get_player(
            player_1_type_,
            player_1_n_sims,
            player_1_duration,
            player_1_n_filters, player_1_fc_dims, player_1_blocks, player_1_load_name)};

        auto match = rl::common::Match(std::move(state_ptr), std::move(player_0), std::move(player_1), n_sets_, render_);

        std::cout << "Starting the match ..." << std::endl;
        auto [p_0_score, p_1_score] = match.start();
        float ratio = (p_0_score + n_sets_) / (2 * n_sets_);

        std::cout << "Player 0 win ratio is " << ratio << std::endl;
    }

    void MatchConsole::edit_game_settings()
    {
        int choice = 1;
        while (choice != 0)
        {
            std::cout << "Choose game\n";
            std::cout << "[0] Nothing, back to previous menu\n";
            std::cout << "[1] Tic Tac Toe\n";
            std::cout << "[2] Othello\n";
            std::cout << "[3] English  Draughts";
            std::cout << "[4] Walls";

            std::cout << std::endl;

            std::cin >> choice;
            switch (choice)
            {
            case 0:
                break;
            case 1:
                state_index_ = TIC_TAC_TOE_GAME;
                break;
            case 2:
                state_index_ = OTHELLO_GAME;
                break;
            case 3:
                state_index_ = ENGLISH_DRAUGHTS_GAME;
                break;
            case 4:
                state_index_ = WALLS_GAME;
                break;
            default:
                break;
            }
        }
    }

    void MatchConsole::edit_player_0_settings()
    {
        int choice = 1;
        while (choice != 0)
        {
            std::cout << "Player 0 settings\n";
            std::cout << "[0] Nothing, back to previous menu\n";
            std::cout << "[1] Player type\n";
            std::cout << "[2] Simulations\n";
            std::cout << "[3] Duration in milli seconds\n";
            std::cout << "[4] Network filters [If exits]\n";
            std::cout << "[5] Network Fc dims [If exits]\n";
            std::cout << "[6] Network Blocks [If exits]\n";
            std::cout << "[7] Network Load name [If exits]\n";
            std::cout << std::endl;
            std::cin >> choice;

            switch (choice)
            {
            case 0:
                break;
            case 1:
                player_0_type_ = pick_player_type();
                break;
            case 2:
                std::cout << "Simulations (" << player_0_n_sims << ") : " << std::endl;
                std::cin >> player_0_n_sims;
                break;
            case 3:
                std::cout << "Duration in millis (" << player_0_duration.count() << ") : " << std::endl;
                int d;
                std::cin >> d;
                player_0_duration = std::chrono::duration<int, std::milli>(d);
                break;

            case 4:
                std::cout << "Network filters (" << player_0_n_filters << ") : " << std::endl;
                std::cin >> player_0_n_filters;
                break;

            case 5:
                std::cout << "Network Fc dims (" << player_0_fc_dims << ") : " << std::endl;
                std::cin >> player_0_fc_dims;
                break;
            case 6:
                std::cout << "Network blocks (" << player_0_blocks << ") : " << std::endl;
                std::cin >> player_0_blocks;
                break;
            case 7:
                std::cout << "Network load name (" << player_0_load_name << ") : " << std::endl;
                char load[50];
                std::cin >> load;
                player_0_load_name = std::string(load);
                break;
            default:
                break;
            }
        }
    }

    void MatchConsole::edit_player_1_settings()
    {
        int choice = 1;
        while (choice != 0)
        {
            std::cout << "Player 1 settings\n";
            std::cout << "[0] Nothing, back to previous menu\n";
            std::cout << "[1] Player type\n";
            std::cout << "[2] Simulations\n";
            std::cout << "[3] Duration in milli seconds\n";
            std::cout << "[4] Network filters [If exits]\n";
            std::cout << "[5] Network Fc dims [If exits]\n";
            std::cout << "[6] Network Blocks [If exits]\n";
            std::cout << "[7] Network Load name [If exits]\n";
            std::cout << std::endl;
            std::cin >> choice;

            switch (choice)
            {
            case 0:
                break;
            case 1:
                player_1_type_ = pick_player_type();
                break;
            case 2:
                std::cout << "Simulations (" << player_1_n_sims << ") : " << std::endl;
                std::cin >> player_1_n_sims;
                break;
            case 3:
                std::cout << "Duration in millis (" << player_1_duration.count() << ") : " << std::endl;
                int d;
                std::cin >> d;
                player_1_duration = std::chrono::duration<int, std::milli>(d);
                break;

            case 4:
                std::cout << "Network filters (" << player_1_n_filters << ") : " << std::endl;
                std::cin >> player_1_n_filters;
                break;

            case 5:
                std::cout << "Network Fc dims (" << player_1_fc_dims << ") : " << std::endl;
                std::cin >> player_1_fc_dims;
                break;
            case 6:
                std::cout << "Network blocks (" << player_1_blocks << ") : " << std::endl;
                std::cin >> player_1_blocks;
                break;
            case 7:
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

    IPlayerPtr MatchConsole::get_player(int player_type, int n_sims, std::chrono::duration<int, std::milli> minimum_duration,
                                        int filters, int fc_dims, int blocks, std::string load_name)
    {
        INetworkPtr network_ptr{nullptr};
        std::unique_ptr<rl::players::IEvaluator> evaluator_ptr{nullptr};
        switch (player_type)
        {
        case NETWORK_AMCTS_PLAYER:
            network_ptr = get_network_ptr(filters, fc_dims, blocks, load_name);
            evaluator_ptr = get_network_evaluator_ptr(network_ptr);
            return get_amcts_player(evaluator_ptr, n_sims, minimum_duration);
            break;
        case NETWORK_MCTS_PLAYER:
            network_ptr = get_network_ptr(filters, fc_dims, blocks, load_name);
            evaluator_ptr = get_network_evaluator_ptr(network_ptr);
            return get_mcts_player(evaluator_ptr, n_sims, minimum_duration);
            break;

        case CPUCT_RANDOM_ROLLOUT_MCTS:
            evaluator_ptr = get_random_rollout_evaluator_ptr();
            return get_mcts_player(evaluator_ptr, n_sims, minimum_duration);
            break;

        case UCT_PLAYER:
            return get_default_uct_player(n_sims, minimum_duration);
            break;

        case G_PLAYER:
            return get_default_g_player(n_sims, minimum_duration);
            break;
        case MC_RAVE_PLAYER:
            return get_default_mc_rave_player(n_sims, minimum_duration);
            break;
        case HUMAN_PLAYER:
            return get_human_player();
            break;
        case RANDOM_ACTION_PLAYER:
            return std::make_unique<rl::players::RandomActionPlayer>();
            break;
        case NETWORK_EVALUATOR_PLAYER:

            network_ptr = get_network_ptr(filters, fc_dims, blocks, load_name);
            evaluator_ptr = get_network_evaluator_ptr(network_ptr);
            return std::make_unique<rl::players::EvaluatorPlayer>(evaluator_ptr->clone());
            break;
        default:
            throw "";
            break;
        }
    }

    IStatePtr MatchConsole::get_state_ptr()
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
        default:
            throw "";
            break;
        }
    }
    INetworkPtr MatchConsole::get_network_ptr(int filters, int fc_dims, int blocks, std::string load_name)
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

        network_ptr->forward(torch::randn(n_examples * n_channels * n_rows * n_cols).to(device).reshape({n_examples, n_channels, n_rows, n_cols}));
        return network_ptr;
    }

    std::unique_ptr<rl::players::IEvaluator> MatchConsole::get_network_evaluator_ptr(INetworkPtr &network_ptr)
    {
        auto state_ptr = get_state_ptr();
        return std::make_unique<rl::deeplearning::NetworkEvaluator>(network_ptr->copy(), state_ptr->get_n_actions(), state_ptr->get_observation_shape());
    }

    IPlayerPtr MatchConsole::get_amcts_player(std::unique_ptr<rl::players::IEvaluator> &evaluator_ptr, int n_sims, std::chrono::duration<int, std::milli> minimum_duration)
    {
        auto state_ptr = get_state_ptr();
        return std::make_unique<rl::players::AmctsPlayer>(state_ptr->get_n_actions(), evaluator_ptr->copy(), n_sims, minimum_duration, 1.0f, 2.0f, 8);
    }

    IPlayerPtr MatchConsole::get_mcts_player(std::unique_ptr<rl::players::IEvaluator> &evaluator_ptr, int n_sims, std::chrono::duration<int, std::milli> minimum_duration)
    {
        auto state_ptr = get_state_ptr();
        return std::make_unique<rl::players::MctsPlayer>(state_ptr->get_n_actions(), evaluator_ptr->copy(), n_sims, minimum_duration, 1.0f, 2.0f);
    }

    std::unique_ptr<rl::players::IEvaluator> MatchConsole::get_random_rollout_evaluator_ptr()
    {
        auto state_ptr = get_state_ptr();
        return std::make_unique<rl::players::RandomRolloutEvaluator>(state_ptr->get_n_actions());
    }

    IPlayerPtr MatchConsole::get_default_uct_player(int n_sims, std::chrono::duration<int, std::milli> minimum_duration)
    {
        return std::make_unique<rl::players::UctPlayer>(n_sims, minimum_duration, 1.0f, std::sqrtf(2.0f));
    }

    IPlayerPtr MatchConsole::get_default_g_player(int n_sims, std::chrono::duration<int, std::milli> minimum_duration)
    {
        return std::make_unique<rl::players::GPlayer>(n_sims, minimum_duration, 15, 0.04f);
    }

    IPlayerPtr MatchConsole::get_default_mc_rave_player(int n_sims, std::chrono::duration<int, std::milli> minimum_duration)
    {
        return std::make_unique<rl::players::McravePlayer>(n_sims, minimum_duration, 0.01f);
    }

    IPlayerPtr MatchConsole::get_human_player()
    {
        return std::make_unique<rl::players::HumanPlayer>();
    }
    int MatchConsole::pick_player_type()
    {
        std::cout << "Choose Player Type\n";
        std::cout << "[0] Network amcts player\n";
        std::cout << "[1] Network mcts player\n";
        std::cout << "[2] Random rollout mcts player\n";
        std::cout << "[3] Uct player\n";
        std::cout << "[4] G-rave player\n";
        std::cout << "[5] Mc-rave player\n";
        std::cout << "[6] Human player\n";
        std::cout << "[7] Random action player\n";
        std::cout << "[8] Network player without tree\n";
        std::cout << std::endl;

        int choice;
        std::cin >> choice;
        switch (choice)
        {
        case 0:
            return NETWORK_AMCTS_PLAYER;
            break;
        case 1:
            return NETWORK_MCTS_PLAYER;
            break;
        case 2:
            return CPUCT_RANDOM_ROLLOUT_MCTS;
            break;
        case 3:
            return UCT_PLAYER;
            break;
        case 4:
            return G_PLAYER;
            break;
        case 5:
            return MC_RAVE_PLAYER;
            break;
        case 6:
            return HUMAN_PLAYER;
            break;
        case 7:
            return RANDOM_ACTION_PLAYER;
            break;
        case 8:
            return NETWORK_EVALUATOR_PLAYER;
            break;
        default:
            return -1;
            break;
        }
    }
} // rl::run