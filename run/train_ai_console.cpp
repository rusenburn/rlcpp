#include <iostream>
#include <deeplearning/alphazero/alphazero.hpp>
#include <deeplearning/alphazero/networks/shared_res_nn.hpp>
#include "train_ai_console.hpp"
#include <games/tictactoe.hpp>
#include <games/othello.hpp>
#include <games/english_draughts.hpp>
#include <games/walls.hpp>

namespace rl::run
{
    TrainAIConsole::TrainAIConsole() = default;
    TrainAIConsole::~TrainAIConsole() = default;
    void TrainAIConsole::run()
    {
        int choice = 0;
        do
        {
            print_current_settings();
            std::cout << "Do you want to run current settings ?\n";
            std::cout << "[0] Yes, run\n";
            std::cout << "[1] No, I want to edit settings\n";
            std::cout << "[2] quit training" << std::endl;

            std::cin >> choice;
            if (choice == 1)
            {
                edit_settings();
            }
            else if (choice == 2)
            {
                return;
            }
        } while (choice);

        train_ai();

        std::cout << "AI Training is done" << std::endl;
    }

    void TrainAIConsole::print_current_settings()
    {
        std::cout << "**** AI Training settings ****\n";
        std::cout << "[game] ";
        switch (state_choice_)
        {
        case TICTACTOE:
            std::cout << "Tic Tac Toe";
            break;
        case OTHELLO:
            std::cout << "Othello";
            break;
        case ENGLISH_DRAUGHTS:
            std::cout << "English Draughts";
            break;
        case WALLS:
            std::cout << "Walls";
            break;
        default:
            std::cout << "Default";
            break;
        }
        std::cout << '\n';

        std::cout << "[Iterations] " << n_iterations_ << " [Episodes per iteration] " << n_episodes_ << '\n';
        std::cout << "[Simulations] " << n_sims_ << " [Epochs] " << n_epochs_ << " [Batches] " << n_batches_ << '\n';
        std::cout << "[Learning rate] " << lr_ << " [Critic Coef] " << critic_coef_ << " [Testing eposodes] " << n_testing_episodes_ << '\n';
        std::cout << "[Load name] " << load_path_ << '\n';
        std::cout << "[Save name] " << save_name_ << '\n';
        std::cout << "[NETWORK] [Filters] " << filters << " Fc Dims " << fc_dimensions << "blocks" << blocks << std::endl;
    }

    void TrainAIConsole::edit_settings()
    {
        int choice = 1;
        while (choice != 0)
        {
            std::cout << "What do you want to edit?\n";
            std::cout << "[0] Nothing\n";
            std::cout << "[1] Game\n";
            std::cout << "[2] Iterations \n";
            std::cout << "[3] Episodes per iteraion \n";
            std::cout << "[4] Simulations \n";
            std::cout << "[5] Epochs \n";
            std::cout << "[6] Batches \n";
            std::cout << "[7] Learning rate \n";
            std::cout << "[8] Critic coef \n";
            std::cout << "[9] Testing episodes \n";
            std::cout << "[10] Load path \n";
            std::cout << "[11] save name \n";
            std::cout << "[12] Network filters \n";
            std::cout << "[13] Network FC DIMS \n";
            std::cout << "[14] Network Blocks \n";
            std::cout << std::endl;

            std::cin >> choice;
            switch (choice)
            {
            case 0:
                break;
            case 1:
                edit_game_settings();
                break;
            case 2:
                std::cout << "[2] Iterations (" << n_iterations_ << ")" << std::endl;
                std::cout << "Enter new value: ";
                std::cin >> n_iterations_;
                break;
            case 3:
                std::cout << "[3] Episodes per turn (" << n_episodes_ << ")" << std::endl;
                std::cout << "Enter new value: ";
                std::cin >> n_episodes_;
                break;
            case 4:
                std::cout << "[4] Simulations (" << n_sims_ << ")" << std::endl;
                std::cout << "Enter new value: ";
                std::cin >> n_sims_;
                break;
            case 5:
                std::cout << "[5] Epochs (" << n_epochs_ << ")" << std::endl;
                std::cout << "Enter new value: ";
                std::cin >> n_epochs_;

                break;
            case 6:
                std::cout << "[6] Batches (" << n_batches_ << ")" << std::endl;
                std::cout << "Enter new value: ";
                std::cin >> n_batches_;
                break;

            case 7:
                std::cout << "[7] Learning rate (" << lr_ << ")" << std::endl;
                std::cout << "Enter new value: ";
                std::cin >> lr_;
                break;

            case 8:
                std::cout << "[8] Critic Coef (" << critic_coef_ << ")" << std::endl;
                std::cout << "Enter new value: ";
                std::cin >> critic_coef_;
                break;

            case 9:
                std::cout << "[9] Testing episodes (" << n_testing_episodes_ << ")" << std::endl;
                std::cout << "Enter new value: ";
                std::cin >> n_testing_episodes_;
                break;

            case 10:
                std::cout << "[10] Load path (" << load_path_ << ")" << std::endl;
                std::cout << "Enter new value: ";
                std::cin >> load_path_;
                break;

            case 11:
                std::cout << "[11] Testing episodes (" << save_name_ << ")" << std::endl;
                std::cout << "Enter new value: ";
                std::cin >> save_name_;
                break;
            case 12:
                std::cout << "[12] Network Filters (" << filters << ")" << std::endl;
                std::cout << "Enter new value: ";
                std::cin >> filters;
                break;

            case 13:
                std::cout << "[13] Network Fc dims (" << fc_dimensions << ")" << std::endl;
                std::cout << "Enter new value: ";
                std::cin >> fc_dimensions;
                break;

            case 14:
                std::cout << "[14] Network Blocks (" << blocks << ")" << std::endl;
                std::cout << "Enter new value: ";
                std::cin >> blocks;
                break;
            default:
                break;

                std::cout << '\n';
            }
        }
    }

    void TrainAIConsole::train_ai()
    {
        IStatePtr state_ptr{get_state_ptr()};
        IAlphazeroNetworkPtr network{get_network_ptr()};
        auto alphazero_trainer = rl::deeplearning::alphazero::AlphaZero(
            state_ptr->clone(),
            state_ptr->clone(),
            n_iterations_,
            n_episodes_,
            n_sims_,
            n_epochs_,
            n_batches_,
            lr_,
            critic_coef_,
            n_testing_episodes_,
            std::move(network),
            load_path_,
            save_name_);
        std::cout << "Starting to train Alphazero bot ..." << std::endl;
        alphazero_trainer.train();
        std::cout << "Training alphazero bot is done." << std::endl;
    }

    IStatePtr TrainAIConsole::get_state_ptr()
    {
        switch (state_choice_)
        {
        case TICTACTOE:
            return rl::games::TicTacToeState::initialize();
            break;
        case OTHELLO:
            return rl::games::OthelloState::initialize();
            break;
        case ENGLISH_DRAUGHTS:
            return rl::games::EnglishDraughtState::initialize();
            break;
        case WALLS:
            return rl::games::WallsState::initialize();
            break;
        default:
            return rl::games::OthelloState::initialize();
            break;
        }
    }

    IAlphazeroNetworkPtr TrainAIConsole::get_network_ptr()
    {
        auto state_ptr = get_state_ptr();
        return std::make_unique<rl::deeplearning::alphazero::SharedResNetwork>(
            state_ptr->get_observation_shape(),
            state_ptr->get_n_actions(),
            filters,
            fc_dimensions,
            blocks);
    }

    void TrainAIConsole::edit_game_settings()
    {
        std::string game_name;
        switch (state_choice_)
        {
        case TICTACTOE:
            game_name = "Tic Tac Toe";
            break;
        case OTHELLO:
            game_name = "Othello";
            break;
        case ENGLISH_DRAUGHTS:
            game_name = "English Draughts";
            break;
        case WALLS:
            game_name = "Walls";
            break;
        default:
            game_name = "Default";
            break;
        }
        std::cout << "Game (" << game_name.c_str() << ")" << std::endl;
        std::cout << "[0] Tic Tac Toe";
        std::cout << "[1] Othello";
        std::cout << "[2] English Draughts";
        std::cout << "[3] Walls";
        std::cout << "Enter new value: ";
        std::cin >> state_choice_;
    }
} // namespace rl::run
