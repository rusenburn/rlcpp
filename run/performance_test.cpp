#include "performance_test.hpp"
#include <games/games.hpp>
#include <common/exceptions.hpp>
#include <common/random.hpp>

namespace rl::run
{


void PerformanceTest::run()
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
            edit_game_settings();
        }
    } while (choice != 0);


    start();

}

PerformanceTest::~PerformanceTest() = default;



IStatePtr PerformanceTest::get_state_ptr()
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
    case GOBBLET_GAME:
        return rl::games::GobbletGoblersState::initialize();
        break;
    case MIGOYUGO_GAME:
        return rl::games::MigoyugoState::initialize();
        break;
    default:
        throw "";
        break;
    }
}

void PerformanceTest::edit_game_settings()
{
    int choice = 1;

    std::cout << "Choose game\n";
    std::cout << "[" << TIC_TAC_TOE_GAME << "] Tic Tac Toe\n";
    std::cout << "[" << OTHELLO_GAME << "] Othello\n";
    std::cout << "[" << ENGLISH_DRAUGHTS_GAME << "] English  Draughts\n";
    std::cout << "[" << WALLS_GAME << "] Walls\n";
    std::cout << "[" << DAMMA_GAME << "] DAMMA\n";
    std::cout << "[" << SANTORINI_GAME << "] Santorini\n";
    std::cout << "[" << GOBBLET_GAME << "] Gobblet goblers\n";
    std::cout << "[" << MIGOYUGO_GAME << "] Migoyugo\n";

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
    case GOBBLET_GAME:
        state_index_ = GOBBLET_GAME;
        break;
    case MIGOYUGO_GAME:
        state_index_ = MIGOYUGO_GAME;
        break;
    default:
        break;
    }
}

void PerformanceTest::print_current_settings()
{
    std::cout << "**** Performance Test Settings ****\n";
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
    case GOBBLET_GAME:
        std::cout << "Gobllet Goblers\n";
        break;
    case MIGOYUGO_GAME:
        std::cout << "Migoyugo\n";
        break;
    default:
        std::cout << "Default\n";
        break;
    }
}

// void PerformanceTest::start()
// {
//     auto state_ptr = get_state_ptr();
//     int n_game_actions = state_ptr->get_n_actions();
//     if (state_ptr->is_terminal())
//     {
//         rl::common::SteppingTerminalStateException("Evaluator trying to step a terminal state");
//     }

//     long long steps = 0;
//     int starting_player = state_ptr->player_turn();
//     std::vector<bool> masks = state_ptr->actions_mask();
//     int duration_s = 60;
//     auto min_duration = std::chrono::milliseconds(duration_s * 1000);
//     auto t_start = std::chrono::high_resolution_clock::now();
//     auto t_end = t_start + min_duration;
//     while (std::chrono::high_resolution_clock::now() < t_end)
//     {
//         masks = state_ptr->actions_mask();
//         int action = choose_action(masks);
//         state_ptr = state_ptr->step(action);
//         steps++;
//         if (state_ptr->is_terminal()) {
//             state_ptr = get_state_ptr();
//         }
//     }

//     long long steps_per_second = steps / duration_s;
//     std::cout << "[Test Ended] Environment stepped " << steps_per_second << " steps/second" << std::endl;

// }



void PerformanceTest::start()
{
    auto state_ptr = get_state_ptr();
    int n_game_actions = state_ptr->get_n_actions();
    if (state_ptr->is_terminal())
    {
        rl::common::SteppingTerminalStateException("Evaluator trying to step a terminal state");
    }

    long long steps = 0;
    int starting_player = state_ptr->player_turn();
    std::vector<bool> masks = state_ptr->actions_mask();
    int duration_s = 60;
    auto min_duration = std::chrono::milliseconds(duration_s * 1000);
    auto t_start = std::chrono::high_resolution_clock::now();
    auto t_end = t_start + min_duration;
    while (std::chrono::high_resolution_clock::now() < t_end)
    {
        masks = state_ptr->actions_mask();
        int action = choose_action(masks);
        state_ptr = state_ptr->step(action);
        steps++;
        if (state_ptr->is_terminal()) {
            state_ptr = get_state_ptr();
        }
    }

    long long steps_per_second = steps / duration_s;
    std::cout << "[Test Ended] Environment stepped " << steps_per_second << " steps/second" << std::endl;

}

int PerformanceTest::choose_action(const std::vector<bool>& masks) const
{
    std::vector<int> legal_actions;
    int n_game_actions = masks.size();
    int n_legal_actions = 0;
    for (auto action{ 0 }; action < n_game_actions; action++)
    {
        n_legal_actions += masks.at(action);
        if (masks.at(action))
        {
            legal_actions.push_back(action);
        }
    }

    int action_idx = rl::common::get(n_legal_actions);
    int action = legal_actions[action_idx];
    return action;
}

}