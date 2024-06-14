#include "examples.hpp"
#include <memory>
#include <string>
#include <filesystem>
#include <chrono>
#include <games/othello.hpp>
#include <games/tictactoe.hpp>
#include <games/english_draughts.hpp>
#include <deeplearning/alphazero/alphazero.hpp>
#include <deeplearning/alphazero/networks/smallnn.hpp>
#include <deeplearning/alphazero/networks/shared_res_nn.hpp>
#include <players/evaluator.hpp>
#include <players/random_rollout_evaluator.hpp>
#include <common/player.hpp>
#include <common/match.hpp>
// #include <players/amcts_player.hpp>
// #include <players/mcts_player.hpp>
// #include <players/uct_player.hpp>
// #include <players/mcrave_player.hpp>
// #include <players/grave_player.hpp>
// #include <players/g_player.hpp>
#include <players/players.hpp>
#include <deeplearning/network_evaluator.hpp>
#include <cmath>

void run_match(int n_sims, int duration_in_millis, int n_games)
{
    const std::string folder_name = "../checkpoints";
    // std::string file_name = "temp.pt";
    std::string file_name = "othello_res_20.pt";
    auto file_path = folder_name + "/" + file_name;
    std::filesystem::path folder(folder_name);

    // check if folder exist else create folder
    if (!std::filesystem::is_directory(folder))
    {

        // throw error if you cannot create a folder
        if (!std::filesystem::create_directory(folder))
        {
            throw "Could not create saving directory";
        }
    }

    std::unique_ptr<rl::common::IState> s{rl::games::OthelloState::initialize()};
    // std::unique_ptr<rl::common::IState> s{rl::games::EnglishDraughtState::initialize()};
    int n_game_actions = s->get_n_actions();
    // int n_games = 10;
    // int n_sims = 100;
    std::chrono::duration<int, std::milli> d(duration_in_millis);
    // std::unique_ptr<rl::deeplearning::alphazero::IAlphazeroNetwork> aznet_ptr = std::make_unique<rl::deeplearning::alphazero::SmallAlphaNetwork>(s->get_observation_shape(), s->get_n_actions(),128,512);
    std::unique_ptr<rl::deeplearning::alphazero::IAlphazeroNetwork> aznet_ptr = std::make_unique<rl::deeplearning::alphazero::SharedResNetwork>(s->get_observation_shape(), s->get_n_actions(), 128, 512, 5);
    aznet_ptr->to(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    // aznet_ptr->load(file_path);

    // std::unique_ptr<rl::players::IEvaluator> ev1 = std::make_unique<rl::deeplearning::NetworkEvaluator>(aznet_ptr->copy(), n_game_actions, s->get_observation_shape());
    std::unique_ptr<rl::players::IEvaluator> ev2 = std::make_unique<rl::deeplearning::NetworkEvaluator>(aznet_ptr->copy(), n_game_actions, s->get_observation_shape());
    // std::unique_ptr<rl::common::Player> p1{std::make_unique<rl::players::AmctsPlayer>(n_game_actions, std::move(ev1), n_sims, d, 1.0f, 2.0f, 8)};

    std::unique_ptr<rl::players::IEvaluator> ev1 = std::make_unique<rl::players::RandomRolloutEvaluator>(n_game_actions);
    // std::unique_ptr<rl::players::IEvaluator> ev2 = std::make_unique<rl::players::RandomRolloutEvaluator>(n_game_actions);

    std::unique_ptr<rl::common::IPlayer> p1{std::make_unique<rl::players::MctsPlayer>(n_game_actions, std::move(ev1), n_sims, d, 1.0f, 2.0f)};
    // std::unique_ptr<rl::common::IPlayer> p2{std::make_unique<rl::players::UctPlayer>(n_game_actions, n_sims, d, 1.0f, 2.0f)};
    std::unique_ptr<rl::common::IPlayer> p2{std::make_unique<rl::players::GPlayer>(n_sims, d,15)};
    // std::unique_ptr<rl::common::IPlayer> p2{std::make_unique<rl::players::AmctsPlayer>(n_game_actions, std::move(ev2), n_sims, d, 1.0f, 2.0f,8)};
    // std::unique_ptr<rl::common::Player> p2 = std::make_unique<rl::players::HumanPlayer>() ;
    // std::unique_ptr<rl::common::Player> p2{std::make_unique<rl::players::RandomActionPlayer>()};

    rl::common::Match m(std::move(s), std::move(p1), std::move(p2), n_games, false);
    auto score = m.start();

    float ratio = float(std::get<0>(score) + n_games) / (2 * n_games);
    std::cout << "Player 1 win ratio is " << ratio << std::endl;
}

void run_alphazero()
{
    // auto s_ptr = rl::games::OthelloState::initialize_state();
    auto s_ptr = rl::games::EnglishDraughtState::initialize_state();
    // auto aznet_ptr = std::make_unique<rl::deeplearning::alphazero::SmallAlphaNetwork>(s_ptr->get_observation_shape(),s_ptr->get_n_actions(),128,512);
    auto aznet_ptr = std::make_unique<rl::deeplearning::alphazero::SharedResNetwork>(s_ptr->get_observation_shape(), s_ptr->get_n_actions(), 128, 512);
    rl::deeplearning::alphazero::AlphaZero az{s_ptr->clone(), s_ptr->clone(), 20, 128, 200, 4, 8, 2.5e-4f, 0.5f, 32, std::move(aznet_ptr), "", "othello_res.pt"};
    az.train();
}