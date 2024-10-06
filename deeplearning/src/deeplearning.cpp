#include <deeplearning/deeplearning.h>
#include <deeplearning/alphazero/alphazero.hpp>
#include <deeplearning/alphazero/networks/shared_res_nn.hpp>
#include <games/games.hpp>
#include <common/state.hpp>
#include <string>
#include <memory>
#include <locale>
#include <filesystem>
#include <sstream>
#include <iostream>

void train_alphazero(AzTrainParameters train_parameters)
{
    std::string env_name(train_parameters.environment_name);
    std::unique_ptr<rl::common::IState> state_ptr{ nullptr };
    std::function<std::unique_ptr<rl::common::IState>()> fn = nullptr;
    for (auto& elem : env_name)
    {
        elem = std::tolower(elem);
    }
    if (env_name == "tictactoe")
    {
        fn = rl::games::TicTacToeState::initialize;
    }
    else if (env_name == "othello" || env_name == "reversi")
    {
        fn = rl::games::OthelloState::initialize;
    }
    else if (env_name == "english_draughts" || env_name == "checkers")
    {
        fn = rl::games::EnglishDraughtState::initialize;
    }
    else if (env_name == "damma")
    {
        fn = rl::games::DammaState::initialize;
    }

    else if (env_name == "walls")
    {
        fn = rl::games::WallsState::initialize;
    }
    else if (env_name == "santorini")
    {
        fn = rl::games::SantoriniState::initialize;
    }

    auto dummy_env_ptr = fn();
    auto network_ptr = std::make_unique<rl::deeplearning::alphazero::SharedResNetwork>(dummy_env_ptr->get_observation_shape(), dummy_env_ptr->get_n_actions());
    std::string loadname{ train_parameters.network_load_name };
    std::string savename{ train_parameters.network_save_name };
    const std::string folder_name = "../checkpoints";
    std::filesystem::path folder(folder_name);
    std::filesystem::path file_path;
    file_path = loadname.size() ? (folder / loadname) : std::filesystem::path();

    rl::deeplearning::AZConfig config{};
    config.n_iterations = train_parameters.n_iterations;
    config.n_episodes = train_parameters.n_episodes;
    config.n_sims = train_parameters.n_sims;
    config.lr = train_parameters.lr;
    config.critic_coef = train_parameters.critic_coef;
    config.n_epochs = train_parameters.n_epochs;
    config.n_batches = train_parameters.n_batches;
    config.n_testing_episodes = train_parameters.n_testing_episodes;
    config.network_load_path.emplace(file_path.string());
    config.network_save_name.emplace(savename);
    config.eval_async_steps = train_parameters.eval_async_steps;
    config.n_visits = train_parameters.n_visits;
    config.n_wins = train_parameters.n_wins;
    config.cpuct = train_parameters.cpuct;
    config.dirichlet_epsilon = train_parameters.dirichlet_epsilon;
    config.dirichlet_alpha = train_parameters.dirichlet_alpha;
    config.n_subtrees = train_parameters.n_subtrees;
    config.n_subtree_async_steps = train_parameters.n_subtree_async_steps;
    config.complete_to_end_ratio = train_parameters.complete_to_end_ratio;
    config.no_resign_threshold = train_parameters.no_resign_threshold;
    config.no_resign_steps = train_parameters.no_resign_steps;
    rl::deeplearning::alphazero::AlphaZero az(
        fn, fn, std::move(network_ptr->deepcopy()), network_ptr->deepcopy(), config);
    std::cout << "To train alphazero type Y or y" << std::endl;
    char inp[200];
    std::cin >> inp;
    auto a = &inp[0];
    std::string inp_string{ a };
    if (inp_string == "Y" || inp_string == "y")
    {
        std::cout << "Training " << train_parameters.environment_name << " On alphazero using " << config.n_sims << " simulations";
        az.train();
    }
}