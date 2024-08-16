#include "players_utils.hpp"
#include <deeplearning/alphazero/networks/shared_res_nn.hpp>
#include <deeplearning/alphazero/networks/tinynn.hpp>
#include <deeplearning/network_evaluator.hpp>
#include <filesystem>
#include <players/random_rollout_evaluator.hpp>
#include <sstream>

namespace rl::ui
{

    PlayerInfoFull::PlayerInfoFull(std::unique_ptr<rl::common::IPlayer> player_ptr, std::string name)
        : player_ptr_(std::move(player_ptr)), name_(name)
    {}

    PlayerInfoFull::~PlayerInfoFull() = default;
    
    std::unique_ptr<PlayerInfoFull> get_default_g_player(rl::common::IState *state_ptr, int n_sims, std::chrono::duration<int, std::milli> minimum_duration)
    {
        return std::make_unique<PlayerInfoFull>(std::move(std::make_unique<rl::players::GPlayer>(n_sims, minimum_duration, 15, 0.04f)), "G_player");
    }

    std::unique_ptr<PlayerInfoFull> get_random_rollout_player_ptr(rl::common::IState *state_ptr, int n_sims, std::chrono::duration<int, std::milli> minimum_duration)
    {
        auto ev_ptr = std::make_unique<rl::players::RandomRolloutEvaluator>(state_ptr->get_n_actions());
        ev_ptr->evaluate(state_ptr);
        return std::make_unique<PlayerInfoFull>(std::move(std::make_unique<rl::players::MctsPlayer>(state_ptr->get_n_actions(), ev_ptr->copy(), n_sims, minimum_duration, 1.0f, 2.0f)), "Random_Rollout");
    }

    std::unique_ptr<PlayerInfoFull> get_network_amcts_player(rl::common::IState *state_ptr, int n_sims, std::chrono::duration<int, std::milli> minimum_duration, std::string load_name)
    {
        auto network_ptr = std::make_unique<rl::deeplearning::alphazero::SharedResNetwork>(state_ptr->get_observation_shape(), state_ptr->get_n_actions(),
                                                                                           128, 512, 5);
        std::stringstream ss;
        auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        const std::string folder_name = "../checkpoints";
        std::filesystem::path folder(folder_name);
        std::filesystem::path file_path;
        file_path = folder / load_name;
        network_ptr->load(file_path.string());
        network_ptr->to(device);
        auto ev_ptr = std::make_unique<rl::deeplearning::NetworkEvaluator>(std::move(network_ptr), state_ptr->get_n_actions(), state_ptr->get_observation_shape());
        ev_ptr->evaluate(state_ptr);
        auto player_ptr = std::make_unique<rl::players::AmctsPlayer>(state_ptr->get_n_actions(), std::move(ev_ptr), n_sims, minimum_duration, 0.5f, 2.0f, 8);
        ss << "AMCTS NN " << load_name;
        return std::make_unique<PlayerInfoFull>(std::move(player_ptr),ss.str());
        
    }

    std::unique_ptr<PlayerInfoFull> get_network_mcts_player(rl::common::IState *state_ptr, int n_sims, std::chrono::duration<int, std::milli> minimum_duration, std::string load_name)
    {
        auto network_ptr = std::make_unique<rl::deeplearning::alphazero::SharedResNetwork>(state_ptr->get_observation_shape(), state_ptr->get_n_actions(),
                                                                                           128, 512, 5);
        auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        // const std::string load_name = "santorini_strongest_120.pt";
        const std::string folder_name = "../checkpoints";
        std::filesystem::path folder(folder_name);
        std::filesystem::path file_path;
        file_path = folder / load_name;
        network_ptr->load(file_path.string());
        network_ptr->to(device);
        auto ev_ptr = std::make_unique<rl::deeplearning::NetworkEvaluator>(std::move(network_ptr), state_ptr->get_n_actions(), state_ptr->get_observation_shape());
        ev_ptr->evaluate(state_ptr);
        auto player_ptr = std::make_unique<rl::players::MctsPlayer>(state_ptr->get_n_actions(), std::move(ev_ptr), n_sims, minimum_duration, 0.5f, 2.0f);
        std::stringstream ss{};
        ss << "MCTS NN " << load_name;
        return std::make_unique<PlayerInfoFull>(std::move(player_ptr),ss.str());
    }

    std::unique_ptr<PlayerInfoFull> get_network_lm_mcts_player(rl::common::IState *state_ptr, int n_sims, std::chrono::duration<int, std::milli> minimum_duration, std::string load_name)
    {
        auto network_ptr = std::make_unique<rl::deeplearning::alphazero::SharedResNetwork>(state_ptr->get_observation_shape(), state_ptr->get_n_actions(),
                                                                                           128, 512, 5);
        auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        const std::string folder_name = "../checkpoints";
        std::filesystem::path folder(folder_name);
        std::filesystem::path file_path;
        file_path = folder / load_name;
        network_ptr->load(file_path.string());
        network_ptr->to(device);
        auto ev_ptr = std::make_unique<rl::deeplearning::NetworkEvaluator>(std::move(network_ptr), state_ptr->get_n_actions(), state_ptr->get_observation_shape());
        ev_ptr->evaluate(state_ptr);
        auto player_ptr = std::make_unique<rl::players::LMMctsPlayer>(state_ptr->get_n_actions(), std::move(ev_ptr), n_sims, minimum_duration, 0.5f, 2.0f);
        std::stringstream ss{};
        ss << "LMCTS NN " << load_name;
        return std::make_unique<PlayerInfoFull>(std::move(player_ptr),ss.str());
        
    }

    std::unique_ptr<PlayerInfoFull> get_network_evaluator_ptr(rl::common::IState *state_ptr, std::string load_name)
    {
        auto network_ptr = std::make_unique<rl::deeplearning::alphazero::SharedResNetwork>(state_ptr->get_observation_shape(), state_ptr->get_n_actions(),
                                                                                           128, 512, 5);
        auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        const std::string folder_name = "../checkpoints";
        std::filesystem::path folder(folder_name);
        std::filesystem::path file_path;
        file_path = folder / load_name;
        network_ptr->load(file_path.string());
        network_ptr->to(device);
        auto ev_ptr = std::make_unique<rl::deeplearning::NetworkEvaluator>(std::move(network_ptr), state_ptr->get_n_actions(), state_ptr->get_observation_shape());
        ev_ptr->evaluate(state_ptr);
        std::stringstream ss{};
        ss << "NN " << load_name;
        auto player_ptr = std::make_unique<rl::players::EvaluatorPlayer>(std::move(ev_ptr));
        return std::make_unique<PlayerInfoFull>(std::move(player_ptr),ss.str());
    }

    std::unique_ptr<PlayerInfoFull> get_tiny_network_mcts_player(rl::common::IState *state_ptr, int n_sims, std::chrono::duration<int, std::milli> minimum_duration, std::string load_name)
    {
        auto network_ptr = std::make_unique<rl::deeplearning::alphazero::TinyNetwork>(state_ptr->get_observation_shape(), state_ptr->get_n_actions());
        auto device = torch::kCPU;
        const std::string folder_name = "../checkpoints";
        std::filesystem::path folder(folder_name);
        std::filesystem::path file_path;
        file_path = folder / load_name;
        network_ptr->load(file_path.string());
        network_ptr->to(device);
        auto ev_ptr = std::make_unique<rl::deeplearning::NetworkEvaluator>(std::move(network_ptr), state_ptr->get_n_actions(), state_ptr->get_observation_shape());
        ev_ptr->evaluate(state_ptr);
        auto player_ptr = std::make_unique<rl::players::MctsPlayer>(state_ptr->get_n_actions(), std::move(ev_ptr), n_sims, minimum_duration, 0.5f, 2.0f);
        std::stringstream ss{};
        ss << "Tiny MCTS NN" << load_name;
        return std::make_unique<PlayerInfoFull>(std::move(player_ptr),ss.str());
    }
} // namespace rl::ui::players_utils
