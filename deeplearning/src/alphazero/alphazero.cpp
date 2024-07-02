
#include <chrono>
#include <cmath>
#include <filesystem>
#include <tuple>

#include <common/random.hpp>
#include <common/utils.hpp>
#include <players/amcts_player.hpp>
#include <players/amcts.hpp>
#include <players/random_rollout_evaluator.hpp>
#include <deeplearning/network_evaluator.hpp>
#include <common/match.hpp>
#include <deeplearning/alphazero/alphazero.hpp>

namespace rl::deeplearning::alphazero
{

    constexpr int N_ASYNC = 8;
    constexpr float N_VISITS = 1.0f;
    constexpr float N_WINS = -1.0f;
    constexpr float EPS = 1e-4f;
    constexpr int N_TREES = 8;
    constexpr int N_COMPLETE_TO_END = N_TREES / 4;
    constexpr float NO_RESIGN_THRESHOLD = -0.8f;
    constexpr int MINIMUM_STEPS = 30;
    AlphaZero::AlphaZero(
        std::unique_ptr<rl::common::IState> initial_state_ptr,
        std::unique_ptr<rl::common::IState> test_state_ptr,
        int n_iterations,
        int n_episodes,
        int n_sims,
        int n_epochs,
        int n_batches,
        float lr,
        float critic_coef,
        int n_testing_episodes,
        std::unique_ptr<IAlphazeroNetwork> network_ptr,
        std::string load_path,
        std::string save_name)

        : initial_state_ptr_{std::move(initial_state_ptr)},
          test_state_ptr_{std::move(test_state_ptr)},
          n_iterations_{n_iterations},
          n_episodes_{n_episodes},
          n_sims_{n_sims},
          lr_{lr},
          critic_ceof_{critic_coef},
          n_epoches_{n_epochs},
          n_batches_{n_batches},
          n_testing_episodes_{n_testing_episodes},
          n_game_actions_{0},
          base_network_ptr_{std::move(network_ptr)},
          dev_{torch::cuda::is_available() ? torch::kCUDA : torch::kCPU},
          load_path_{load_path},
          save_name_{save_name}
    {
        n_game_actions_ = initial_state_ptr_->get_n_actions();
        base_network_ptr_->to(dev_);
        if (load_path.size())
        {
            base_network_ptr_->load(load_path);
        }
        for (int i = 0; i < N_TREES; i++)
        {
            states_ptrs_.push_back(initial_state_ptr_->reset());
            subtrees_.push_back(get_new_subtree_ptr());
            episode_obsevations_.push_back({});
            episode_probs_.push_back({});
            episode_wdls_.push_back({});
            episode_players_.push_back({});
            episode_steps_.push_back(0);
        }
    }

    AlphaZero::~AlphaZero() = default;

    void AlphaZero::train()
    {
        const std::string folder_name = "../checkpoints";
        std::filesystem::path folder(folder_name);
        std::filesystem::path file_path;
        std::filesystem::path strongest_path;
        if (save_name_.size())
        {
            file_path = folder / save_name_;
        }
        else
        {
            file_path = folder / "temp.pt";
        }
        strongest_path = folder / "strongest.pt";

        // check if folder exist else create folder
        if (!std::filesystem::is_directory(folder))
        {

            // throw error if you cannot create a folder
            if (!std::filesystem::create_directory(folder))
            {
                throw "Could not create saving directory";
            }
        }

        auto s = initial_state_ptr_->get_observation_shape();
        auto strongest = base_network_ptr_->deepcopy();
        strongest->to(dev_);
        auto observation_shape = initial_state_ptr_->get_observation_shape();
        int channels = observation_shape.at(0);
        int rows = observation_shape.at(1);
        int cols = observation_shape.at(2);
        int observation_size = channels * rows * cols;

        // // prepare data collection vectors
        // std::vector<float> observations{};
        // std::vector<float> probabilities{};
        // std::vector<float> wdls{}; // wins draw losses

        // std::vector<std::unique_ptr<rl::common::IState>> states_ptrs{};
        // std::vector<std::unique_ptr<AmctsSubTree>> subtrees{};
        // std::vector<std::vector<float>> env_obsevations{};
        // std::vector<std::vector<float>> env_probs{};
        // std::vector<std::vector<float>> env_wdls{};
        // std::vector<std::vector<int>> env_players{};
        // for (int i = 0; i < N_TREES; i++)
        // {
        //     states_ptrs.push_back(initial_state_ptr_->reset());
        //     subtrees.push_back(get_new_subtree_ptr());
        //     env_obsevations.push_back({});
        //     env_probs.push_back({});
        //     env_wdls.push_back({});
        //     env_players.push_back({});
        // }
        int iteration{0};
        while (iteration < n_iterations_)
        {
            auto collection_start = std::chrono::high_resolution_clock::now();

            std::cout << "Iteration " << iteration + 1 << " of " << n_iterations_ << std::endl;

            collect_data();
            auto collection_end = std::chrono::high_resolution_clock::now();
            auto collection_duration = collection_end - collection_start;
            auto collection_duration_in_seconds = (collection_duration) / std::chrono::seconds(1);
            std::cout << "Collection phase ended , took " << collection_duration_in_seconds << " s " << std::endl;

            int n_examples = all_observations_.size() / observation_size;
            if (n_examples * observation_size != all_observations_.size())
            {
                std::runtime_error("observations number is wrong");
            }
            else if (n_examples != all_probabilities_.size() / n_game_actions_)
            {
                std::runtime_error("probabilities examples count are different from observations");
            }
            else if (n_examples != all_wdls_.size() / 3)
            {
                std::runtime_error("wdls examples count are different from observations");
            }

            std::cout << "Training phase using " << n_examples << " examples..." << std::endl;

            train_network(all_observations_, all_probabilities_, all_wdls_);

            iteration++;

            if (iteration % 20 == 0 /*|| iteration == 1*/ || iteration == n_iterations_)
            {
                torch::NoGradGuard nograd;
                std::cout << "Evaluation Phase" << std::endl;
                std::chrono::duration<int, std::milli> zero_duration{0};
                std::unique_ptr<rl::players::IEvaluator> ev1_ptr{std::make_unique<rl::deeplearning::NetworkEvaluator>(base_network_ptr_->copy(), n_game_actions_, observation_shape)};
                auto p1_ptr = std::make_unique<rl::players::AmctsPlayer>(n_game_actions_, std::move(ev1_ptr), n_sims_, zero_duration, 1, 2.0, N_ASYNC, N_VISITS, N_WINS);
                std::unique_ptr<rl::players::IEvaluator> ev2_ptr{std::make_unique<rl::deeplearning::NetworkEvaluator>(strongest->copy(), n_game_actions_, observation_shape)};
                auto p2_ptr = std::make_unique<rl::players::AmctsPlayer>(n_game_actions_, std::move(ev2_ptr), n_sims_, zero_duration, 1, 2.0, N_ASYNC, N_VISITS, N_WINS);
                rl::common::Match m(test_state_ptr_->reset(), std::move(p1_ptr), std::move(p2_ptr), n_testing_episodes_, false);
                std::tuple<float, float> tp{m.start()};
                float p1_score = std::get<0>(tp);
                float ratio = float(p1_score + n_testing_episodes_) / (2 * n_testing_episodes_);
                std::cout << "Win ratio is " << ratio << std::endl;
                if (ratio > 0.5)
                {
                    strongest = base_network_ptr_->deepcopy();
                }
            }

            all_observations_.clear();
            all_probabilities_.clear();
            all_wdls_.clear();
            base_network_ptr_->save(file_path.string());
            strongest->save(strongest_path.string());
        }
    }

    void AlphaZero::execute_episode(std::vector<float> &observations_out, std::vector<float> &probabilities_out, std::vector<float> &wdls_out, bool is_use_network)
    {
        torch::NoGradGuard nograd;
        auto state_ptr = initial_state_ptr_->reset();
        std::vector<int> players;
        std::unique_ptr<rl::players::IEvaluator> ev_ptr{nullptr};
        int max_async = N_ASYNC;
        int n_sims = n_sims_;
        if (is_use_network)
        {
            ev_ptr = std::make_unique<NetworkEvaluator>(base_network_ptr_->copy(), n_game_actions_, state_ptr->get_observation_shape());
        }
        else
        {
            ev_ptr = std::make_unique<rl::players::RandomRolloutEvaluator>(initial_state_ptr_->get_n_actions());
            max_async = 1;
        }
        auto search_tree = rl::players::Amcts(n_game_actions_, std::move(ev_ptr), 2.0f, 1.0f, max_async, N_VISITS, N_WINS);
        std::chrono::duration<int, std::milli> zero_duration{0};
        while (!state_ptr->is_terminal())
        {
            int current_player = state_ptr->player_turn();
            players.push_back(current_player);

            std::vector<float> probs = search_tree.search(state_ptr.get(), n_sims, zero_duration);
            for (float cell : state_ptr->get_observation())
            {
                observations_out.push_back(cell);
            }
            for (float p : probs)
            {
                probabilities_out.push_back(p);
            }
            int action = choose_action(probs);
            state_ptr = state_ptr->step(action);
        }

        int last_player = state_ptr->player_turn();
        float result = state_ptr->get_reward();

        // convert result to win draw loss
        float win = result > 0.001f ? 1.0f : 0.0f;
        float loss = result < -0.001f ? 1.0f : 0.0f;
        float draw = result <= 0.001f && result >= -0.001f ? 1.0f : 0.0f;
        for (auto p : players)
        {
            if (last_player == p)
            {
                wdls_out.push_back(win);
                wdls_out.push_back(draw);
                wdls_out.push_back(loss);
            }
            else
            {
                wdls_out.push_back(loss);
                wdls_out.push_back(draw);
                wdls_out.push_back(win);
            }
        }
    }

    int AlphaZero::choose_action(std::vector<float> &probs)
    {
        float p = rl::common::get();
        float remaining_p = p;
        int action = 0;
        int last_action = n_game_actions_ - 1;
        while ((action < last_action) && ((remaining_p -= probs.at(action)) >= 0))
        {
            action++;
        }

        return action;
    }

    void AlphaZero::train_network(std::vector<float> &observations, std::vector<float> &probabilities, std::vector<float> &wdls)
    {
        torch::optim::AdamW optimizer(base_network_ptr_->parameters(), torch::optim::AdamWOptions{lr_}.betas({0.0f, 0.999f}).eps(1e-8));
        // torch::optim::AdamW optimizer(base_network_ptr_->parameters(), torch::optim::AdamWOptions{lr_}.betas({0.0f, 0.999f}).eps(1e-8).weight_decay(1e-4));
        // torch::optim::SGD optimizer(base_network_ptr_->parameters(), torch::optim::SGDOptions{lr_}.weight_decay(1e-4));
        std::array<int, 3> observation_shape = initial_state_ptr_->get_observation_shape();
        int channels = observation_shape.at(0);
        int rows = observation_shape.at(1);
        int cols = observation_shape.at(2);
        int n_examples = observations.size() / (channels * rows * cols);
        int batch_size = n_examples / n_batches_;

        std::vector<float> total_losses{};
        std::vector<float> actor_losses{};
        std::vector<float> critic_losses{};

        torch::Tensor observations_tensor = torch::tensor(observations, torch::kFloat32).reshape({n_examples, channels, rows, cols});
        torch::Tensor probabilities_tensor = torch::tensor(probabilities, torch::kFloat32).reshape({n_examples, n_game_actions_});
        torch::Tensor wdl_tensor = torch::tensor(wdls, torch::kFloat32).reshape({n_examples, 3});

        // Avoid extreme prob values like 1 or 0
        probabilities_tensor = probabilities_tensor + EPS;
        probabilities_tensor = probabilities_tensor / probabilities_tensor.sum(-1, true);
        wdl_tensor = wdl_tensor + EPS;
        wdl_tensor = wdl_tensor / wdl_tensor.sum(-1, true);

        for (int epoch{0}; epoch < n_epoches_; epoch++)
        {
            torch::Tensor indices = torch::randperm(n_examples);
            for (int batch{0}; batch < n_batches_; batch++)
            {
                int batch_start = batch * batch_size;
                int batch_end = batch_start + batch_size;
                torch::Tensor sample_ids = indices.index({torch::indexing::Slice(batch_start, batch_end)});
                torch::Tensor observations_batch = observations_tensor.index({sample_ids}).to(dev_);
                torch::Tensor target_probs_batch = probabilities_tensor.index({sample_ids}).to(dev_);
                torch::Tensor target_wdl_batch = wdl_tensor.index({sample_ids}).to(dev_);

                std::tuple predicted = base_network_ptr_->forward(observations_batch);
                torch::Tensor predicted_probs = std::get<0>(predicted);
                torch::Tensor predicted_wdls = std::get<1>(predicted);

                // To remove nan if it exist....
                torch::Tensor filter_0 = torch::any(torch::isnan(predicted_probs), -1) == 0;
                torch::Tensor filter_1 = torch::any(torch::isnan(target_probs_batch), -1) == 0;
                torch::Tensor filter_2 = torch::any(torch::isnan(predicted_wdls), -1) == 0;
                torch::Tensor filter_3 = torch::any(torch::isnan(target_wdl_batch), -1) == 0;

                torch::Tensor filter = filter_0 * filter_1 * filter_2 * filter_3;

                predicted_probs = predicted_probs.index({filter});
                target_probs_batch = target_probs_batch.index({filter});
                predicted_wdls = predicted_wdls.index({filter});
                target_wdl_batch = target_wdl_batch.index({filter});

                torch::Tensor actor_loss = cross_entropy_loss_(target_probs_batch, predicted_probs);
                torch::Tensor critic_loss = cross_entropy_loss_(target_wdl_batch, predicted_wdls);

                torch::Tensor total_loss = actor_loss + critic_loss * critic_ceof_;
                optimizer.zero_grad();

                total_loss.backward();

                torch::nn::utils::clip_grad_value_(base_network_ptr_->parameters(), 10000);
                torch::nn::utils::clip_grad_norm_(base_network_ptr_->parameters(), 10, 2.0, true);

                optimizer.step();
                actor_losses.push_back(actor_loss.detach().cpu().item<float>());
                critic_losses.push_back(critic_loss.detach().cpu().item<float>());
                total_losses.push_back(total_loss.detach().cpu().item<float>());
            }
        }

        float actor_sum = 0;
        float critic_sum = 0;
        float total_sum = 0;
        for (int i = 0; i < actor_losses.size(); i++)
        {
            actor_sum += actor_losses.at(i);
            critic_sum += critic_losses.at(i);
            total_sum += total_losses.at(i);
        }
        std::cout << "Actor Loss = " << actor_sum / actor_losses.size() << "\n";
        std::cout << "Critic Loss = " << critic_sum / critic_losses.size() << "\n";
        std::cout << "Total Loss = " << total_sum / total_losses.size() << "\n";
    }

    torch::Tensor AlphaZero::cross_entropy_loss_(torch::Tensor &target, torch::Tensor &prediction)
    {
        auto log_probs = prediction.log();
        auto loss = -(target * log_probs).sum(-1).mean();
        return loss;
    }
    void AlphaZero::collect_data()
    {
        int max_n_trees = states_ptrs_.size();
        std::cout << "running " << max_n_trees << "parallel trees" << std::endl;
        int completed_episodes = 0;
        int trees_n_simulations = 0;
        auto ev_ptr = std::make_unique<NetworkEvaluator>(base_network_ptr_->copy(), n_game_actions_, initial_state_ptr_->get_observation_shape());

        // TODO a lot of pushbacks , we can reserve some places
        while (completed_episodes < n_episodes_)
        {
            std::vector<int> trees_idx{};
            std::vector<const rl::common::IState *> rollout_states{};
            int current_n_trees = static_cast<int>(subtrees_.size());
            for (int i = 0; i < current_n_trees; i++)
            {
                auto &subtree = subtrees_.at(i);
                auto &state = states_ptrs_.at(i);
                for (int j = 0; j < N_ASYNC; j++)
                {
                    subtree->roll(state.get());
                }
                auto tree_rollouts = subtree->get_rollouts();
                int rollouts_size = tree_rollouts.size();
                for (int j = 0; j < rollouts_size; j++)
                {
                    auto state_ptr = tree_rollouts.at(j);
                    rollout_states.push_back(state_ptr);
                    trees_idx.push_back(i);
                }
            }

            std::vector<std::tuple<std::vector<float>, std::vector<float>>> trees_evaluations{};
            for (int i = 0; i < current_n_trees; i++)
            {
                trees_evaluations.push_back(std::make_tuple<std::vector<float>, std::vector<float>>(
                    std::vector<float>(), std::vector<float>()));
            }
            auto [probs, vs] = ev_ptr->evaluate(rollout_states);
            for (int i = 0; i < rollout_states.size(); i++)
            {
                int probs_start = i * n_game_actions_;
                int probs_end = probs_start + n_game_actions_;
                int current_tree_id = trees_idx.at(i);
                std::vector<float> &current_probs_ref = std::get<0>(trees_evaluations.at(current_tree_id));
                std::vector<float> &current_vs_ref = std::get<1>(trees_evaluations.at(current_tree_id));
                for (int j = probs_start; j < probs_end; j++)
                {
                    current_probs_ref.push_back(probs.at(j));
                }
                current_vs_ref.push_back(vs.at(i));
            }
            for (int i = 0; i < current_n_trees; i++)
            {
                auto &current_sub_tree = subtrees_.at(i);
                auto &current_evaluation_tuple = trees_evaluations.at(i);
                current_sub_tree->evaluate_collected_states(current_evaluation_tuple);
            }
            trees_n_simulations += N_ASYNC;

            if (trees_n_simulations < n_sims_)
            {
                continue;
            }
            trees_n_simulations = 0;
            for (int i = 0; i < current_n_trees; i++)
            {
                auto &subtree = subtrees_.at(i);
                auto &state_ptr = states_ptrs_.at(i);
                int current_player = state_ptr->player_turn();
                episode_players_.at(i).push_back(current_player);
                std::vector<float> state_probs = subtree->get_probs(state_ptr.get());
                float ev = subtree->get_evaluation(state_ptr.get());
                for (float cell : state_ptr->get_observation())
                {
                    episode_obsevations_.at(i).push_back(cell);
                }
                for (float p : state_probs)
                {
                    episode_probs_.at(i).push_back(p);
                }

                bool is_complete_to_end = i < N_COMPLETE_TO_END;
                if (episode_steps_.at(i) >= MINIMUM_STEPS && is_complete_to_end == false && ev < NO_RESIGN_THRESHOLD)
                {
                    int last_player = state_ptr->player_turn();
                    float result = -1;
                    end_subtree(i, last_player, result);
                    completed_episodes++;
                }
                else
                {
                    int episode_length = episode_steps_.at(i);
                    float temperature = episode_length > 30 ? powf(0.5, (episode_length - 30) / 30) : 1.0f;
                    std::vector<float> probs_with_temp = rl::common::utils::apply_temperature(state_probs, temperature);
                    int action = choose_action(probs_with_temp);
                    state_ptr = state_ptr->step(action);
                    episode_steps_.at(i)++;
                    if (state_ptr->is_terminal())
                    {
                        int last_player = state_ptr->player_turn();
                        float result = state_ptr->get_reward();
                        end_subtree(i, last_player, result);
                        completed_episodes++;
                    }
                }
            }
        }
    }

    void AlphaZero::end_subtree(int i, int last_player, float result)
    {
        // convert result to win draw loss
        float win = result > 0.001f ? 1.0f : 0.0f;
        float loss = result < -0.001f ? 1.0f : 0.0f;
        float draw = result <= 0.001f && result >= -0.001f ? 1.0f : 0.0f;
        for (auto p : episode_players_.at(i))
        {
            if (last_player == p)
            {
                episode_wdls_.at(i).push_back(win);
                episode_wdls_.at(i).push_back(draw);
                episode_wdls_.at(i).push_back(loss);
            }
            else
            {
                episode_wdls_.at(i).push_back(loss);
                episode_wdls_.at(i).push_back(draw);
                episode_wdls_.at(i).push_back(win);
            }
        }

        for (float cell : episode_obsevations_.at(i))
        {
            all_observations_.push_back(cell);
        }
        for (float p : episode_probs_.at(i))
        {
            all_probabilities_.push_back(p);
        }
        for (float v : episode_wdls_.at(i))
        {
            all_wdls_.push_back(v);
        }

        // reset everything belong to this state
        episode_obsevations_.at(i).clear();
        episode_players_.at(i).clear();
        episode_probs_.at(i).clear();
        episode_wdls_.at(i).clear();
        episode_steps_.at(i) = 0;

        subtrees_.at(i) = get_new_subtree_ptr();
        states_ptrs_.at(i) = initial_state_ptr_->reset();
    }
    std::unique_ptr<AmctsSubTree> AlphaZero::get_new_subtree_ptr()
    {
        return std::make_unique<AmctsSubTree>(n_game_actions_, 2.0f, 1.0f, N_VISITS, N_WINS);
    }
} // namespace rl::DeepLearning::alphazero
