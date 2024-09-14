
#include <chrono>
#include <cmath>
#include <filesystem>
#include <tuple>

#include <common/random.hpp>
#include <common/utils.hpp>
#include <players/amcts_player.hpp>
#include <players/amcts.hpp>
#include <players/amcts2_player.hpp>
#include <players/bandits/amcts2/amcts2.hpp>
#include <players/random_rollout_evaluator.hpp>
#include <deeplearning/network_evaluator.hpp>
#include <common/match.hpp>
#include <common/concurrent_match.hpp>
#include <deeplearning/alphazero/alphazero.hpp>


namespace rl::deeplearning::alphazero
{
// These are being used during evaluation, and not during data collection
constexpr int N_ASYNC = 8;
constexpr float N_VISITS = 1.0f;
constexpr float N_WINS = -1.0f;


constexpr float EPS = 1e-4f;

// Number of workers/games that run asynchronously , each has its own tree, used during data collection
constexpr int N_TREES = 64;
// Number of states to be collected per sub tree before evaluation
constexpr int N_SUB_TREE_ASYNC = 1;
// Number of  workers/games that cannot end early and must complete to end ( until terminal ) , the rest can skip when it is losing horribly
constexpr int N_COMPLETE_TO_END = N_TREES / 4;
// Players that reached below this score can resign IF THEY ARE NOT COMPLETE TO END PLAYERS
constexpr float NO_RESIGN_THRESHOLD = -0.8f;
// Players are not allowed to resign if the number of steps is below this number
constexpr int MINIMUM_STEPS = 30;
// MCTS CPUCT
constexpr float CPUCT = 2.5f;
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
    std::unique_ptr<IAlphazeroNetwork> tiny_ptr,
    std::string load_path,
    std::string save_name)

    : initial_state_ptr_{ std::move(initial_state_ptr) },
    test_state_ptr_{ std::move(test_state_ptr) },
    n_iterations_{ n_iterations },
    n_episodes_{ n_episodes },
    n_sims_{ n_sims },
    lr_{ lr },
    critic_ceof_{ critic_coef },
    n_epoches_{ n_epochs },
    n_batches_{ n_batches },
    n_testing_episodes_{ n_testing_episodes },
    n_game_actions_{ 0 },
    base_network_ptr_{ std::move(network_ptr) },
    tiny_network_ptr_{ std::move(tiny_ptr) },
    dev_{ torch::cuda::is_available() ? torch::kCUDA : torch::kCPU },
    load_path_{ load_path },
    save_name_{ save_name }
{
    n_game_actions_ = initial_state_ptr_->get_n_actions();
    base_network_ptr_->to(dev_);
    tiny_network_ptr_->to(dev_);
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
    std::filesystem::path tiny_path;
    std::filesystem::path strongest_path;
    tiny_path = folder / "tiny.pt";
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
    torch::optim::AdamW optimizer(base_network_ptr_->parameters(), torch::optim::AdamWOptions{ lr_ }.eps(1e-8).weight_decay(1e-4));
    auto observation_shape = initial_state_ptr_->get_observation_shape();
    int channels = observation_shape.at(0);
    int rows = observation_shape.at(1);
    int cols = observation_shape.at(2);
    int observation_size = channels * rows * cols;

    int iteration{ 0 };
    while (iteration < n_iterations_)
    {
        auto collection_start = std::chrono::high_resolution_clock::now();

        std::cout << "\nIteration " << iteration + 1 << " of " << n_iterations_ << std::endl;

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

        train_network(base_network_ptr_, optimizer, all_observations_, all_probabilities_, all_wdls_);

        iteration++;

        if (iteration % 20 == 0 /*|| iteration == 1*/ || iteration == n_iterations_)
        {
            auto evaluation_start = std::chrono::high_resolution_clock::now();
            torch::NoGradGuard nograd;
            std::cout << "Evaluation Phase" << std::endl;
            std::chrono::duration<int, std::milli> zero_duration{ 0 };
            std::unique_ptr<rl::players::IEvaluator> ev1_ptr{ std::make_unique<rl::deeplearning::NetworkEvaluator>(base_network_ptr_->copy(), n_game_actions_, observation_shape) };
            // auto p1_ptr = std::make_unique<rl::players::Amcts2Player>(n_game_actions_, std::move(ev1_ptr), n_sims_, zero_duration, 0.5, CPUCT, N_ASYNC, N_VISITS, N_WINS);
            auto p1_ptr = std::make_unique<rl::players::ConcurrentPlayer>(n_game_actions_, std::move(ev1_ptr), n_sims_, zero_duration, 0.5, CPUCT, N_ASYNC, N_VISITS, N_WINS);
            std::unique_ptr<rl::players::IEvaluator> ev2_ptr{ std::make_unique<rl::deeplearning::NetworkEvaluator>(strongest->copy(), n_game_actions_, observation_shape) };
            // auto p2_ptr = std::make_unique<rl::players::Amcts2Player>(n_game_actions_, std::move(ev2_ptr), n_sims_, zero_duration, 0.5, CPUCT, N_ASYNC, N_VISITS, N_WINS);
            auto p2_ptr = std::make_unique<rl::players::ConcurrentPlayer>(n_game_actions_, std::move(ev2_ptr), n_sims_, zero_duration, 0.5, CPUCT, N_ASYNC, N_VISITS, N_WINS);
            // rl::common::Match m(test_state_ptr_->reset(), p1_ptr.get(), p2_ptr.get(), n_testing_episodes_, false);
            rl::common::ConcurrentMatch m(test_state_ptr_->reset(), p1_ptr.get(), p2_ptr.get(), n_testing_episodes_, n_testing_episodes_);
            float p1_score_average = m.start();
            // std::tuple<float, float> tp{ m.start() };
            // float p1_score = std::get<0>(tp);
            // float ratio = float(p1_score + n_testing_episodes_) / (2 * n_testing_episodes_);
            float ratio = static_cast<float>(p1_score_average + 1.0f) / (2.0f);
            auto duration = std::chrono::high_resolution_clock::now() - evaluation_start;
            auto duration_in_seconds = duration / std::chrono::seconds(1);
            std::cout << "Evaluation Phase Ended : Win ratio is " << ratio << " took is " << duration_in_seconds << " s" << std::endl;
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
        tiny_network_ptr_->save(tiny_path.string());
    }
}

int AlphaZero::choose_action(std::vector<float>& probs)
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

void AlphaZero::train_network(std::unique_ptr<IAlphazeroNetwork>& network_ptr, torch::optim::Optimizer& optimizer_ref, std::vector<float>& observations, std::vector<float>& probabilities, std::vector<float>& wdls)
{
    std::array<int, 3> observation_shape = initial_state_ptr_->get_observation_shape();
    int channels = observation_shape.at(0);
    int rows = observation_shape.at(1);
    int cols = observation_shape.at(2);
    int n_examples = observations.size() / (channels * rows * cols);
    int batch_size = n_examples / n_batches_;
    // int batch_size = 1024;
    // n_batches_ = n_examples / batch_size;

    std::vector<float> total_losses{};
    std::vector<float> actor_losses{};
    std::vector<float> critic_losses{};

    torch::Tensor observations_tensor = torch::tensor(observations, torch::kFloat32).reshape({ n_examples, channels, rows, cols });
    torch::Tensor probabilities_tensor = torch::tensor(probabilities, torch::kFloat32).reshape({ n_examples, n_game_actions_ });
    torch::Tensor wdl_tensor = torch::tensor(wdls, torch::kFloat32).reshape({ n_examples, 3 });

    // Avoid extreme prob values like 1 or 0
    probabilities_tensor = probabilities_tensor + EPS;
    probabilities_tensor = probabilities_tensor / probabilities_tensor.sum(-1, true);
    wdl_tensor = wdl_tensor + EPS;
    wdl_tensor = wdl_tensor / wdl_tensor.sum(-1, true);

    for (int epoch{ 0 }; epoch < n_epoches_; epoch++)
    {
        torch::Tensor indices = torch::randperm(n_examples);
        for (int batch{ 0 }; batch < n_batches_; batch++)
        {
            int batch_start = batch * batch_size;
            int batch_end = batch_start + batch_size;
            torch::Tensor sample_ids = indices.index({ torch::indexing::Slice(batch_start, batch_end) });
            torch::Tensor observations_batch = observations_tensor.index({ sample_ids }).to(dev_);
            torch::Tensor target_probs_batch = probabilities_tensor.index({ sample_ids }).to(dev_);
            torch::Tensor target_wdl_batch = wdl_tensor.index({ sample_ids }).to(dev_);

            std::tuple predicted = network_ptr->forward(observations_batch);
            torch::Tensor predicted_probs = std::get<0>(predicted);
            torch::Tensor predicted_wdls = std::get<1>(predicted);

            // To remove nan if it exist....
            torch::Tensor filter_0 = torch::any(torch::isnan(predicted_probs), -1) == 0;
            torch::Tensor filter_1 = torch::any(torch::isnan(target_probs_batch), -1) == 0;
            torch::Tensor filter_2 = torch::any(torch::isnan(predicted_wdls), -1) == 0;
            torch::Tensor filter_3 = torch::any(torch::isnan(target_wdl_batch), -1) == 0;

            torch::Tensor filter = filter_0 * filter_1 * filter_2 * filter_3;

            predicted_probs = predicted_probs.index({ filter });
            target_probs_batch = target_probs_batch.index({ filter });
            predicted_wdls = predicted_wdls.index({ filter });
            target_wdl_batch = target_wdl_batch.index({ filter });

            torch::Tensor actor_loss = cross_entropy_loss_(target_probs_batch, predicted_probs);
            torch::Tensor critic_loss = cross_entropy_loss_(target_wdl_batch, predicted_wdls);

            torch::Tensor total_loss = actor_loss + critic_loss * critic_ceof_;
            optimizer_ref.zero_grad();

            total_loss.backward();

            torch::nn::utils::clip_grad_value_(network_ptr->parameters(), 10000);
            torch::nn::utils::clip_grad_norm_(network_ptr->parameters(), 10, 2.0, true);

            optimizer_ref.step();
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

torch::Tensor AlphaZero::cross_entropy_loss_(torch::Tensor& target, torch::Tensor& prediction)
{
    auto log_probs = prediction.log();
    auto loss = -(target * log_probs).sum(-1).mean();
    return loss;
}
void AlphaZero::collect_data()
{
    int max_n_trees = states_ptrs_.size();
    std::cout << "running " << max_n_trees << " parallel trees" << std::endl;
    int completed_episodes = 0;
    int trees_n_simulations = 0;
    auto ev_ptr = std::make_unique<NetworkEvaluator>(base_network_ptr_->copy(), n_game_actions_, initial_state_ptr_->get_observation_shape());

    // TODO a lot of pushbacks , we can reserve some places

    // Collect at least this number of episodes
    while (completed_episodes < n_episodes_)
    {
        /*
        Plan is to traverse multiple games/subtrees , collect states that need to evaluated
        Send them to GPU and evaluate them all at once , then return evaluation info
        to the subtrees , repeat the same process for a number of simulations ,
        then step the root states and saving training examples localy, repeat the steps for each
        sub tree until , when a game reaches a terminal state , properly set rewards for its states
        and send add its training examples globaly , when a certain number of episodes are done exit
        the phase
        */




        // the number of sub trees
        int current_n_trees = static_cast<int>(subtrees_.size());

        for (int i = 0;i < current_n_trees;i++)
        {
            subtrees_.at(i)->set_root(states_ptrs_.at(i).get());
        }
        for (int sim = 0;sim < n_sims_ / N_SUB_TREE_ASYNC;sim++)
        {
            // Used to save rollout states that needs to be evaluated
            std::vector<const rl::common::IState*> rollout_states{};
            // Used to save the tree id that a state in rollout states belongs
            std::vector<int> trees_idx{};
            for (int tree_idx = 0;tree_idx < current_n_trees;tree_idx++)
            {
                constexpr bool USE_DIRICHLET_NOISE{ true };
                auto& subtree = subtrees_.at(tree_idx);
                subtree->clear_rollout();
                for (int async_roll = 0;async_roll < N_SUB_TREE_ASYNC;async_roll++)
                {
                    subtree->roll(USE_DIRICHLET_NOISE);
                }
                auto tree_rollouts = subtree->get_rollouts();

                for (auto state_ptr : tree_rollouts)
                {
                    rollout_states.push_back(state_ptr);
                    trees_idx.push_back(tree_idx);
                }
            }

            std::vector<std::tuple<std::vector<float>, std::vector<float>>> trees_evaluations{};
            for (int tree_idx = 0;tree_idx < current_n_trees;tree_idx++)
            {
                trees_evaluations.push_back(std::make_tuple<std::vector<float>, std::vector<float>>(
                    std::vector<float>(), std::vector<float>()));
            }
            auto& [probs, vs] = ev_ptr->evaluate(rollout_states);
            int n_rollouts = rollout_states.size();
            for (int rollout_id = 0; rollout_id < n_rollouts; rollout_id++)
            {
                int probs_start = rollout_id * n_game_actions_;
                int probs_end = probs_start + n_game_actions_;
                int current_tree_id = trees_idx.at(rollout_id);
                std::vector<float>& current_probs_ref = std::get<0>(trees_evaluations.at(current_tree_id));
                std::vector<float>& current_vs_ref = std::get<1>(trees_evaluations.at(current_tree_id));
                for (int cell = probs_start; cell < probs_end; cell++)
                {
                    current_probs_ref.push_back(probs.at(cell));
                }
                current_vs_ref.push_back(vs.at(rollout_id));
            }

            for (int tree_id = 0; tree_id < current_n_trees; tree_id++)
            {
                auto& current_sub_tree = subtrees_.at(tree_id);
                auto& current_evaluation_tuple = trees_evaluations.at(tree_id);
                current_sub_tree->evaluate_collected_states(current_evaluation_tuple);
            }
        } // end sims

        for (int tree_id = 0;tree_id < current_n_trees;tree_id++)
        {
            auto& subtree = subtrees_.at(tree_id);
            auto& state_ptr = states_ptrs_.at(tree_id);
            int current_player = state_ptr->player_turn();
            episode_players_.at(tree_id).push_back(current_player);
            std::vector<float> state_probs = subtree->get_probs();
            float ev = subtree->get_evaluation();

            auto current_obs = state_ptr->get_observation();
            for (float cell : current_obs)
            {
                episode_obsevations_.at(tree_id).push_back(cell);
            }
            for (float p : state_probs)
            {
                episode_probs_.at(tree_id).push_back(p);
            }
            std::vector<std::vector<float>> obs_syms_vector;
            std::vector<std::vector<float>> probs_sym_vector;
            state_ptr->get_symmetrical_obs_and_actions(current_obs, state_probs, obs_syms_vector, probs_sym_vector);
            if (obs_syms_vector.size() != probs_sym_vector.size())
            {
                throw std::runtime_error("alphazero number of symmetrical observations and symmetrical actions are not equal");
            }

            for (auto& obs_sym : obs_syms_vector)
            {
                for (float cell : obs_sym)
                {
                    episode_obsevations_.at(tree_id).push_back(cell);
                }
                // push current player equal to the number of symmertical observations
                // wdl is counting on it
                episode_players_.at(tree_id).push_back(current_player);
            }

            for (auto& probs_sym : probs_sym_vector)
            {
                for (float p : probs_sym)
                {
                    episode_probs_.at(tree_id).push_back(p);
                }
            }

            bool is_complete_to_end = tree_id < N_COMPLETE_TO_END;
            /*
            should resign if all these condition is met:
            * if minimum number of steps is played and
            * if this sub tree is not forced to complete to end and
            * if the current state evaluation is too low
            */
            if (episode_steps_.at(tree_id) >= MINIMUM_STEPS && is_complete_to_end == false && ev < NO_RESIGN_THRESHOLD)
            {
                int last_player = state_ptr->player_turn();
                float result = -1.0f;
                end_subtree(tree_id, last_player, result);
                completed_episodes++;
            }
            else
            {
                int episode_length = episode_steps_.at(tree_id);
                float temperature = episode_length > 30 ? powf(0.5, (episode_length - 30) / 30) : 1.0f;
                std::vector<float> probs_with_temp = rl::common::utils::apply_temperature(state_probs, temperature);
                int action = choose_action(probs_with_temp);
                state_ptr = state_ptr->step(action);
                episode_steps_.at(tree_id)++;
                if (state_ptr->is_terminal())
                {
                    int last_player = state_ptr->player_turn();
                    float result = state_ptr->get_reward();
                    end_subtree(tree_id, last_player, result);
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

    // reset everything belongs to this state
    episode_obsevations_.at(i).clear();

    episode_players_.at(i).clear();

    episode_probs_.at(i).clear();

    episode_wdls_.at(i).clear();

    episode_steps_.at(i) = 0;

    subtrees_.at(i) = get_new_subtree_ptr();
    states_ptrs_.at(i) = initial_state_ptr_->reset();
}
std::unique_ptr<players::Amcts2> AlphaZero::get_new_subtree_ptr()
{
    return std::make_unique<players::Amcts2>(n_game_actions_, nullptr, CPUCT, 1.0f, N_SUB_TREE_ASYNC, N_VISITS, N_WINS);
}
} // namespace rl::DeepLearning::alphazero
