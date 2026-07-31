#include "mcts_nnue_data_generator.hpp"
#include <iostream>
#include <cmath>
#include <common/random.hpp>

namespace rl::training {

MctsNNUEDataGenerator::MctsNNUEDataGenerator(
    std::unique_ptr<rl::deeplearning::NetworkEvaluator> evaluator_ptr,
    int n_game_actions,
    int n_concurrent_games,
    int n_simulations_per_move,
    int max_async_simulations_per_tree,
    float cpuct,
    float dirichlet_epsilon,
    float dirichlet_alpha,
    float default_visits,
    float default_wins,
    ValueLabelMode value_label_mode)
    : n_game_actions_(n_game_actions),
    n_concurrent_games_(n_concurrent_games),
    n_simulations_per_move_(n_simulations_per_move),
    value_label_mode_(value_label_mode) {

    tree_ptr_ = std::make_unique<rl::players::ConcurrentAmcts>(
        n_game_actions_,
        std::move(evaluator_ptr),
        cpuct,
        1.0f, // raw visit distribution; generate()'s temperature reweights it
        max_async_simulations_per_tree,
        dirichlet_epsilon,
        dirichlet_alpha,
        default_visits,
        default_wins);
}

MctsNNUEDataGenerator::~MctsNNUEDataGenerator() = default;

void MctsNNUEDataGenerator::generate(const rl::common::IState& initial_state,
    int total_samples,
    const std::string& output_path,
    float temperature) {

    std::ofstream outfile(output_path, std::ios::binary | std::ios::out);
    if (!outfile.is_open()) {
        throw std::runtime_error("Could not open output file: " + output_path);
    }

    std::vector<std::unique_ptr<rl::common::IState>> states;
    std::vector<std::vector<float>> episode_obs(n_concurrent_games_);
    std::vector<std::vector<int>> episode_players(n_concurrent_games_);

    for (int i = 0; i < n_concurrent_games_; ++i) {
        states.push_back(start_new_episode(initial_state));
    }

    int collected = 0;
    int next_log_threshold = 5000;

    while (collected < total_samples) {
        std::vector<const rl::common::IState*> ptr_vec;
        ptr_vec.reserve(n_concurrent_games_);
        for (const auto& s : states) {
            ptr_vec.push_back(s.get());
        }

        auto [trees_probs, trees_values] = tree_ptr_->search_multiple(
            ptr_vec, n_simulations_per_move_, std::chrono::milliseconds(0));

        for (int i = 0; i < n_concurrent_games_; ++i) {
            int current_player = states[i]->player_turn();
            auto current_obs = states[i]->get_observation();
            std::vector<float>& state_probs = trees_probs[i];

            std::vector<std::vector<float>> sym_obs;
            std::vector<std::vector<float>> sym_probs; // ignored, only obs orientations are needed for NNUE
            states[i]->get_symmetrical_obs_and_actions(current_obs, state_probs, sym_obs, sym_probs);

            if (value_label_mode_ == ValueLabelMode::kRootValue) {
                // The root's value estimate is already relative to the state's
                // current player (same perspective get_observation() uses), so
                // every orientation can be written out immediately - no need
                // to wait for the episode to finish.
                float score = trees_values[i];
                save_sample_binary(outfile, score, current_obs);
                for (const auto& s_obs : sym_obs) {
                    save_sample_binary(outfile, score, s_obs);
                }
                collected += 1 + static_cast<int>(sym_obs.size());
            } else {
                auto& obs_buffer = episode_obs[i];
                auto& player_buffer = episode_players[i];

                // Save all 8 orientations: the original observation plus its 7 symmetries.
                for (float cell : current_obs) {
                    obs_buffer.push_back(cell);
                }
                player_buffer.push_back(current_player);

                for (const auto& s_obs : sym_obs) {
                    for (float cell : s_obs) {
                        obs_buffer.push_back(cell);
                    }
                    player_buffer.push_back(current_player);
                }
            }

            int action = sample_action(state_probs, temperature);
            states[i] = states[i]->step(action);

            if (states[i]->is_terminal()) {
                if (value_label_mode_ == ValueLabelMode::kFinalOutcome) {
                    float result = states[i]->get_reward();
                    int last_player = states[i]->player_turn();

                    auto& obs_buffer = episode_obs[i];
                    auto& player_buffer = episode_players[i];
                    int obs_size = static_cast<int>(current_obs.size());
                    int n_records = static_cast<int>(player_buffer.size());
                    for (int r = 0; r < n_records; ++r) {
                        int p = player_buffer[r];
                        float score = (p == last_player) ? result : -result;
                        std::vector<float> obs_slice(
                            obs_buffer.begin() + r * obs_size,
                            obs_buffer.begin() + (r + 1) * obs_size);
                        save_sample_binary(outfile, score, obs_slice);
                    }
                    collected += n_records;

                    obs_buffer.clear();
                    player_buffer.clear();
                }
                states[i] = start_new_episode(initial_state);
            }
        }

        if (collected >= next_log_threshold) {
            std::cout << "[MCTS Generator] Collected " << collected << " / " << total_samples << " examples..." << std::endl;
            next_log_threshold += 5000;
        }
    }

    outfile.close();
    std::cout << "[MCTS Generator] Generation complete. File saved to: " << output_path << std::endl;
}

int MctsNNUEDataGenerator::sample_action(const std::vector<float>& probs, float temp) {
    int n_actions = static_cast<int>(probs.size());

    std::vector<float> adjusted_probs;
    adjusted_probs.reserve(n_actions);
    float sum = 0.0f;

    for (int i = 0; i < n_actions; ++i) {
        float p = std::pow(probs[i], 1.0f / temp);
        adjusted_probs.push_back(p);
        sum += p;
    }

    float r = rl::common::get() * sum;
    for (int i = 0; i < n_actions; ++i) {
        if ((r -= adjusted_probs[i]) <= 0) return i;
    }
    return n_actions - 1;
}

void MctsNNUEDataGenerator::save_sample_binary(std::ofstream& out, float score, const std::vector<float>& obs) {
    // 1. Write Score (4 bytes)
    out.write(reinterpret_cast<const char*>(&score), sizeof(float));

    // 2. Filter Active Feature IDs (Sparse Representation)
    std::vector<int16_t> active_ids;
    int obs_size = static_cast<int>(obs.size());
    for (int i = 0; i < obs_size; ++i) {
        if (obs[i] > 0.5f) {
            active_ids.push_back(static_cast<int16_t>(i));
        }
    }

    // 3. Write Count (2 bytes) and Indices (N * 2 bytes)
    int16_t count = static_cast<int16_t>(active_ids.size());
    out.write(reinterpret_cast<const char*>(&count), sizeof(int16_t));
    out.write(reinterpret_cast<const char*>(active_ids.data()), count * sizeof(int16_t));
}

std::unique_ptr<rl::common::IState> MctsNNUEDataGenerator::start_new_episode(
    const rl::common::IState& initial_state) {
    int depth = rl::common::get(0, 5); // uniform random opening length in [0,4]

    while (true) {
        auto state = initial_state.reset();
        bool failed = false;

        for (int s = 0; s < depth; ++s) {
            auto mask = state->actions_mask();

            std::vector<int> legal_actions;
            for (int a = 0; a < static_cast<int>(mask.size()); ++a) {
                if (mask[a]) {
                    legal_actions.push_back(a);
                }
            }

            if (legal_actions.empty()) {
                failed = true;
                break;
            }

            int idx = rl::common::get(0, static_cast<int>(legal_actions.size()));
            int action = legal_actions[idx];

            state = state->step(action);

            if (state->is_terminal()) {
                failed = true;
                break;
            }
        }

        if (!failed) {
            return state;
        }
    }
}

} // namespace rl::training
