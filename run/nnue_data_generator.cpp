#include "nnue_data_generator.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <common/random.hpp>

namespace rl::training {

NNUEDataGenerator::NNUEDataGenerator(std::shared_ptr<rl::deeplearning::NetworkEvaluator> evaluator,
    int n_concurrent_games)
    : evaluator_ptr_(std::move(evaluator)),
    n_concurrent_games_(n_concurrent_games) {
    // We'll initialize n_actions_ from the first state we see in generate()
}

NNUEDataGenerator::~NNUEDataGenerator() = default;

void NNUEDataGenerator::generate(const rl::common::IState& initial_state,
    int total_samples,
    const std::string& output_path,
    float temperature) {

    std::ofstream outfile(output_path, std::ios::binary | std::ios::out);
    if (!outfile.is_open()) {
        throw std::runtime_error("Could not open output file: " + output_path);
    }

    this->n_actions_ = initial_state.get_n_actions();
    std::vector<std::unique_ptr<rl::common::IState>> states;

    // 1. Initialize Vectorized States
    for (int i = 0; i < n_concurrent_games_; ++i) {
        states.push_back(initial_state.reset());
    }

    int collected = 0;
    while (collected < total_samples) {
        // 2. Prepare Batch for Evaluation
        std::vector<const rl::common::IState*> ptr_vec;
        ptr_vec.reserve(n_concurrent_games_);
        for (const auto& s : states) {
            ptr_vec.push_back(s.get());
        }

        // 3. Vectorized Forward Pass (High Efficiency)
        auto [probs_batch, values_batch] = evaluator_ptr_->evaluate(ptr_vec);

        // 4. Process Batch results
        for (int i = 0; i < n_concurrent_games_; ++i) {
            float teacher_v = values_batch[i];
            auto current_obs = states[i]->get_observation();

            // --- Symmetry Augmentation ---
            std::vector<std::vector<float>> sym_obs;
            std::vector<std::vector<float>> sym_probs; // ignored for NNUE
            std::vector<float> probs_single(
                probs_batch.begin() + i * n_actions_,
                probs_batch.begin() + (i + 1) * n_actions_
            );

            states[i]->get_symmetrical_obs_and_actions(
                current_obs, probs_single, sym_obs, sym_probs
            );

            for (const auto& s_obs : sym_obs) {
                save_sample_binary(outfile, teacher_v, s_obs);
            }

            // --- Step State ---
            int action = sample_action(probs_batch, i, n_actions_, temperature);
            states[i] = states[i]->step(action);

            // --- Terminal Reset ---
            if (states[i]->is_terminal()) {
                states[i] = initial_state.reset();
            }
        }

        collected += n_concurrent_games_;
        if (collected % 5000 == 0) {
            std::cout << "[Generator] Collected " << collected << " / " << total_samples << " base positions..." << std::endl;
        }
    }

    outfile.close();
    std::cout << "[Generator] Generation complete. File saved to: " << output_path << std::endl;
}

int NNUEDataGenerator::sample_action(const std::vector<float>& probs, int game_idx, int n_actions, float temp) {
    // Calculate start of this game's probabilities in the batch vector
    int start_idx = game_idx * n_actions;

    // Apply Temperature for diversity
    std::vector<float> adjusted_probs;
    adjusted_probs.reserve(n_actions);
    float sum = 0.0f;

    for (int i = 0; i < n_actions; ++i) {
        float p = std::pow(probs[start_idx + i], 1.0f / temp);
        adjusted_probs.push_back(p);
        sum += p;
    }

    // Weighted Random Selection
    float r = rl::common::get() * sum;
    for (int i = 0; i < n_actions; ++i) {
        if ((r -= adjusted_probs[i]) <= 0) return i;
    }
    return n_actions - 1;
}

void NNUEDataGenerator::save_sample_binary(std::ofstream& out, float score, const std::vector<float>& obs) {
    // 1. Write Score (4 bytes)
    out.write(reinterpret_cast<const char*>(&score), sizeof(float));

    // 2. Filter Active Feature IDs (Sparse Representation)
    std::vector<int16_t> active_ids;
    for (int i = 0; i < 256; ++i) {
        if (obs[i] > 0.5f) {
            active_ids.push_back(static_cast<int16_t>(i));
        }
    }

    // 3. Write Count (2 bytes) and Indices (N * 2 bytes)
    int16_t count = static_cast<int16_t>(active_ids.size());
    out.write(reinterpret_cast<const char*>(&count), sizeof(int16_t));
    out.write(reinterpret_cast<const char*>(active_ids.data()), count * sizeof(int16_t));
}

} // namespace rl::training