#ifndef RL_NNUE_NNUE_PLAYER_HPP_
#define RL_NNUE_NNUE_PLAYER_HPP_

#include <common/player.hpp>
#include <games/migoyugo_light.hpp> // Assuming your light state is here
#include <memory>
#include "nnue_model.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <cmath>
#include <common/random.hpp>
namespace rl::players {

class NNUEPlayer : public rl::common::IPlayer {

public:
    NNUEPlayer(NNUEModel model, std::chrono::duration<int, std::milli> max_duration)
        : model_(model), max_duration_{ max_duration } {
    }

    // int choose_action(const std::unique_ptr<rl::common::IState>& state_ptr) override {
    //     auto start_time = std::chrono::high_resolution_clock::now();
    //     auto light_state = rl::games::MigoyugoLightState::from_short(state_ptr->to_short());

    //     // Initialize Accumulators (Only once per move)
    //     std::array<int16_t, 256> acc_w, acc_b;
    //     rl::games::NNUEUpdate initial_features;
    //     light_state->get_active_features(initial_features);
    //     for (int i = 0; i < 256; ++i) acc_w[i] = acc_b[i] = model_.l1_bias[i];
    //     apply_update(acc_w, acc_b, initial_features);

    //     float features_weight = light_state->calculate_feature_weight();
    //     int best_move_global = -1;
    //     time_up_ = false;
    //     nodes_visited_ = 0;
    //     float best_score_global = -1e9;
    //     auto moves = light_state->actions_mask_2();
    //     auto priority = light_state->detect_threats();
    //     // --- ITERATIVE DEEPENING LOOP ---
    //     for (int d = 1; d <= 100; ++d) {
    //         float alpha = -1e9;
    //         float beta = 1e9;
    //         int best_move_at_this_depth = -1;
    //         float best_score = -1e9;

    //         // Root Search (special case of Alpha-Beta that tracks the move)
    //         for (int action = 0; action < moves.size(); ++action) {
    //             if (!moves[action]) continue;

    //             rl::games::NNUEUpdate move_update;
    //             auto next_state = light_state->step_state_light(action, move_update);
    //             apply_update(acc_w, acc_b, move_update);

    //             // The recursive call
    //             float score = -alphabeta(*next_state, acc_w, acc_b, d - 1, -beta, -alpha, start_time);

    //             reverse_update(acc_w, acc_b, move_update);

    //             if (time_up_) break; // Stop immediately if time ran out

    //             if (score > best_score) {
    //                 best_score = score;
    //                 best_move_at_this_depth = action;
    //             }
    //             alpha = std::max(alpha, best_score);
    //         }

    //         if (time_up_) break; // Don't use the incomplete results from this depth

    //         best_move_global = best_move_at_this_depth;
    //         best_score_global = best_score;

    //         // Check if we already spent > 50% of our time. If so, don't start a new depth.
    //         auto now = std::chrono::high_resolution_clock::now();
    //         if ((now - start_time) * 2 > max_duration_) break;
    //     }

    //     auto end_time = std::chrono::high_resolution_clock::now();
    //     std::chrono::duration<double> duration_seconds = end_time - start_time;
    //     double nps = (duration_seconds.count() > 0) ? (nodes_visited_ / duration_seconds.count()) : 0;

    //     std::cout << "NNUE Score: " << best_score_global
    //         << "\tNPS: " << static_cast<uint64_t>(nps)
    //         << "\tMove: " << best_move_global << std::endl;

    //     return best_move_global;
    // }

    int choose_action(const std::unique_ptr<rl::common::IState>& state_ptr) override {
        auto start_time = std::chrono::high_resolution_clock::now();
        auto light_state = rl::games::MigoyugoLightState::from_short(state_ptr->to_short());

        // Initialize Accumulators (Only once per move)
        std::array<int16_t, 256> acc_w, acc_b;
        rl::games::NNUEUpdate initial_features;
        light_state->get_active_features(initial_features);
        for (int i = 0; i < 256; ++i) acc_w[i] = acc_b[i] = model_.l1_bias[i];
        apply_update(acc_w, acc_b, initial_features);

        float features_weight = light_state->calculate_feature_weight();
        int best_move_global = -1;
        time_up_ = false;
        nodes_visited_ = 0;
        float best_score_global = -1e9;
        auto moves = light_state->actions_mask_2();
        auto priority = light_state->detect_threats();

        // --- ITERATIVE DEEPENING LOOP ---
        for (int d = 1; d <= 100; ++d) {
            float alpha = -1e9;
            float beta = 1e9;
            int best_move_at_this_depth = -1;
            float best_score = -1e9;

            std::array<int, 64> used{};

            for (int action : priority) {
                if (!moves[action]) continue;

                used[action] = 1;

                rl::games::NNUEUpdate move_update;
                auto next_state = light_state->step_state_light(action, move_update);
                apply_update(acc_w, acc_b, move_update);

                float score = -alphabeta(*next_state, acc_w, acc_b, d - 1, -beta, -alpha, start_time);

                reverse_update(acc_w, acc_b, move_update);

                if (time_up_) break;

                if (score > best_score) {
                    best_score = score;
                    best_move_at_this_depth = action;
                }
                alpha = std::max(alpha, best_score);
            }

            // Root Search (special case of Alpha-Beta that tracks the move)
            for (int action = 0; action < moves.size(); ++action) {
                if (!moves[action] || used[action]) continue;

                rl::games::NNUEUpdate move_update;
                auto next_state = light_state->step_state_light(action, move_update);
                apply_update(acc_w, acc_b, move_update);

                // The recursive call
                float score = -alphabeta(*next_state, acc_w, acc_b, d - 1, -beta, -alpha, start_time);

                reverse_update(acc_w, acc_b, move_update);

                if (time_up_) break; // Stop immediately if time ran out

                if (score > best_score) {
                    best_score = score;
                    best_move_at_this_depth = action;
                }
                alpha = std::max(alpha, best_score);
            }

            if (time_up_) break; // Don't use the incomplete results from this depth

            best_move_global = best_move_at_this_depth;
            best_score_global = best_score;

            // Check if we already spent > 50% of our time. If so, don't start a new depth.
            auto now = std::chrono::high_resolution_clock::now();
            if ((now - start_time) * 2 > max_duration_) break;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration_seconds = end_time - start_time;
        double nps = (duration_seconds.count() > 0) ? (nodes_visited_ / duration_seconds.count()) : 0;

        std::cout << "NNUE Score: " << best_score_global
            << "\tNPS: " << static_cast<uint64_t>(nps)
            << "\tMove: " << best_move_global << std::endl;

        return best_move_global;
    }

private:
    const NNUEModel model_;
    int max_depth_;
    int nodes_visited_{ 0 };
    std::chrono::duration<int, std::milli> max_duration_;
    bool time_up_{ false };
    bool is_loaded_{ false };

    static float evaluate_nnue_simd(const std::array<int16_t, 256>& accumulator, const NNUEModel& model) {
        // 1. Pre-calculate Activated Layer 1 (Clipped ReLU)
        // We do this once so we don't have to clamp inside the nested loops.
        alignas(32) std::array<int16_t, 256> activated_l1;
        for (size_t i = 0; i < 256; ++i) {
            activated_l1[i] = static_cast<int16_t>(std::clamp<int32_t>(accumulator[i], 0, 127));
        }

        // --- Layer 2: 256 -> 16 ---
        alignas(32) std::array<int32_t, 16> l2_out;
        for (size_t i = 0; i < 16; ++i) {
            // Initialize sum with bias
            __m128i sum_v = _mm_setzero_si128();

            // Process 256 inputs in chunks of 8 (128-bit SSE)
            for (size_t j = 0; j < 256; j += 8) {
                // Load 8 weights and 8 activated inputs
                __m128i weights = _mm_load_si128((__m128i*) & model.l2_weights[i][j]);
                __m128i inputs = _mm_load_si128((__m128i*) & activated_l1[j]);

                // _mm_madd_epi16: (w0*i0 + w1*i1), (w2*i2 + w3*i3)... 
                // Result is 4 int32s in one register
                __m128i madd = _mm_madd_epi16(weights, inputs);
                sum_v = _mm_add_epi32(sum_v, madd);
            }

            // Horizontal sum of the 4 ints in sum_v
            alignas(16) int32_t temp_sums[4];
            _mm_store_si128((__m128i*)temp_sums, sum_v);
            int32_t total_sum = model.l2_bias[i] + temp_sums[0] + temp_sums[1] + temp_sums[2] + temp_sums[3];

            l2_out[i] = std::clamp(total_sum >> 7, 0, 127);
        }

        // --- Layer 3: 16 -> 32 ---
        // (16 inputs is small, but we can still SSE it)
        alignas(32) std::array<int32_t, 32> l3_out;
        for (size_t i = 0; i < 32; ++i) {
            __m128i sum_v = _mm_setzero_si128();

            // Convert l2_out (int32) to int16 temporarily for madd instruction
            // or just do standard math since 16 iterations is tiny.
            int32_t total_sum = model.l3_bias[i];
            for (size_t j = 0; j < 16; ++j) {
                total_sum += model.l3_weights[i][j] * l2_out[j];
            }
            l3_out[i] = std::clamp(total_sum >> 7, 0, 127);
        }

        // --- Output Layer: 32 -> 1 ---
        int32_t final_sum = model.out_bias;
        for (size_t i = 0; i < 32; i += 4) {
            // Simple 32-bit math for the final layer
            for (size_t k = 0; k < 4; ++k) {
                final_sum += model.out_weights[i + k] * l3_out[i + k];
            }
        }

        return static_cast<float>(final_sum) / (128.0f * 128.0f);
    }

    // float alphabeta(rl::games::MigoyugoLightState& state,
    //     std::array<int16_t, 256>& acc_w,
    //     std::array<int16_t, 256>& acc_b,
    //     int depth, float alpha, float beta,
    //     std::chrono::time_point<std::chrono::high_resolution_clock> start_time)
    // {
    //     // Periodic Time Check (Every 2048 nodes to keep overhead low)
    //     if ((nodes_visited_++ & 2047) == 0) {
    //         if (std::chrono::high_resolution_clock::now() - start_time > max_duration_) {
    //             time_up_ = true;
    //         }
    //     }
    //     if (time_up_) return 0;

    //     // Terminal/Leaf Evaluation
    //     // if (depth == 0 || state.is_terminal()) {
    //     //     auto& current_acc = (state.player_turn() == 0) ? acc_w : acc_b;
    //     //     return evaluate_nnue_simd(current_acc, model_);
    //     // }
    //     if (state.is_terminal()) {
    //         // 1. Get the relative reward (-1.0 = current player lost, 1.0 = current player won)
    //         float reward = state.get_reward();

    //         // 2. DO NOT flip the sign if the reward is already relative.
    //         // Use a large constant to ensure it overrides NNUE.
    //         // Adding (reward * depth) ensures:
    //         // - Wins: +depth (prefers faster win)
    //         // - Losses: -depth (prefers delayed loss/putting up a fight)
    //         return (reward * 10000.0f) + (reward * depth);
    //     }

    //     // 2. If we hit the max depth but the game is still going, use NNUE
    //     if (depth <= 0) {
    //         auto& current_acc = (state.player_turn() == 0) ? acc_w : acc_b;
    //         return evaluate_nnue_simd(current_acc, model_);
    //     }

    //     auto moves = state.actions_mask_2();
    //     float value = -1e9;

    //     for (int action = 0; action < moves.size(); ++action) {
    //         if (!moves[action]) continue;

    //         rl::games::NNUEUpdate move_update;
    //         auto next_state = state.step_state_light(action, move_update);

    //         apply_update(acc_w, acc_b, move_update);
    //         value = std::max(value, -alphabeta(*next_state, acc_w, acc_b, depth - 1, -beta, -alpha, start_time));
    //         reverse_update(acc_w, acc_b, move_update);

    //         if (time_up_) return 0;

    //         alpha = std::max(alpha, value);
    //         if (alpha >= beta) break;
    //     }
    //     return value;
    // }

    float alphabeta(rl::games::MigoyugoLightState& state,
        std::array<int16_t, 256>& acc_w,
        std::array<int16_t, 256>& acc_b,
        int depth, float alpha, float beta,
        std::chrono::time_point<std::chrono::high_resolution_clock> start_time)
    {
        // Periodic Time Check (Every 2048 nodes to keep overhead low)
        if ((nodes_visited_++ & 2047) == 0) {
            if (std::chrono::high_resolution_clock::now() - start_time > max_duration_) {
                time_up_ = true;
            }
        }
        if (time_up_) return 0;

        // Terminal/Leaf Evaluation
        // if (depth == 0 || state.is_terminal()) {
        //     auto& current_acc = (state.player_turn() == 0) ? acc_w : acc_b;
        //     return evaluate_nnue_simd(current_acc, model_);
        // }
        if (state.is_terminal()) {
            // 1. Get the relative reward (-1.0 = current player lost, 1.0 = current player won)
            float reward = state.get_reward();

            // 2. DO NOT flip the sign if the reward is already relative.
            // Use a large constant to ensure it overrides NNUE.
            // Adding (reward * depth) ensures:
            // - Wins: +depth (prefers faster win)
            // - Losses: -depth (prefers delayed loss/putting up a fight)
            return (reward * 10000.0f) + (reward * depth);
        }

        // 2. If we hit the max depth but the game is still going, use NNUE
        if (depth <= 0) {
            auto& current_acc = (state.player_turn() == 0) ? acc_w : acc_b;
            return evaluate_nnue_simd(current_acc, model_);
        }

        auto moves = state.actions_mask_2();
        auto priority = state.detect_threats();
        std::array<int, 64> used{};
        float value = -1e9;

        // 1. Priority moves
        for (int action : priority) {
            if (!moves[action]) continue;

            used[action] = true;

            rl::games::NNUEUpdate move_update;
            auto next_state = state.step_state_light(action, move_update);

            apply_update(acc_w, acc_b, move_update);
            value = std::max(value, -alphabeta(*next_state, acc_w, acc_b, depth - 1, -beta, -alpha, start_time));
            reverse_update(acc_w, acc_b, move_update);

            if (time_up_) return 0;

            alpha = std::max(alpha, value);
            if (alpha >= beta) return value;
        }
        for (int action = 0; action < moves.size(); ++action) {
            if (!moves[action] || used[action]) continue;

            rl::games::NNUEUpdate move_update;
            auto next_state = state.step_state_light(action, move_update);

            apply_update(acc_w, acc_b, move_update);
            value = std::max(value, -alphabeta(*next_state, acc_w, acc_b, depth - 1, -beta, -alpha, start_time));
            reverse_update(acc_w, acc_b, move_update);

            if (time_up_) return 0;

            alpha = std::max(alpha, value);
            if (alpha >= beta) break;
        }
        return value;
    }

    // Helper to push NNUE changes
    void apply_update(std::array<int16_t, 256>& aw, std::array<int16_t, 256>& ab, const rl::games::NNUEUpdate& u) {
        for (int f : u.white_added)   for (int j = 0; j < 256; ++j) aw[j] += model_.l1_weights[j][f];
        for (int f : u.white_removed) for (int j = 0; j < 256; ++j) aw[j] -= model_.l1_weights[j][f];
        for (int f : u.black_added)   for (int j = 0; j < 256; ++j) ab[j] += model_.l1_weights[j][f];
        for (int f : u.black_removed) for (int j = 0; j < 256; ++j) ab[j] -= model_.l1_weights[j][f];
    }

    // Helper to pop NNUE changes (Undo)
    void reverse_update(std::array<int16_t, 256>& aw, std::array<int16_t, 256>& ab, const rl::games::NNUEUpdate& u) {
        for (int f : u.white_added)   for (int j = 0; j < 256; ++j) aw[j] -= model_.l1_weights[j][f];
        for (int f : u.white_removed) for (int j = 0; j < 256; ++j) aw[j] += model_.l1_weights[j][f];
        for (int f : u.black_added)   for (int j = 0; j < 256; ++j) ab[j] -= model_.l1_weights[j][f];
        for (int f : u.black_removed) for (int j = 0; j < 256; ++j) ab[j] += model_.l1_weights[j][f];
    }

    int select_randomized_move(const std::vector<std::pair<int, float>>& move_scores, float feature_count) {
        if (move_scores.empty()) return -1;

        // 1. Calculate Dynamic Temperature
        // Starts high, hits 0 around "10 moves worth of features"
        // Assuming 10 moves * (~2 features/move) = 20 total weight
        float temperature = 0.0f;
        if (feature_count < 20.0f) {
            temperature = 2.0f * (1.0f - (feature_count / 20.0f));
        }

        // If temperature is effectively 0, just take the best move (Greedy)
        if (temperature < 0.05f) {
            return move_scores[0].first;
        }

        // 2. Calculate Softmax Weights
        std::vector<double> weights;
        double sum_weights = 0.0;

        // Move_scores should be sorted by the search already
        float best_score = move_scores[0].second;
        return move_scores[0].first;
        for (const auto& [action, score] : move_scores) {
            // We subtract best_score for numerical stability (prevents overflow)
            double w = std::exp((score - best_score) / temperature);
            weights.push_back(w);
            sum_weights += w;
        }

        // 3. Weighted Random Pick
        double pick = (double)rand() / RAND_MAX * sum_weights;
        double current_sum = 0.0;
        for (size_t i = 0; i < weights.size(); ++i) {
            current_sum += weights[i];
            if (pick <= current_sum) {
                return move_scores[i].first;
            }
        }

        return move_scores[0].first;
    }
};

} // namespace rl::players


#endif