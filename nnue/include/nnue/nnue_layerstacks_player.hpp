#ifndef RL_NNUE_NNUE_LAYERSTACKS_PLAYER_HPP_
#define RL_NNUE_NNUE_LAYERSTACKS_PLAYER_HPP_

#include <common/player.hpp>
#include <games/migoyugo_light.hpp> // Assuming your light state is here
#include <memory>
#include "nnue_layerstacks_model.hpp"
#include <immintrin.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <cmath>
#include <common/random.hpp>

namespace rl::players {

// Same alpha-beta + incremental dual-accumulator design as NNUEPlayer, but
// the small 256->16->32->1 head is replaced by NNUELayerStacksModel::NUM_BUCKETS
// per-bucket experts, selected by an estimated turn count (migo_count +
// 4*yugo_count) tracked incrementally alongside the accumulators.
class NNUELayerStacksPlayer : public rl::common::IPlayer {

public:
    NNUELayerStacksPlayer(NNUELayerStacksModel model, std::chrono::duration<int, std::milli> max_duration)
        : model_(model), max_duration_{ max_duration } {
    }

    int choose_action(const std::unique_ptr<rl::common::IState>& state_ptr) override {
        auto start_time = std::chrono::high_resolution_clock::now();
        auto light_state = rl::games::MigoyugoLightState::from_short(state_ptr->to_short());

        // Initialize Accumulators (Only once per move)
        std::array<int16_t, 256> acc_w, acc_b;
        rl::games::NNUEUpdate initial_features;
        light_state->get_active_features(initial_features);
        for (int i = 0; i < 256; ++i) acc_w[i] = acc_b[i] = model_.l1_bias[i];

        int migo_count = 0;
        int yugo_count = 0;
        apply_update(acc_w, acc_b, migo_count, yugo_count, initial_features);

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
                apply_update(acc_w, acc_b, migo_count, yugo_count, move_update);

                float score = -alphabeta(*next_state, acc_w, acc_b, migo_count, yugo_count, d - 1, -beta, -alpha, start_time);

                reverse_update(acc_w, acc_b, migo_count, yugo_count, move_update);

                if (time_up_) break;

                if (score > best_score) {
                    best_score = score;
                    best_move_at_this_depth = action;
                }
                alpha = std::max(alpha, best_score);
            }

            for (int action = 0; action < moves.size(); ++action) {
                if (!moves[action] || used[action]) continue;

                rl::games::NNUEUpdate move_update;
                auto next_state = light_state->step_state_light(action, move_update);
                apply_update(acc_w, acc_b, migo_count, yugo_count, move_update);

                float score = -alphabeta(*next_state, acc_w, acc_b, migo_count, yugo_count, d - 1, -beta, -alpha, start_time);

                reverse_update(acc_w, acc_b, migo_count, yugo_count, move_update);

                if (time_up_) break;

                if (score > best_score) {
                    best_score = score;
                    best_move_at_this_depth = action;
                }
                alpha = std::max(alpha, best_score);
            }

            if (time_up_) break;

            best_move_global = best_move_at_this_depth;
            best_score_global = best_score;

            auto now = std::chrono::high_resolution_clock::now();
            if ((now - start_time) * 2 > max_duration_) break;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration_seconds = end_time - start_time;
        double nps = (duration_seconds.count() > 0) ? (nodes_visited_ / duration_seconds.count()) : 0;

        std::cout << "NNUE-LayerStacks Score: " << best_score_global
            << "\tNPS: " << static_cast<uint64_t>(nps)
            << "\tMove: " << best_move_global << std::endl;

        return best_move_global;
    }

    // turns ~= migo_count*1 + yugo_count*4 (a yugo consumes 3 migos + 1
    // placement), clamped to 80, split into NUM_BUCKETS buckets of 10.
    // Must be kept in sync with compute_bucket_index in
    // scripts/train_nnue_layerstacks.py.
    static int compute_bucket_index(int migo_count, int yugo_count) {
        int turns = migo_count + 4 * yugo_count;
        turns = std::min(turns, 80);
        return std::min(turns / 10, NNUELayerStacksModel::NUM_BUCKETS - 1);
    }

private:
    const NNUELayerStacksModel model_;
    int nodes_visited_{ 0 };
    std::chrono::duration<int, std::milli> max_duration_;
    bool time_up_{ false };

    static float evaluate_nnue_simd(const std::array<int16_t, 256>& accumulator, const NNUELayerStacksModel& model, int bucket) {
        // 1. Pre-calculate Activated Layer 1 (Clipped ReLU)
        alignas(32) std::array<int16_t, 256> activated_l1;
        for (size_t i = 0; i < 256; ++i) {
            activated_l1[i] = static_cast<int16_t>(std::clamp<int32_t>(accumulator[i], 0, 127));
        }

        // --- Layer 2: 256 -> 16 (bucketed) ---
        alignas(32) std::array<int32_t, 16> l2_out;
        for (size_t i = 0; i < 16; ++i) {
            __m128i sum_v = _mm_setzero_si128();

            for (size_t j = 0; j < 256; j += 8) {
                __m128i weights = _mm_load_si128((__m128i*) & model.l2_weights[bucket][i][j]);
                __m128i inputs = _mm_load_si128((__m128i*) & activated_l1[j]);
                __m128i madd = _mm_madd_epi16(weights, inputs);
                sum_v = _mm_add_epi32(sum_v, madd);
            }

            alignas(16) int32_t temp_sums[4];
            _mm_store_si128((__m128i*)temp_sums, sum_v);
            int32_t total_sum = model.l2_bias[bucket][i] + temp_sums[0] + temp_sums[1] + temp_sums[2] + temp_sums[3];

            l2_out[i] = std::clamp(total_sum >> 7, 0, 127);
        }

        // --- Layer 3: 16 -> 32 (bucketed) ---
        alignas(32) std::array<int32_t, 32> l3_out;
        for (size_t i = 0; i < 32; ++i) {
            int32_t total_sum = model.l3_bias[bucket][i];
            for (size_t j = 0; j < 16; ++j) {
                total_sum += model.l3_weights[bucket][i][j] * l2_out[j];
            }
            l3_out[i] = std::clamp(total_sum >> 7, 0, 127);
        }

        // --- Output Layer: 32 -> 1 (bucketed) ---
        int32_t final_sum = model.out_bias[bucket];
        for (size_t i = 0; i < 32; ++i) {
            final_sum += model.out_weights[bucket][i] * l3_out[i];
        }

        return static_cast<float>(final_sum) / (128.0f * 128.0f);
    }

    float alphabeta(rl::games::MigoyugoLightState& state,
        std::array<int16_t, 256>& acc_w,
        std::array<int16_t, 256>& acc_b,
        int& migo_count, int& yugo_count,
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

        if (state.is_terminal()) {
            float reward = state.get_reward();
            return (reward * 10000.0f) + (reward * depth);
        }

        if (depth <= 0) {
            auto& current_acc = (state.player_turn() == 0) ? acc_w : acc_b;
            int bucket = compute_bucket_index(migo_count, yugo_count);
            return evaluate_nnue_simd(current_acc, model_, bucket);
        }

        auto moves = state.actions_mask_2();
        auto priority = state.detect_threats();
        std::array<int, 64> used{};
        float value = -1e9;

        for (int action : priority) {
            if (!moves[action]) continue;

            used[action] = true;

            rl::games::NNUEUpdate move_update;
            auto next_state = state.step_state_light(action, move_update);

            apply_update(acc_w, acc_b, migo_count, yugo_count, move_update);
            value = std::max(value, -alphabeta(*next_state, acc_w, acc_b, migo_count, yugo_count, depth - 1, -beta, -alpha, start_time));
            reverse_update(acc_w, acc_b, migo_count, yugo_count, move_update);

            if (time_up_) return 0;

            alpha = std::max(alpha, value);
            if (alpha >= beta) return value;
        }
        for (int action = 0; action < moves.size(); ++action) {
            if (!moves[action] || used[action]) continue;

            rl::games::NNUEUpdate move_update;
            auto next_state = state.step_state_light(action, move_update);

            apply_update(acc_w, acc_b, migo_count, yugo_count, move_update);
            value = std::max(value, -alphabeta(*next_state, acc_w, acc_b, migo_count, yugo_count, depth - 1, -beta, -alpha, start_time));
            reverse_update(acc_w, acc_b, migo_count, yugo_count, move_update);

            if (time_up_) return 0;

            alpha = std::max(alpha, value);
            if (alpha >= beta) break;
        }
        return value;
    }

    // channel = feature_id / 64: {0,2} = migo (OUR_MIGO/OPPONENT_MIGO),
    // {1,3} = yugo (OUR_YUGO/OPPONENT_YUGO) - same layout MigoyugoLightState
    // uses for NNUEUpdate ids and MigoyugoState::get_observation() uses for
    // the training data.
    static bool is_migo_channel(int feature_id) {
        int channel = feature_id / 64;
        return channel == 0 || channel == 2;
    }

    // Helper to push NNUE changes. Counts are derived only from the
    // white_added/white_removed lists - both perspectives describe the same
    // physical pieces (just with flipped channel bits), so counting twice
    // would double-count.
    void apply_update(std::array<int16_t, 256>& aw, std::array<int16_t, 256>& ab,
        int& migo_count, int& yugo_count, const rl::games::NNUEUpdate& u) {
        for (int f : u.white_added) {
            for (int j = 0; j < 256; ++j) aw[j] += model_.l1_weights[j][f];
            if (is_migo_channel(f)) ++migo_count; else ++yugo_count;
        }
        for (int f : u.white_removed) {
            for (int j = 0; j < 256; ++j) aw[j] -= model_.l1_weights[j][f];
            if (is_migo_channel(f)) --migo_count; else --yugo_count;
        }
        for (int f : u.black_added)   for (int j = 0; j < 256; ++j) ab[j] += model_.l1_weights[j][f];
        for (int f : u.black_removed) for (int j = 0; j < 256; ++j) ab[j] -= model_.l1_weights[j][f];
    }

    // Helper to pop NNUE changes (Undo)
    void reverse_update(std::array<int16_t, 256>& aw, std::array<int16_t, 256>& ab,
        int& migo_count, int& yugo_count, const rl::games::NNUEUpdate& u) {
        for (int f : u.white_added) {
            for (int j = 0; j < 256; ++j) aw[j] -= model_.l1_weights[j][f];
            if (is_migo_channel(f)) --migo_count; else --yugo_count;
        }
        for (int f : u.white_removed) {
            for (int j = 0; j < 256; ++j) aw[j] += model_.l1_weights[j][f];
            if (is_migo_channel(f)) ++migo_count; else ++yugo_count;
        }
        for (int f : u.black_added)   for (int j = 0; j < 256; ++j) ab[j] -= model_.l1_weights[j][f];
        for (int f : u.black_removed) for (int j = 0; j < 256; ++j) ab[j] += model_.l1_weights[j][f];
    }
};

} // namespace rl::players

#endif
