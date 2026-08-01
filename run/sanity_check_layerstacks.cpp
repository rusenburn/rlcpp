#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <filesystem>
#include <games/migoyugo.hpp>
#include <nnue/nnue_layerstacks_model.hpp>
#include <nnue/nnue_layerstacks_player.hpp> // for compute_bucket_index
#include <deeplearning/network_evaluator.hpp>
#include <deeplearning/alphazero/networks/shared_res_nn.hpp>
#include <algorithm>
#include <cstdint>
#include <array>
#include <common/random.hpp>

// Compares the big residual network (migoyugo_strongest_900.pt) against the
// quantized layer-stacked NNUE (nnue_layerstacks_weights.bin) move-by-move
// over a self-played game, the same way run/sanity_check.cpp's
// run_sanity_check_simd compares the big net against the single-net NNUE -
// no traced/torch NNUE reference involved, straight big-net-vs-quantized-NNUE.

int choose_action(const std::vector<float>& probs, int n_game_actions) {
    float p = rl::common::get();
    float remaining_prob = p;
    int action = 0;
    int last_action = n_game_actions - 1;
    while ((action < last_action) && ((remaining_prob -= probs.at(action)) >= 0)) {
        action++;
    }
    return action;
}

// Standalone eval function mirroring run/sanity_check.cpp's own local copy of
// evaluate_nnue_simd, rather than reaching into NNUELayerStacksPlayer's
// private implementation - same convention that file already established.
float evaluate_nnue_layerstacks_simd(const std::array<int16_t, 256>& accumulator,
    const NNUELayerStacksModel& model, int bucket) {
    alignas(32) std::array<int16_t, 256> activated_l1;
    for (size_t i = 0; i < 256; ++i) {
        activated_l1[i] = static_cast<int16_t>(std::clamp<int32_t>(accumulator[i], 0, 127));
    }

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

    alignas(32) std::array<int32_t, 32> l3_out;
    for (size_t i = 0; i < 32; ++i) {
        int32_t total_sum = model.l3_bias[bucket][i];
        for (size_t j = 0; j < 16; ++j) {
            total_sum += model.l3_weights[bucket][i][j] * l2_out[j];
        }
        l3_out[i] = std::clamp(total_sum >> 7, 0, 127);
    }

    int32_t final_sum = model.out_bias[bucket];
    for (size_t i = 0; i < 32; ++i) {
        final_sum += model.out_weights[bucket][i] * l3_out[i];
    }

    return static_cast<float>(final_sum) / (128.0f * 128.0f);
}

void run_sanity_check_layerstacks(const std::string& torch_model_path, const std::string& nnue_bin_path) {
    auto state = rl::games::MigoyugoState::initialize();

    // 1. Load the big residual network (the ground truth being distilled)
    auto network_ptr = std::make_unique<rl::deeplearning::alphazero::SharedResNetwork>(
        state->get_observation_shape(), state->get_n_actions(), 128, 512, 5, true);

    auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    network_ptr->load(torch_model_path);
    network_ptr->to(device);

    auto ev_ptr = std::make_unique<rl::deeplearning::NetworkEvaluator>(
        std::move(network_ptr), state->get_n_actions(), state->get_observation_shape());

    // 2. Load the quantized layer-stacked NNUE
    NNUELayerStacksModel nnue_model;
    FILE* f = fopen(nnue_bin_path.c_str(), "rb");
    if (!f) {
        std::cerr << "Could not open " << nnue_bin_path << std::endl;
        return;
    }
    size_t read_count = fread(&nnue_model, sizeof(NNUELayerStacksModel), 1, f);
    fclose(f);
    if (read_count != 1) {
        std::cerr << "Failed to read a full NNUELayerStacksModel from " << nnue_bin_path << std::endl;
        return;
    }

    std::cout << std::left << std::setw(10) << "Move"
        << std::setw(15) << "BigNet Eval"
        << std::setw(15) << "NNUE Eval"
        << std::setw(10) << "Bucket"
        << "Difference" << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    for (int i = 0; i < 100 && !state->is_terminal(); ++i) {
        auto [probs, output] = ev_ptr->evaluate(state);
        float torch_val = output[0];

        // Build the NNUE accumulator + migo/yugo counts from scratch for
        // this position (not incremental - this check is purely about how
        // closely the quantized layer-stacks net tracks the big net).
        std::array<int16_t, 256> accumulator{};
        for (int j = 0; j < 256; ++j) {
            accumulator[j] = nnue_model.l1_bias[j];
        }

        auto observation = state->get_observation();
        int migo_count = 0;
        int yugo_count = 0;
        for (size_t idx = 0; idx < observation.size(); ++idx) {
            if (observation[idx] > 0.5f) {
                int channel = static_cast<int>(idx) / 64;
                if (channel == 0 || channel == 2) migo_count++; else yugo_count++;
                for (int j = 0; j < 256; ++j) {
                    accumulator[j] += nnue_model.l1_weights[j][idx];
                }
            }
        }

        int bucket = rl::players::NNUELayerStacksPlayer::compute_bucket_index(migo_count, yugo_count);
        float nnue_val = evaluate_nnue_layerstacks_simd(accumulator, nnue_model, bucket);

        float diff = std::abs(torch_val - nnue_val);
        std::cout << std::left << std::setw(10) << i
            << std::setw(15) << torch_val
            << std::setw(15) << nnue_val
            << std::setw(10) << bucket
            << (diff > 0.3 ? "warn " : "ok ") << diff << std::endl;

        int action = choose_action(probs, state->get_n_actions());
        state = state->step(action);
    }
}

int main() {
    const std::string folder_name = "../checkpoints";
    std::filesystem::path folder(folder_name);
    std::filesystem::path nn_path = folder / "migoyugo_strongest_900.pt";
    std::filesystem::path nnue_path = folder / "nnue_layerstacks_weights.bin";

    run_sanity_check_layerstacks(nn_path.string(), nnue_path.string());
}
