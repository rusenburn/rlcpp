// Compares the big residual network (migoyugo_strongest_900.pt) against the
// quantized 384-feature layer-stacked NNUE, move by move over a self-played
// game - the v2 counterpart of run/sanity_check_layerstacks.cpp.
//
// The NNUE is what the big net was distilled into, so the two evaluations
// should track each other closely. A column of large differences means either
// the distillation did not converge or, more likely, that the feature
// extraction here and the feature extraction the trainer saw disagree.
//
// Features come from MigoyugoBB, the same environment the search uses, so this
// also checks that the 384-feature encoding is wired up consistently.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <vector>

#include <common/random.hpp>
#include <deeplearning/alphazero/networks/shared_res_nn.hpp>
#include <deeplearning/network_evaluator.hpp>
#include <games/migoyugo.hpp>
#include <games/migoyugo_bb.hpp>
#include <immintrin.h>
#include <nnue/nnue_layerstacks_model_v2.hpp>
#include <nnue/nnue_layerstacks_player_v2.hpp>

namespace mgbb = rl::games::mgbb;

int choose_action(const std::vector<float>& probs, int n_game_actions)
{
    float remaining = rl::common::get();
    int action = 0;
    const int last = n_game_actions - 1;
    while ((action < last) && ((remaining -= probs.at(action)) >= 0)) ++action;
    return action;
}

// Standalone copy of the evaluation, mirroring what run/sanity_check.cpp
// already does for the v1 net rather than reaching into the player's privates.
float evaluate_nnue_v2(const std::array<int16_t, 256>& accumulator,
    const NNUELayerStacksModelV2& model, int bucket)
{
    alignas(64) std::array<int16_t, 256> activated;
    for (size_t i = 0; i < 256; ++i)
        activated[i] = static_cast<int16_t>(std::clamp<int32_t>(accumulator[i], 0, 127));

    alignas(32) std::array<int32_t, 16> l2_out;
    for (size_t i = 0; i < 16; ++i)
    {
        __m128i sum = _mm_setzero_si128();
        for (size_t j = 0; j < 256; j += 8)
        {
            const __m128i w = _mm_loadu_si128((const __m128i*)&model.l2_weights[bucket][i][j]);
            const __m128i in = _mm_load_si128((const __m128i*)&activated[j]);
            sum = _mm_add_epi32(sum, _mm_madd_epi16(w, in));
        }
        alignas(16) int32_t parts[4];
        _mm_store_si128((__m128i*)parts, sum);
        const int32_t total = model.l2_bias[bucket][i] + parts[0] + parts[1] + parts[2] + parts[3];
        l2_out[i] = std::clamp(total >> 7, 0, 127);
    }

    alignas(32) std::array<int32_t, 32> l3_out;
    for (size_t i = 0; i < 32; ++i)
    {
        int32_t total = model.l3_bias[bucket][i];
        for (size_t j = 0; j < 16; ++j) total += model.l3_weights[bucket][i][j] * l2_out[j];
        l3_out[i] = std::clamp(total >> 7, 0, 127);
    }

    int32_t final_sum = model.out_bias[bucket];
    for (size_t i = 0; i < 32; ++i) final_sum += model.out_weights[bucket][i] * l3_out[i];

    return static_cast<float>(final_sum) / (128.0f * 128.0f);
}

void run_sanity_check_v2(const std::string& torch_model_path, const std::string& nnue_bin_path)
{
    auto state = rl::games::MigoyugoState::initialize();

    auto network_ptr = std::make_unique<rl::deeplearning::alphazero::SharedResNetwork>(
        state->get_observation_shape(), state->get_n_actions(), 128, 512, 5, true);

    const auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    network_ptr->load(torch_model_path);
    network_ptr->to(device);

    auto ev_ptr = std::make_unique<rl::deeplearning::NetworkEvaluator>(
        std::move(network_ptr), state->get_n_actions(), state->get_observation_shape());

    auto model = load_nnue_layerstacks_v2(nnue_bin_path);
    if (!model) return;

    std::cout << std::left << std::setw(8) << "Move"
        << std::setw(15) << "BigNet Eval"
        << std::setw(15) << "NNUE Eval"
        << std::setw(9) << "Bucket"
        << std::setw(9) << "Piline"
        << "Difference" << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    double sum_abs_diff = 0.0;
    double max_diff = 0.0;
    int samples = 0;

    for (int i = 0; i < 100 && !state->is_terminal(); ++i)
    {
        auto [probs, output] = ev_ptr->evaluate(state);
        const float torch_val = output[0];

        // Build the accumulator from scratch for this position - this check is
        // about how closely the quantized net tracks the big one, not about the
        // incremental update path.
        const auto bb = mgbb::MigoyugoBB::from_short(state->to_short());

        uint16_t feats[192];
        const int n = bb.active_features(feats);

        std::array<int16_t, 256> accumulator{};
        for (int j = 0; j < 256; ++j) accumulator[j] = model->l1_bias[j];

        for (int k = 0; k < n; ++k)
        {
            // active_features() reports from White's point of view; the net is
            // trained on the side to move's point of view.
            const int id = (bb.stm == 0) ? feats[k] : mgbb::flip_perspective(feats[k]);
            for (int j = 0; j < 256; ++j) accumulator[j] += model->l1_weights[id][j];
        }

        const int bucket = rl::players::NNUELayerStacksPlayerV2::compute_bucket_index(bb);
        const float nnue_val = evaluate_nnue_v2(accumulator, *model, bucket);
        const int n_piline = mgbb::popcount64(bb.piline[0]) + mgbb::popcount64(bb.piline[1]);

        const float diff = std::abs(torch_val - nnue_val);
        sum_abs_diff += diff;
        max_diff = std::max<double>(max_diff, diff);
        ++samples;

        std::cout << std::left << std::setw(8) << i
            << std::setw(15) << torch_val
            << std::setw(15) << nnue_val
            << std::setw(9) << bucket
            << std::setw(9) << n_piline
            << (diff > 0.3 ? "warn " : "ok ") << diff << std::endl;

        state = state->step(choose_action(probs, state->get_n_actions()));
    }

    if (samples)
    {
        std::cout << "\nmean |difference| " << sum_abs_diff / samples
            << "   max " << max_diff << "   over " << samples << " positions" << std::endl;
    }
}

int main(int argc, char** argv)
{
    const std::filesystem::path folder("../checkpoints");
    const std::string nn_name = argc > 1 ? argv[1] : "migoyugo_strongest_900.pt";
    const std::string nnue_name = argc > 2 ? argv[2] : "nnue_layerstacks_v2_weights.bin";

    run_sanity_check_v2((folder / nn_name).string(), (folder / nnue_name).string());
    return 0;
}
