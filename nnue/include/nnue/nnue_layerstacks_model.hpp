#pragma once
#include <cstdint>
#include <array>

#pragma pack(push, 1) // Ensure no padding between members

// Layer-stacked (bucketed multi-expert) NNUE model. The feature transformer
// (l1) is shared across every position, like the single-net NNUEModel; the
// small 256->16->32->1 head is replicated once per game-phase bucket
// (NUM_BUCKETS experts), selected at inference time by an estimated turn
// count (see NNUELayerStacksPlayer::compute_bucket_index). Field order must
// exactly match scripts/export_nnue_layerstacks.py's write order.
struct alignas(32) NNUELayerStacksModel {
    static constexpr int NUM_BUCKETS = 8;

    // Shared feature transformer: [Feature Index][Neuron Index]
    std::array<std::array<int16_t, 256>, 256> l1_weights;
    std::array<int16_t, 256> l1_bias;

    // Per-bucket L2: 16 neurons, each has 256 weights
    std::array<std::array<std::array<int16_t, 256>, 16>, NUM_BUCKETS> l2_weights;
    std::array<std::array<int32_t, 16>, NUM_BUCKETS> l2_bias;

    // Per-bucket L3: 32 neurons, each has 16 weights
    std::array<std::array<std::array<int16_t, 16>, 32>, NUM_BUCKETS> l3_weights;
    std::array<std::array<int32_t, 32>, NUM_BUCKETS> l3_bias;

    // Per-bucket output: 32 weights
    std::array<std::array<int16_t, 32>, NUM_BUCKETS> out_weights;
    std::array<int32_t, NUM_BUCKETS> out_bias;
};
#pragma pack(pop)
