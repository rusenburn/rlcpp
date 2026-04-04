#pragma once
#include <cstdint>
#include <immintrin.h>
#pragma pack(push, 1) // Ensure no padding between members
struct NNUEModel {
    // L1: 256 neurons, 256 inputs
    int16_t l1_weights[256][256];
    int16_t l1_bias[256];

    // L2: 16 neurons, 256 inputs
    int16_t l2_weights[16][256];
    int32_t l2_bias[16];

    // L3: 32 neurons, 16 inputs
    int16_t l3_weights[32][16];
    int32_t l3_bias[32];

    // Output: 1 neuron, 32 inputs
    int16_t out_weights[32];
    int32_t out_bias;
};


struct alignas(32) NNUEModel2 {
    // [Feature Index][Neuron Index]
    // Using 256x256 for L1
    std::array<std::array<int16_t, 256>, 256> l1_weights;
    std::array<int16_t, 256> l1_bias;

    // L2: 16 Neurons, each has 256 weights
    std::array<std::array<int16_t, 256>, 16> l2_weights;
    std::array<int32_t, 16> l2_bias;

    // L3: 32 Neurons, each has 16 weights
    std::array<std::array<int16_t, 16>, 32> l3_weights;
    std::array<int32_t, 32> l3_bias;

    // Output: 32 weights
    std::array<int16_t, 32> out_weights;
    int32_t out_bias;
};
#pragma pack(pop)