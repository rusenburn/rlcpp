#pragma once

// Layer-stacked NNUE, version 2.
//
// Two changes from NNUELayerStacksModel:
//
//  1. l1_weights is stored [feature][neuron], not [neuron][feature]. The v1
//     layout is what PyTorch's nn.Linear emits ([out, in]) and the exporter
//     wrote it through unchanged, so toggling one feature in the accumulator
//     walked a column: stride 512 bytes, 256 distinct cache lines, the whole
//     weight table streamed per toggle. Transposed, a toggle is 512
//     contiguous bytes - 32 aligned SSE loads off one page.
//
//  2. 384 input features instead of 256. Channels 0-3 are unchanged
//     (OUR_MIGO, OUR_YUGO, OPP_MIGO, OPP_YUGO); channels 4 and 5 are the
//     "piline" sets - the empty squares each player is forbidden to play on
//     because doing so would build an unbroken line of more than four. See
//     games/include/games/migoyugo_bb.hpp.
//
// The file is a 64-byte header followed by the raw struct, so a stale or
// mismatched weights file is rejected instead of being read as noise. Field
// order must match scripts/export_nnue_layerstacks_v2.py exactly.

#include <cstdint>
#include <cstddef>
#include <array>
#include <cstdio>
#include <memory>
#include <new>
#include <string>
#include <iostream>

struct alignas(64) NNUELayerStacksModelV2
{
    static constexpr int NUM_BUCKETS = 8;
    static constexpr int NUM_FEATURES = 384;
    static constexpr int L1_SIZE = 256;
    static constexpr int L2_SIZE = 16;
    static constexpr int L3_SIZE = 32;

    // Shared feature transformer, TRANSPOSED: [feature][neuron].
    // Every row is 512 bytes, and the struct is 64-byte aligned, so every row
    // is 64-byte aligned too and _mm_load_si128 is always safe.
    std::array<std::array<int16_t, L1_SIZE>, NUM_FEATURES> l1_weights;
    std::array<int16_t, L1_SIZE> l1_bias;

    // Per-bucket heads, each a full 256 -> 16 -> 32 -> 1 stack.
    std::array<std::array<std::array<int16_t, L1_SIZE>, L2_SIZE>, NUM_BUCKETS> l2_weights;
    std::array<std::array<int32_t, L2_SIZE>, NUM_BUCKETS> l2_bias;

    std::array<std::array<std::array<int16_t, L2_SIZE>, L3_SIZE>, NUM_BUCKETS> l3_weights;
    std::array<std::array<int32_t, L3_SIZE>, NUM_BUCKETS> l3_bias;

    std::array<std::array<int16_t, L3_SIZE>, NUM_BUCKETS> out_weights;
    std::array<int32_t, NUM_BUCKETS> out_bias;
};

// alignas(64) rounds sizeof() up past the last field, so the number of bytes
// actually on disk is the extent of the fields, not sizeof(). Reading sizeof()
// bytes would demand 32 bytes of padding the exporter never writes.
inline constexpr size_t kNNUEModelV2PayloadBytes =
offsetof(NNUELayerStacksModelV2, out_bias) + sizeof(NNUELayerStacksModelV2::out_bias);

static_assert(kNNUEModelV2PayloadBytes <= sizeof(NNUELayerStacksModelV2),
    "payload cannot exceed the struct it is read into");

// 64 bytes, so the struct that follows it in the file stays 64-byte aligned
// if anyone ever mmaps the thing.
struct NNUEModelV2Header
{
    uint32_t magic;      // 'M','Y','U','2'
    uint32_t version;    // 2
    uint32_t n_features; // 384
    uint32_t l1_size;    // 256
    uint32_t l2_size;    // 16
    uint32_t l3_size;    // 32
    uint32_t n_buckets;  // 8
    uint32_t quant_shift; // 7  (weights scaled by 1 << quant_shift)
    uint8_t reserved[32];
};

inline constexpr uint32_t kNNUEModelV2Magic = 0x3255594dU; // "MYU2" little-endian
inline constexpr uint32_t kNNUEModelV2Version = 2;

// Loads a v2 weights file. Returns nullptr and explains itself on any problem -
// the v1 loaders silently accepted a missing or truncated file and played on
// with an uninitialised model.
inline std::shared_ptr<const NNUELayerStacksModelV2> load_nnue_layerstacks_v2(const std::string& path)
{
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f)
    {
        std::cerr << "[nnue-v2] cannot open " << path << std::endl;
        return nullptr;
    }

    NNUEModelV2Header header{};
    if (std::fread(&header, sizeof(header), 1, f) != 1)
    {
        std::cerr << "[nnue-v2] " << path << " is too short to hold a header" << std::endl;
        std::fclose(f);
        return nullptr;
    }

    if (header.magic != kNNUEModelV2Magic)
    {
        std::cerr << "[nnue-v2] " << path << " is not a v2 weights file"
            << " (magic 0x" << std::hex << header.magic << std::dec
            << "); re-export with scripts/export_nnue_layerstacks_v2.py" << std::endl;
        std::fclose(f);
        return nullptr;
    }
    if (header.version != kNNUEModelV2Version ||
        header.n_features != NNUELayerStacksModelV2::NUM_FEATURES ||
        header.l1_size != NNUELayerStacksModelV2::L1_SIZE ||
        header.l2_size != NNUELayerStacksModelV2::L2_SIZE ||
        header.l3_size != NNUELayerStacksModelV2::L3_SIZE ||
        header.n_buckets != NNUELayerStacksModelV2::NUM_BUCKETS)
    {
        std::cerr << "[nnue-v2] " << path << " has geometry "
            << header.n_features << "x" << header.l1_size << "x" << header.l2_size
            << "x" << header.l3_size << " with " << header.n_buckets << " buckets (version "
            << header.version << "), which this build does not implement" << std::endl;
        std::fclose(f);
        return nullptr;
    }

    auto model = std::make_shared<NNUELayerStacksModelV2>();
    if (std::fread(model.get(), kNNUEModelV2PayloadBytes, 1, f) != 1)
    {
        std::cerr << "[nnue-v2] " << path << " is truncated: expected "
            << kNNUEModelV2PayloadBytes << " bytes of weights" << std::endl;
        std::fclose(f);
        return nullptr;
    }

    // Anything past the weights means the exporter and this struct disagree.
    char extra;
    const bool trailing = std::fread(&extra, 1, 1, f) == 1;
    std::fclose(f);
    if (trailing)
    {
        std::cerr << "[nnue-v2] " << path << " has trailing bytes after the weights;"
            << " the exporter and NNUELayerStacksModelV2 are out of sync" << std::endl;
        return nullptr;
    }

    return model;
}
