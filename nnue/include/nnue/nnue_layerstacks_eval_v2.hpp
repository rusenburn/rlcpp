#ifndef RL_NNUE_NNUE_LAYERSTACKS_EVAL_V2_HPP_
#define RL_NNUE_NNUE_LAYERSTACKS_EVAL_V2_HPP_

// The arithmetic of the 384-input layer-stacked NNUE, separated from the
// search that used to own it.
//
// This was private to NNUELayerStacksPlayerV2 and read that class's members
// (`acc_`, `state_`, `model_`), so a second search could not call it without
// copying ninety lines of intrinsics. The functions here take everything
// explicitly instead. NNUELayerStacksPlayerV2 now forwards to them, and the
// results must stay bit-identical - `bench_migoyugo_bb determinism` and
// `bench_migoyugo_bb match` are the check.
//
// Two conventions worth stating once:
//
//   * Accumulators are int16_t[256] and MUST be at least 16-byte aligned;
//     every load here is _mm_load_si128, not the unaligned form. Declare them
//     `alignas(64)`.
//
//   * Feature ids from MigoyugoBB::active_features() and FeatureDelta are
//     from White's point of view. The network wants the side to move's point
//     of view, so both perspectives are kept side by side: index 0 is White's
//     and index 1 is Black's, and evaluation reads `perspective[board.stm]`.
//
//   * evaluate_accumulator returns the RAW quantized sum. Callers apply their
//     own scaling: the alpha-beta search shifts it right by 4 to get engine
//     units, anything wanting the network's own scale divides by 128*128.

#include <games/migoyugo_bb.hpp>

#include "nnue_layerstacks_model_v2.hpp"

#include <immintrin.h>
#include <algorithm>
#include <cstdint>
#include <cstring>

namespace rl::nnue
{

namespace mgbb = rl::games::mgbb;

// The quantized output is scaled by (1 << QUANT_SHIFT) twice - once by the
// feature transformer and once by the head - so this is what converts a raw
// sum back to the scale the network was trained on, roughly [-1, +1] with +1
// a win for the side to move. Must match scripts/export_nnue_layerstacks_v2.py.
inline constexpr float kNNUEV2OutputScale = 128.0f * 128.0f;

// Layer-stack bucket, derived from the board rather than tracked alongside the
// accumulator so it cannot drift. Each Migo on the board cost about one turn
// and each Yugo about four (three Migos consumed plus the placement). Must stay
// in sync with compute_bucket_index in scripts/train_nnue_layerstacks_v2.py.
inline int compute_bucket_index(const mgbb::MigoyugoBB& board)
{
    const int turns = std::min(board.estimated_turns(), 80);
    return std::min(turns / 10, NNUELayerStacksModelV2::NUM_BUCKETS - 1);
}

// ---------------------------------------------------------- accumulator ---

inline void accumulator_add_feature(const NNUELayerStacksModelV2& model,
    int16_t* accumulator, int feature)
{
    const __m128i* w = reinterpret_cast<const __m128i*>(model.l1_weights[feature].data());
    __m128i* a = reinterpret_cast<__m128i*>(accumulator);
    for (int i = 0; i < 32; ++i) a[i] = _mm_add_epi16(a[i], w[i]);
}

// Both perspectives of a position, from scratch. Used for a root, or for any
// position reached other than by a move from one we already have.
inline void build_accumulator(const NNUELayerStacksModelV2& model,
    const mgbb::MigoyugoBB& board, int16_t (*perspective)[256])
{
    uint16_t features[192];
    const int n = board.active_features(features);

    for (int p = 0; p < 2; ++p)
        std::memcpy(perspective[p], model.l1_bias.data(), sizeof(model.l1_bias));

    for (int i = 0; i < n; ++i)
    {
        accumulator_add_feature(model, perspective[0], features[i]);
        accumulator_add_feature(model, perspective[1], mgbb::flip_perspective(features[i]));
    }
}

// Writes `dst` while reading `src`, so the copy is free: no separate memcpy
// pass, and undoing costs nothing because `src` is never touched.
inline void accumulator_transform(const NNUELayerStacksModelV2& model,
    int16_t* __restrict dst, const int16_t* __restrict src,
    const uint16_t* add, int n_add, const uint16_t* sub, int n_sub)
{
    // 64 int16 at a time: 8 live vectors, comfortably inside the 16 XMM
    // registers Ivy Bridge gives us, with the weight rows streamed through.
    for (int c = 0; c < 256; c += 64)
    {
        const __m128i* s = reinterpret_cast<const __m128i*>(src + c);
        __m128i v0 = _mm_load_si128(s + 0);
        __m128i v1 = _mm_load_si128(s + 1);
        __m128i v2 = _mm_load_si128(s + 2);
        __m128i v3 = _mm_load_si128(s + 3);
        __m128i v4 = _mm_load_si128(s + 4);
        __m128i v5 = _mm_load_si128(s + 5);
        __m128i v6 = _mm_load_si128(s + 6);
        __m128i v7 = _mm_load_si128(s + 7);

        for (int k = 0; k < n_add; ++k)
        {
            const __m128i* w = reinterpret_cast<const __m128i*>(model.l1_weights[add[k]].data() + c);
            v0 = _mm_add_epi16(v0, _mm_load_si128(w + 0));
            v1 = _mm_add_epi16(v1, _mm_load_si128(w + 1));
            v2 = _mm_add_epi16(v2, _mm_load_si128(w + 2));
            v3 = _mm_add_epi16(v3, _mm_load_si128(w + 3));
            v4 = _mm_add_epi16(v4, _mm_load_si128(w + 4));
            v5 = _mm_add_epi16(v5, _mm_load_si128(w + 5));
            v6 = _mm_add_epi16(v6, _mm_load_si128(w + 6));
            v7 = _mm_add_epi16(v7, _mm_load_si128(w + 7));
        }
        for (int k = 0; k < n_sub; ++k)
        {
            const __m128i* w = reinterpret_cast<const __m128i*>(model.l1_weights[sub[k]].data() + c);
            v0 = _mm_sub_epi16(v0, _mm_load_si128(w + 0));
            v1 = _mm_sub_epi16(v1, _mm_load_si128(w + 1));
            v2 = _mm_sub_epi16(v2, _mm_load_si128(w + 2));
            v3 = _mm_sub_epi16(v3, _mm_load_si128(w + 3));
            v4 = _mm_sub_epi16(v4, _mm_load_si128(w + 4));
            v5 = _mm_sub_epi16(v5, _mm_load_si128(w + 5));
            v6 = _mm_sub_epi16(v6, _mm_load_si128(w + 6));
            v7 = _mm_sub_epi16(v7, _mm_load_si128(w + 7));
        }

        __m128i* d = reinterpret_cast<__m128i*>(dst + c);
        _mm_store_si128(d + 0, v0);
        _mm_store_si128(d + 1, v1);
        _mm_store_si128(d + 2, v2);
        _mm_store_si128(d + 3, v3);
        _mm_store_si128(d + 4, v4);
        _mm_store_si128(d + 5, v5);
        _mm_store_si128(d + 6, v6);
        _mm_store_si128(d + 7, v7);
    }
}

// One move's worth of feature changes, both perspectives, dst <- src + delta.
inline void accumulator_apply_delta(const NNUELayerStacksModelV2& model,
    int16_t (*dst)[256], const int16_t (*src)[256], const mgbb::FeatureDelta& delta)
{
    for (int p = 0; p < 2; ++p)
    {
        uint16_t add[mgbb::FeatureDelta::CAPACITY];
        uint16_t sub[mgbb::FeatureDelta::CAPACITY];

        if (p == 0)
        {
            std::memcpy(add, delta.added, sizeof(uint16_t) * delta.n_added);
            std::memcpy(sub, delta.removed, sizeof(uint16_t) * delta.n_removed);
        }
        else
        {
            for (int i = 0; i < delta.n_added; ++i)
                add[i] = static_cast<uint16_t>(mgbb::flip_perspective(delta.added[i]));
            for (int i = 0; i < delta.n_removed; ++i)
                sub[i] = static_cast<uint16_t>(mgbb::flip_perspective(delta.removed[i]));
        }

        accumulator_transform(model, dst[p], src[p], add, delta.n_added, sub, delta.n_removed);
    }
}

// ----------------------------------------------------------------- eval ---

// Returns the raw quantized sum; see the header comment on scaling.
//
// Identical arithmetic to NNUELayerStacksPlayer::evaluate_nnue_simd, so a v1
// and a v2 export of the same checkpoint produce bit-identical values; only the
// clipped ReLU is vectorised and the L1 index order differs.
inline int32_t evaluate_accumulator(const NNUELayerStacksModelV2& model,
    const int16_t* accumulator, int bucket)
{
    alignas(64) int16_t activated[256];
    {
        const __m128i zero = _mm_setzero_si128();
        const __m128i c127 = _mm_set1_epi16(127);
        const __m128i* a = reinterpret_cast<const __m128i*>(accumulator);
        __m128i* o = reinterpret_cast<__m128i*>(activated);
        for (int i = 0; i < 32; ++i)
            o[i] = _mm_min_epi16(_mm_max_epi16(_mm_load_si128(a + i), zero), c127);
    }

    alignas(32) int32_t l2_out[16];
    for (int i = 0; i < 16; ++i)
    {
        __m128i sum = _mm_setzero_si128();
        const __m128i* w = reinterpret_cast<const __m128i*>(model.l2_weights[bucket][i].data());
        const __m128i* in = reinterpret_cast<const __m128i*>(activated);
        for (int j = 0; j < 32; ++j)
            sum = _mm_add_epi32(sum, _mm_madd_epi16(_mm_load_si128(w + j), _mm_load_si128(in + j)));

        alignas(16) int32_t parts[4];
        _mm_store_si128(reinterpret_cast<__m128i*>(parts), sum);
        const int32_t total = model.l2_bias[bucket][i] + parts[0] + parts[1] + parts[2] + parts[3];
        l2_out[i] = std::clamp(total >> 7, 0, 127);
    }

    alignas(32) int32_t l3_out[32];
    for (int i = 0; i < 32; ++i)
    {
        int32_t total = model.l3_bias[bucket][i];
        for (int j = 0; j < 16; ++j) total += model.l3_weights[bucket][i][j] * l2_out[j];
        l3_out[i] = std::clamp(total >> 7, 0, 127);
    }

    int32_t final_sum = model.out_bias[bucket];
    for (int i = 0; i < 32; ++i) final_sum += model.out_weights[bucket][i] * l3_out[i];

    return final_sum;
}

// Convenience for callers that hold both perspectives and want the value in
// the network's own scale, from the side to move's point of view.
inline float evaluate_position(const NNUELayerStacksModelV2& model,
    const int16_t (*perspective)[256], const mgbb::MigoyugoBB& board)
{
    const int32_t raw = evaluate_accumulator(model, perspective[board.stm], compute_bucket_index(board));
    return static_cast<float>(raw) / kNNUEV2OutputScale;
}

} // namespace rl::nnue

#endif
