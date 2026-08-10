#ifndef RL_NNUE_NNUE_LAYERSTACKS_PLAYER_V2_HPP_
#define RL_NNUE_NNUE_LAYERSTACKS_PLAYER_V2_HPP_

// Alpha-beta Migoyugo engine on the bitboard environment, evaluated by the
// 384-input layer-stacked NNUE.
//
// Differences from NNUELayerStacksPlayer, roughly in order of how much they
// matter:
//
//   * No allocation anywhere in the search. The old player built a new state
//     (heap) per node, and produced four std::vectors per node for the feature
//     update plus two more for the move list and threat list.
//   * Bitboard make/unmake, so legality, promotion and win detection are a
//     handful of shifts instead of 64 squares of ray walking. The old player
//     recomputed the full legality mask three times per node.
//   * Transposed L1 weights, so an accumulator toggle reads 512 contiguous
//     bytes instead of walking a 512-byte stride across 256 cache lines.
//   * An accumulator stack: the child's accumulator is written as the parent's
//     is read, so undo is a pointer decrement rather than a second pass of
//     inverse updates.
//   * A transposition table, principal variation search, killers, history,
//     late move reductions and aspiration windows.
//   * Forced-move detection: because a move can never change the opponent's
//     own no-long-lines mask, a position where the opponent has two winning
//     squares is lost outright, and one where they have exactly one has a
//     move list of size one.
//
// Everything the search touches lives on the stack or in members; the only
// shared table is the TT, which stores scores and moves - never accumulators.

#include <common/player.hpp>
#include <games/migoyugo_bb.hpp>

#include "nnue_layerstacks_model_v2.hpp"

#include <immintrin.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

namespace rl::players
{

namespace mgbb = rl::games::mgbb;

class NNUELayerStacksPlayerV2 : public rl::common::IPlayer
{
public:
    static constexpr int MAX_PLY = 96;
    static constexpr int MATE = 30000;
    static constexpr int MATE_IN_MAX = MATE - 2 * MAX_PLY;

    // Scores are the quantized network output shifted down by 4, so 1024 is
    // roughly "winning by 1.0" in the units the net was trained on.
    static constexpr int EVAL_SHIFT = 4;

    NNUELayerStacksPlayerV2(std::shared_ptr<const NNUELayerStacksModelV2> model,
        std::chrono::duration<int, std::milli> max_duration,
        size_t tt_size_mb = 64,
        bool verbose = true)
        : model_(std::move(model)), max_duration_(max_duration), verbose_(verbose)
    {
        resize_tt(tt_size_mb);
    }

    int choose_action(const std::unique_ptr<rl::common::IState>& state_ptr) override
    {
        start_time_ = std::chrono::high_resolution_clock::now();
        time_up_ = false;
        nodes_ = 0;
        ++generation_;

        root_ = mgbb::MigoyugoBB::from_short(state_ptr->to_short());

        std::memset(killers_, 0xff, sizeof(killers_));
        age_history();

        init_root_accumulator();

        const uint64_t legal = root_.legal_moves();
        if (legal == 0) return -1; // terminal; nothing to choose

        // An immediate Igo needs no search at all.
        const uint64_t instant = root_.winning_moves();
        if (instant) return mgbb::ctz64(instant);

        // Only one legal move: play it.
        if ((legal & (legal - 1)) == 0) return mgbb::ctz64(legal);

        state_ = root_;
        int best_move = mgbb::ctz64(legal);
        int best_score = 0;
        int completed_depth = 0;

        int alpha = -MATE;
        int beta = MATE;
        int window = 64;

        for (int depth = 1; depth <= MAX_PLY - 8; ++depth)
        {
            while (true)
            {
                root_best_move_ = -1;
                const int score = search_root(depth, alpha, beta);
                if (time_up_) break;

                if (score <= alpha)
                {
                    // Fail low: re-search with a lower floor, keeping beta so
                    // the window does not explode in both directions at once.
                    beta = (alpha + beta) / 2;
                    alpha = std::max(-MATE, score - window);
                    window *= 4;
                    continue;
                }
                if (score >= beta)
                {
                    beta = std::min(MATE, score + window);
                    window *= 4;
                    continue;
                }

                best_score = score;
                if (root_best_move_ >= 0) best_move = root_best_move_;
                completed_depth = depth;
                break;
            }

            if (time_up_) break;

            // A proven result cannot improve with more depth.
            if (std::abs(best_score) >= MATE_IN_MAX) break;

            window = 64;
            alpha = std::max(-MATE, best_score - window);
            beta = std::min(MATE, best_score + window);

            const auto now = std::chrono::high_resolution_clock::now();
            if ((now - start_time_) * 2 > max_duration_) break;
        }

        if (verbose_)
        {
            const double secs = std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - start_time_).count();
            std::cout << "NNUE-LS-v2  depth " << completed_depth
                << "\tscore " << best_score / 1024.0
                << "\tnodes " << nodes_
                << "\tnps " << static_cast<uint64_t>(secs > 0 ? nodes_ / secs : 0)
                << "\tmove " << best_move << std::endl;
        }

        return best_move;
    }

    // Fixed-depth entry point for benchmarking and for the forced-move
    // soundness test, which needs to compare scores at equal depth.
    int search_fixed_depth(const mgbb::MigoyugoBB& position, int depth, int* out_move = nullptr)
    {
        start_time_ = std::chrono::high_resolution_clock::now();
        time_up_ = false;
        nodes_ = 0;
        ++generation_;
        root_ = position;
        std::memset(killers_, 0xff, sizeof(killers_));
        init_root_accumulator();
        state_ = root_;
        root_best_move_ = -1;
        const int score = search_root(depth, -MATE, MATE);
        if (out_move) *out_move = root_best_move_;
        return score;
    }

    void set_use_forced_moves(bool on) { use_forced_moves_ = on; }
    // Late move reductions are a heuristic, not a score-preserving transform:
    // at a fixed depth they change the value whenever move ordering changes.
    // Turning them off makes the search exact, which is what the forced-move
    // soundness test needs in order to compare scores at all.
    void set_use_lmr(bool on) { use_lmr_ = on; }
    uint64_t nodes() const { return nodes_; }

    void resize_tt(size_t mb)
    {
        size_t clusters = (mb * 1024 * 1024) / sizeof(TTCluster);
        size_t pow2 = 1;
        while (pow2 * 2 <= clusters) pow2 *= 2;
        if (pow2 < 1024) pow2 = 1024;
        tt_.assign(pow2, TTCluster{});
        tt_mask_ = pow2 - 1;
    }

    void clear_tt()
    {
        std::fill(tt_.begin(), tt_.end(), TTCluster{});
        std::memset(history_, 0, sizeof(history_));
        generation_ = 0;
    }

    // Bucket selection, derived from the board rather than tracked alongside
    // the accumulator so it cannot drift. Each Migo on the board cost about
    // one turn and each Yugo about four (three Migos consumed plus the
    // placement). Must stay in sync with compute_bucket_index in
    // scripts/train_nnue_layerstacks_v2.py.
    static int compute_bucket_index(const mgbb::MigoyugoBB& s)
    {
        const int turns = std::min(s.estimated_turns(), 80);
        return std::min(turns / 10, NNUELayerStacksModelV2::NUM_BUCKETS - 1);
    }

private:
    // ---------------------------------------------------------------- TT ---

    static constexpr uint8_t BOUND_NONE = 0;
    static constexpr uint8_t BOUND_UPPER = 1;
    static constexpr uint8_t BOUND_LOWER = 2;
    static constexpr uint8_t BOUND_EXACT = 3;
    static constexpr int16_t NO_EVAL = INT16_MIN;

    struct TTEntry
    {
        uint32_t key32;
        int16_t value;
        int16_t eval;
        uint8_t depth;
        uint8_t gen_bound; // generation << 2 | bound
        uint8_t move;      // 0..63, 0xff for none
        uint8_t pad[5];    // pads the entry to 16 bytes: 4 per cache line
    };
    struct TTCluster { TTEntry e[4]; };
    static_assert(sizeof(TTEntry) == 16, "TT entry should be 16 bytes");
    static_assert(sizeof(TTCluster) == 64, "TT cluster should be one cache line");

    TTEntry* tt_probe(uint64_t key, bool& hit)
    {
        TTCluster& c = tt_[(key >> 32) & tt_mask_];
        const uint32_t k32 = static_cast<uint32_t>(key);

        for (int i = 0; i < 4; ++i)
        {
            if (c.e[i].key32 == k32 && (c.e[i].gen_bound & 3) != BOUND_NONE)
            {
                c.e[i].gen_bound = static_cast<uint8_t>((generation_ << 2) | (c.e[i].gen_bound & 3));
                hit = true;
                return &c.e[i];
            }
        }

        // Depth-preferred, with entries from older searches heavily discounted.
        hit = false;
        TTEntry* victim = &c.e[0];
        int worst = INT32_MAX;
        for (int i = 0; i < 4; ++i)
        {
            if ((c.e[i].gen_bound & 3) == BOUND_NONE) return &c.e[i];
            const int age = ((c.e[i].gen_bound >> 2) == (generation_ & 63)) ? 0 : 1;
            const int value = c.e[i].depth - 8 * age;
            if (value < worst) { worst = value; victim = &c.e[i]; }
        }
        return victim;
    }

    void tt_store(TTEntry* e, uint64_t key, int value, int eval, int depth, uint8_t bound, int move, int ply)
    {
        const uint32_t k32 = static_cast<uint32_t>(key);

        // Never throw away a deeper result for the same position from this
        // same search, unless the new one is exact.
        if (e->key32 == k32 && depth < e->depth - 2 && bound != BOUND_EXACT) return;

        e->key32 = k32;
        e->value = static_cast<int16_t>(to_tt_score(value, ply));
        e->eval = static_cast<int16_t>(eval);
        e->depth = static_cast<uint8_t>(std::clamp(depth, 0, 255));
        e->gen_bound = static_cast<uint8_t>((generation_ << 2) | bound);
        e->move = static_cast<uint8_t>(move < 0 ? 0xff : move);
    }

    // Mate scores are stored as distance-to-mate from the entry's own position
    // so they stay valid when the position is reached by a different path.
    static int to_tt_score(int v, int ply)
    {
        if (v >= MATE_IN_MAX) return v + ply;
        if (v <= -MATE_IN_MAX) return v - ply;
        return v;
    }
    static int from_tt_score(int v, int ply)
    {
        if (v >= MATE_IN_MAX) return v - ply;
        if (v <= -MATE_IN_MAX) return v + ply;
        return v;
    }

    // ----------------------------------------------------- accumulator ---

    void init_root_accumulator()
    {
        uint16_t feats[192];
        const int n = root_.active_features(feats);

        for (int p = 0; p < 2; ++p)
            std::memcpy(acc_[0][p], model_->l1_bias.data(), sizeof(model_->l1_bias));

        for (int i = 0; i < n; ++i)
        {
            add_row(acc_[0][0], feats[i]);
            add_row(acc_[0][1], mgbb::flip_perspective(feats[i]));
        }
    }

    void add_row(int16_t* acc, int feature)
    {
        const __m128i* w = reinterpret_cast<const __m128i*>(model_->l1_weights[feature].data());
        __m128i* a = reinterpret_cast<__m128i*>(acc);
        for (int i = 0; i < 32; ++i) a[i] = _mm_add_epi16(a[i], w[i]);
    }

    // Writes the child accumulator while reading the parent's, so the copy is
    // free: no separate memcpy pass, and undo costs nothing because the parent
    // slot is never touched.
    void apply_delta(int parent_ply, int child_ply, const mgbb::FeatureDelta& d)
    {
        for (int persp = 0; persp < 2; ++persp)
        {
            uint16_t add[mgbb::FeatureDelta::CAPACITY];
            uint16_t sub[mgbb::FeatureDelta::CAPACITY];

            if (persp == 0)
            {
                std::memcpy(add, d.added, sizeof(uint16_t) * d.n_added);
                std::memcpy(sub, d.removed, sizeof(uint16_t) * d.n_removed);
            }
            else
            {
                for (int i = 0; i < d.n_added; ++i)
                    add[i] = static_cast<uint16_t>(mgbb::flip_perspective(d.added[i]));
                for (int i = 0; i < d.n_removed; ++i)
                    sub[i] = static_cast<uint16_t>(mgbb::flip_perspective(d.removed[i]));
            }

            transform(acc_[child_ply][persp], acc_[parent_ply][persp],
                add, d.n_added, sub, d.n_removed);
        }
    }

    void transform(int16_t* __restrict dst, const int16_t* __restrict src,
        const uint16_t* add, int na, const uint16_t* sub, int ns)
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

            for (int k = 0; k < na; ++k)
            {
                const __m128i* w = reinterpret_cast<const __m128i*>(model_->l1_weights[add[k]].data() + c);
                v0 = _mm_add_epi16(v0, _mm_load_si128(w + 0));
                v1 = _mm_add_epi16(v1, _mm_load_si128(w + 1));
                v2 = _mm_add_epi16(v2, _mm_load_si128(w + 2));
                v3 = _mm_add_epi16(v3, _mm_load_si128(w + 3));
                v4 = _mm_add_epi16(v4, _mm_load_si128(w + 4));
                v5 = _mm_add_epi16(v5, _mm_load_si128(w + 5));
                v6 = _mm_add_epi16(v6, _mm_load_si128(w + 6));
                v7 = _mm_add_epi16(v7, _mm_load_si128(w + 7));
            }
            for (int k = 0; k < ns; ++k)
            {
                const __m128i* w = reinterpret_cast<const __m128i*>(model_->l1_weights[sub[k]].data() + c);
                v0 = _mm_sub_epi16(v0, _mm_load_si128(w + 0));
                v1 = _mm_sub_epi16(v1, _mm_load_si128(w + 1));
                v2 = _mm_sub_epi16(v2, _mm_load_si128(w + 2));
                v3 = _mm_sub_epi16(v3, _mm_load_si128(w + 3));
                v4 = _mm_sub_epi16(v4, _mm_load_si128(w + 4));
                v5 = _mm_sub_epi16(v5, _mm_load_si128(w + 5));
                v6 = _mm_sub_epi16(v6, _mm_load_si128(w + 6));
                v7 = _mm_sub_epi16(v7, _mm_load_si128(w + 7));
            }

            __m128i* dv = reinterpret_cast<__m128i*>(dst + c);
            _mm_store_si128(dv + 0, v0);
            _mm_store_si128(dv + 1, v1);
            _mm_store_si128(dv + 2, v2);
            _mm_store_si128(dv + 3, v3);
            _mm_store_si128(dv + 4, v4);
            _mm_store_si128(dv + 5, v5);
            _mm_store_si128(dv + 6, v6);
            _mm_store_si128(dv + 7, v7);
        }
    }

    // ------------------------------------------------------------- eval ---

    // Identical arithmetic to NNUELayerStacksPlayer::evaluate_nnue_simd, so a
    // v1 and a v2 export of the same checkpoint produce bit-identical values;
    // only the clipped ReLU is vectorised and the L1 index order differs.
    int evaluate(int ply)
    {
        const int16_t* accumulator = acc_[ply][state_.stm];
        const int bucket = compute_bucket_index(state_);
        const NNUELayerStacksModelV2& m = *model_;

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
            const __m128i* w = reinterpret_cast<const __m128i*>(m.l2_weights[bucket][i].data());
            const __m128i* in = reinterpret_cast<const __m128i*>(activated);
            for (int j = 0; j < 32; ++j)
                sum = _mm_add_epi32(sum, _mm_madd_epi16(_mm_load_si128(w + j), _mm_load_si128(in + j)));

            alignas(16) int32_t parts[4];
            _mm_store_si128(reinterpret_cast<__m128i*>(parts), sum);
            const int32_t total = m.l2_bias[bucket][i] + parts[0] + parts[1] + parts[2] + parts[3];
            l2_out[i] = std::clamp(total >> 7, 0, 127);
        }

        alignas(32) int32_t l3_out[32];
        for (int i = 0; i < 32; ++i)
        {
            int32_t total = m.l3_bias[bucket][i];
            for (int j = 0; j < 16; ++j) total += m.l3_weights[bucket][i][j] * l2_out[j];
            l3_out[i] = std::clamp(total >> 7, 0, 127);
        }

        int32_t final_sum = m.out_bias[bucket];
        for (int i = 0; i < 32; ++i) final_sum += m.out_weights[bucket][i] * l3_out[i];

        return final_sum >> EVAL_SHIFT;
    }

    // ----------------------------------------------------------- search ---

    bool out_of_time()
    {
        if ((++nodes_ & 2047) != 0) return time_up_;
        if (std::chrono::high_resolution_clock::now() - start_time_ > max_duration_) time_up_ = true;
        return time_up_;
    }

    // Score for a Wego: the side to move has no legal move, so the game ends
    // now and the Yugo count decides it.
    int wego_score(int ply) const
    {
        const int mine = state_.yugo_count(state_.stm);
        const int theirs = state_.yugo_count(1 - state_.stm);
        if (mine > theirs) return MATE - ply;
        if (mine < theirs) return -MATE + ply;
        return 0;
    }

    // A move never changes the opponent's own no-long-lines mask, so their
    // winning squares stay winning no matter what we play - unless we occupy
    // one. Two of them is therefore a loss, and one of them makes our move
    // list a single element.
    //
    // This is sound in both directions: with two threats our one placement can
    // block at most one, and they still have a legal move so the game cannot
    // end early by Wego; with one threat, any move other than the block leaves
    // it available and losing.
    enum class Verdict { Search, Decided };

    Verdict apply_forced_moves(uint64_t& legal, int ply, int& score)
    {
        if (!use_forced_moves_) return Verdict::Search;

        const int opp = 1 - state_.stm;
        const uint64_t threats = state_.raw_igo[opp] & state_.legal_moves_for(opp);
        if (threats == 0) return Verdict::Search;

        if (threats & (threats - 1)) // two or more
        {
            score = -MATE + ply + 2;
            return Verdict::Decided;
        }

        const uint64_t block = threats & legal;
        if (block == 0)
        {
            score = -MATE + ply + 2;
            return Verdict::Decided;
        }

        legal = block;
        return Verdict::Search;
    }

    struct MoveList
    {
        int move[64];
        int score[64];
        int count;
    };

    void order_moves(uint64_t legal, int ply, int tt_move, MoveList& ml)
    {
        const uint64_t promoting = state_.raw_makes4[state_.stm] & legal;
        const uint64_t blocking = state_.raw_makes4[1 - state_.stm] & legal;
        const int k0 = killers_[ply][0];
        const int k1 = killers_[ply][1];
        const int stm = state_.stm;

        ml.count = 0;
        for (uint64_t b = legal; b; b &= b - 1)
        {
            const int sq = mgbb::ctz64(b);
            int s;
            if (sq == tt_move) s = 1 << 28;
            else if ((promoting >> sq) & 1) s = (1 << 24) + (((blocking >> sq) & 1) << 20);
            else if (sq == k0) s = 1 << 23;
            else if (sq == k1) s = (1 << 23) - 1;
            else if ((blocking >> sq) & 1) s = 1 << 22;
            else s = history_[stm][sq];

            ml.move[ml.count] = sq;
            ml.score[ml.count] = s;
            ++ml.count;
        }
    }

    // Selection sort one element at a time - with a few dozen moves and
    // frequent early cutoffs this beats sorting the whole list.
    static int pick_next(MoveList& ml, int i)
    {
        int best = i;
        for (int j = i + 1; j < ml.count; ++j)
            if (ml.score[j] > ml.score[best]) best = j;
        std::swap(ml.move[i], ml.move[best]);
        std::swap(ml.score[i], ml.score[best]);
        return ml.move[i];
    }

    void record_cutoff(int move, int ply, int depth, const MoveList& ml, int played)
    {
        if (killers_[ply][0] != move)
        {
            killers_[ply][1] = killers_[ply][0];
            killers_[ply][0] = move;
        }

        const int stm = state_.stm;
        const int bonus = depth * depth;
        history_[stm][move] += bonus;
        // Penalise the quiet moves that were tried first and failed, which is
        // what makes history informative rather than just a popularity count.
        for (int i = 0; i < played; ++i)
        {
            const int m = ml.move[i];
            if (m != move) history_[stm][m] -= bonus / 2;
        }

        if (history_[stm][move] > (1 << 20)) age_history();
    }

    void age_history()
    {
        for (int c = 0; c < 2; ++c)
            for (int sq = 0; sq < 64; ++sq)
                history_[c][sq] /= 2;
    }

    int search_root(int depth, int alpha, int beta)
    {
        uint64_t legal = state_.legal_moves();

        int decided = 0;
        if (apply_forced_moves(legal, 0, decided) == Verdict::Decided)
        {
            // The position is lost against best play, but the opponent still
            // has to find it - block one of their winning squares if we can
            // rather than returning whatever move happens to sort first.
            const int opp = 1 - state_.stm;
            const uint64_t block = state_.raw_igo[opp] & legal;
            root_best_move_ = mgbb::ctz64(block ? block : legal);
            return decided;
        }

        bool hit = false;
        TTEntry* tte = tt_probe(state_.key, hit);
        const int tt_move = (hit && tte->move != 0xff) ? tte->move : -1;

        MoveList ml;
        order_moves(legal, 0, tt_move, ml);

        int best = -MATE - 1; // so the first move is always recorded
        int best_move = -1;
        const int alpha_orig = alpha;

        for (int i = 0; i < ml.count; ++i)
        {
            const int move = pick_next(ml, i);

            mgbb::Undo u;
            mgbb::FeatureDelta delta;
            const bool igo = state_.do_move(move, u, delta);

            int score;
            if (igo)
            {
                score = MATE - 1;
            }
            else
            {
                apply_delta(0, 1, delta);
                if (i == 0)
                {
                    score = -search(depth - 1, -beta, -alpha, 1);
                }
                else
                {
                    score = -search(depth - 1, -alpha - 1, -alpha, 1);
                    if (score > alpha && score < beta)
                        score = -search(depth - 1, -beta, -alpha, 1);
                }
            }

            state_.undo_move(u);

            if (time_up_) break;

            if (score > best)
            {
                best = score;
                best_move = move;
                if (score > alpha) alpha = score;
                if (alpha >= beta) break;
            }
        }

        if (!time_up_ && best_move >= 0)
        {
            root_best_move_ = best_move;
            const uint8_t bound = best >= beta ? BOUND_LOWER
                : (best > alpha_orig ? BOUND_EXACT : BOUND_UPPER);
            tt_store(tte, state_.key, best, NO_EVAL, depth, bound, best_move, 0);
        }

        return best;
    }

    int search(int depth, int alpha, int beta, int ply)
    {
        if (out_of_time()) return 0;

        // Mate-distance pruning. Worth its three lines here because the forced
        // move rule above produces a lot of mate scores.
        alpha = std::max(alpha, -MATE + ply);
        beta = std::min(beta, MATE - ply - 1);
        if (alpha >= beta) return alpha;

        uint64_t legal = state_.legal_moves();
        if (legal == 0) return wego_score(ply);

        // An immediate Igo ends the search of this node right here.
        if (state_.winning_moves()) return MATE - ply - 1;

        if (ply >= MAX_PLY - 2) return evaluate(ply);

        int decided = 0;
        if (apply_forced_moves(legal, ply, decided) == Verdict::Decided) return decided;

        bool hit = false;
        TTEntry* tte = tt_probe(state_.key, hit);
        int tt_move = -1;
        int static_eval = NO_EVAL;

        if (hit)
        {
            if (tte->move != 0xff) tt_move = tte->move;
            static_eval = tte->eval;

            if (tte->depth >= depth)
            {
                const int v = from_tt_score(tte->value, ply);
                const uint8_t bound = tte->gen_bound & 3;
                if (bound == BOUND_EXACT) return v;
                if (bound == BOUND_LOWER && v >= beta) return v;
                if (bound == BOUND_UPPER && v <= alpha) return v;
            }
        }

        if (depth <= 0)
        {
            // No quiescence search: the only "noisy" move is a promotion, and
            // promotions beget promotions rather than converging. The Igo test
            // above already catches the tactic that matters at the horizon.
            const int e = (static_eval != NO_EVAL) ? static_eval : evaluate(ply);
            if (static_eval == NO_EVAL)
                tt_store(tte, state_.key, e, e, 0, BOUND_EXACT, -1, ply);
            return e;
        }

        MoveList ml;
        order_moves(legal, ply, tt_move, ml);

        const uint64_t promoting = state_.raw_makes4[state_.stm] & legal;
        const int alpha_orig = alpha;
        int best = -MATE - 1; // so the first move is always recorded
        int best_move = -1;

        for (int i = 0; i < ml.count; ++i)
        {
            const int move = pick_next(ml, i);
            const bool interesting = (move == tt_move)
                || ((promoting >> move) & 1)
                || move == killers_[ply][0]
                || move == killers_[ply][1];

            mgbb::Undo u;
            mgbb::FeatureDelta delta;
            const bool igo = state_.do_move(move, u, delta);

            int score;
            if (igo)
            {
                score = MATE - ply - 1;
            }
            else
            {
                apply_delta(ply, ply + 1, delta);

                // Late move reductions: after the first few moves, and only for
                // moves nothing has flagged as interesting, search shallower
                // first and re-search at full depth if it beats alpha.
                int reduction = 0;
                if (use_lmr_ && depth >= 3 && i >= 4 && !interesting)
                    reduction = 1 + (i > 12 ? 1 : 0);

                if (i == 0)
                {
                    score = -search(depth - 1, -beta, -alpha, ply + 1);
                }
                else
                {
                    score = -search(depth - 1 - reduction, -alpha - 1, -alpha, ply + 1);
                    if (score > alpha && reduction > 0)
                        score = -search(depth - 1, -alpha - 1, -alpha, ply + 1);
                    if (score > alpha && score < beta)
                        score = -search(depth - 1, -beta, -alpha, ply + 1);
                }
            }

            state_.undo_move(u);

            if (time_up_) return 0;

            if (score > best)
            {
                best = score;
                best_move = move;
                if (score > alpha) alpha = score;
                if (alpha >= beta)
                {
                    if (!((promoting >> move) & 1)) record_cutoff(move, ply, depth, ml, i);
                    break;
                }
            }
        }

        const uint8_t bound = best >= beta ? BOUND_LOWER
            : (best > alpha_orig ? BOUND_EXACT : BOUND_UPPER);
        tt_store(tte, state_.key, best, static_eval, depth, bound, best_move, ply);

        return best;
    }

    // ----------------------------------------------------------- members ---

    alignas(64) int16_t acc_[MAX_PLY + 1][2][256];

    std::shared_ptr<const NNUELayerStacksModelV2> model_;
    std::chrono::duration<int, std::milli> max_duration_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;

    mgbb::MigoyugoBB root_;
    mgbb::MigoyugoBB state_;

    std::vector<TTCluster> tt_;
    size_t tt_mask_{ 0 };

    int killers_[MAX_PLY][2]{};
    int history_[2][64]{};

    uint64_t nodes_{ 0 };
    int root_best_move_{ -1 };
    int generation_{ 0 };
    bool time_up_{ false };
    bool verbose_{ true };
    bool use_forced_moves_{ true };
    bool use_lmr_{ true };
};

} // namespace rl::players

#endif
