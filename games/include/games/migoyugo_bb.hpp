#ifndef RL_GAMES_MIGOYUGO_BB_HPP_
#define RL_GAMES_MIGOYUGO_BB_HPP_

// Allocation-free bitboard environment for Migoyugo, built for alpha-beta
// search with an incrementally-updated NNUE accumulator.
//
// MigoyugoLightState is the reference implementation and stays the source of
// truth for the rules; this header must agree with it move for move (see
// run/bench_migoyugo_bb.cpp for the differential test). The difference is
// purely mechanical: no heap allocation, no std::vector, make/unmake instead
// of building a new state per node, and every rule query answered by a
// handful of 64-bit shifts instead of 64 squares' worth of ray walks.
//
// Two invariants of the game do most of the work here:
//
//  1. In any position reachable by legal play, neither side has an unbroken
//     run of 4 or more of its own pieces. A run of 4 can only come into
//     existence as its 4th piece is placed, and that placement immediately
//     promotes to a Yugo and removes every Migo in the run; the run survives
//     only if all four are Yugos, which is an instant Igo win and ends the
//     game. So for every empty square, on every axis, left + right <= 3.
//     That makes ">= 4" and "== 4" interchangeable and lets the promotion
//     test be a plain popcount.
//
//  2. Positions can never repeat. Yugos are permanent, so a promoting move
//     strictly increases the Yugo count and any other move strictly increases
//     the occupied count; (yugo_count, occupancy) rises lexicographically
//     every ply. No repetition detection or ply cap is needed.

#include <cstdint>
#include <cassert>
#include <string>
#include <cctype>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace rl::games::mgbb
{

// --------------------------------------------------------------------------
// Bit helpers
// --------------------------------------------------------------------------

inline int ctz64(uint64_t x)
{
#if defined(_MSC_VER)
    unsigned long idx;
    _BitScanForward64(&idx, x);
    return static_cast<int>(idx);
#else
    return __builtin_ctzll(x);
#endif
}

inline int popcount64(uint64_t x)
{
#if defined(_MSC_VER)
    return static_cast<int>(__popcnt64(x));
#else
    return __builtin_popcountll(x);
#endif
}

// --------------------------------------------------------------------------
// Geometry
//
// Bit index is row * 8 + col, matching MigoyugoLightState's action encoding.
// The four axes are stepped by +1 (east), +8 (south), +9 (south-east) and
// +7 (south-west). fwd<S> moves every bit one step along the axis, bwd<S>
// moves it one step back; each masks the *result* so that bits never wrap
// around a file edge. Multi-step reads are built by iterating these, which
// keeps the masking correct by construction.
// --------------------------------------------------------------------------

constexpr uint64_t NOT_FILE_A = 0xfefefefefefefefeULL; // col >= 1
constexpr uint64_t NOT_FILE_H = 0x7f7f7f7f7f7f7f7fULL; // col <= 6
constexpr uint64_t ALL_SQ = ~0ULL;

template <int S> struct AxisMask;
template <> struct AxisMask<1> { static constexpr uint64_t FWD = NOT_FILE_A, BWD = NOT_FILE_H; };
template <> struct AxisMask<8> { static constexpr uint64_t FWD = ALL_SQ, BWD = ALL_SQ; };
template <> struct AxisMask<9> { static constexpr uint64_t FWD = NOT_FILE_A, BWD = NOT_FILE_H; };
template <> struct AxisMask<7> { static constexpr uint64_t FWD = NOT_FILE_H, BWD = NOT_FILE_A; };

template <int S> inline uint64_t fwd(uint64_t x) { return (x << S) & AxisMask<S>::FWD; }
template <int S> inline uint64_t bwd(uint64_t x) { return (x >> S) & AxisMask<S>::BWD; }

// --------------------------------------------------------------------------
// Run analysis
//
// For occupancy p and axis step S, define
//   R[k][e] = 1 iff the k squares e+S .. e+kS are all in p
//   L[k][e] = 1 iff the k squares e-S .. e-kS are all in p
// so placing at e produces an unbroken run of length left + right + 1.
//
//   illegal (run >= 5) = L4 | R1&L3 | R2&L2 | R3&L1 | R4
//   makes4  (run >= 4) = L3 | R1&L2 | R2&L1 | R3
//
// Verified against a brute-force reference over 30,000 random boards with
// zero mismatches.
// --------------------------------------------------------------------------

template <int S>
inline void axis_runs(uint64_t p, uint64_t& illegal, uint64_t& makes4)
{
    const uint64_t r1 = bwd<S>(p);
    const uint64_t r2 = r1 & bwd<S>(bwd<S>(p));
    const uint64_t r3 = r2 & bwd<S>(bwd<S>(bwd<S>(p)));
    const uint64_t r4 = r3 & bwd<S>(bwd<S>(bwd<S>(bwd<S>(p))));

    const uint64_t l1 = fwd<S>(p);
    const uint64_t l2 = l1 & fwd<S>(fwd<S>(p));
    const uint64_t l3 = l2 & fwd<S>(fwd<S>(fwd<S>(p)));
    const uint64_t l4 = l3 & fwd<S>(fwd<S>(fwd<S>(fwd<S>(p))));

    illegal |= l4 | (r1 & l3) | (r2 & l2) | (r3 & l1) | r4;
    makes4 |= l3 | (r1 & l2) | (r2 & l1) | r3;
}

// Both derived masks for one player's occupancy, NOT yet intersected with the
// empty squares. Depends only on that player's own pieces, which is why an
// opponent move never invalidates it.
inline void compute_runs(uint64_t p, uint64_t& raw_illegal, uint64_t& raw_makes4)
{
    raw_illegal = 0;
    raw_makes4 = 0;
    axis_runs<1>(p, raw_illegal, raw_makes4);
    axis_runs<8>(p, raw_illegal, raw_makes4);
    axis_runs<9>(p, raw_illegal, raw_makes4);
    axis_runs<7>(p, raw_illegal, raw_makes4);
}

// Just the "completes a line of 4" half, used for the Yugo-only board (a
// square in this set wins the game by Igo).
inline uint64_t compute_makes4(uint64_t p)
{
    uint64_t illegal = 0, makes4 = 0;
    compute_runs(p, illegal, makes4);
    return makes4;
}

// Non-zero iff p already contains an unbroken line of 4. Only needed for a
// position handed to us from outside (from_short); during search an Igo is
// detected as it is created.
inline uint64_t has_line_of_4(uint64_t p)
{
    uint64_t acc = 0;
    acc |= p & bwd<1>(p) & bwd<1>(bwd<1>(p)) & bwd<1>(bwd<1>(bwd<1>(p)));
    acc |= p & bwd<8>(p) & bwd<8>(bwd<8>(p)) & bwd<8>(bwd<8>(bwd<8>(p)));
    acc |= p & bwd<9>(p) & bwd<9>(bwd<9>(p)) & bwd<9>(bwd<9>(bwd<9>(p)));
    acc |= p & bwd<7>(p) & bwd<7>(bwd<7>(p)) & bwd<7>(bwd<7>(bwd<7>(p)));
    return acc;
}

// All of `own` contiguous with sqbb along one axis, capped at 3 per side.
// In a legal position the true streak is at most 3 in total, so the cap never
// truncates and popcount(result) == 3 is an exact test for "this placement
// completes a line of exactly 4 on this axis".
template <int S>
inline uint64_t axis_neighbours(uint64_t own, uint64_t sqbb)
{
    uint64_t run = 0, x = sqbb;
    for (int i = 0; i < 3; ++i) { x = fwd<S>(x) & own; if (!x) break; run |= x; }
    x = sqbb;
    for (int i = 0; i < 3; ++i) { x = bwd<S>(x) & own; if (!x) break; run |= x; }
    return run;
}

// --------------------------------------------------------------------------
// Feature layout: 384 inputs = 6 channels of 64, from the side to move's
// point of view. Channels 0-3 keep the meaning MigoyugoState::get_observation
// has always given them, so old training data stays compatible.
// --------------------------------------------------------------------------

constexpr int N_SQUARES = 64;
constexpr int N_CHANNELS = 6;
constexpr int N_FEATURES = N_CHANNELS * N_SQUARES; // 384

constexpr int CH_OUR_MIGO = 0;
constexpr int CH_OUR_YUGO = 1;
constexpr int CH_OPP_MIGO = 2;
constexpr int CH_OPP_YUGO = 3;
constexpr int CH_OUR_PILINE = 4; // empty squares we may not play on
constexpr int CH_OPP_PILINE = 5; // empty squares they may not play on

// Feature ids are stored from White's point of view. Adding PERSP_DELTA of the
// id's channel converts to Black's point of view; the table is an involution,
// so applying it twice is the identity.
constexpr int PERSP_DELTA[N_CHANNELS] = { +128, +128, -128, -128, +64, -64 };

inline int flip_perspective(int white_id) { return white_id + PERSP_DELTA[white_id >> 6]; }

inline int migo_feature(int color, int sq) { return color * 128 + sq; }
inline int yugo_feature(int color, int sq) { return 64 + color * 128 + sq; }
inline int piline_feature(int color, int sq) { return 256 + color * 64 + sq; }

// Fixed-capacity change list for one move.
//
// Capacity 80 per side is provably safe: the piece channels contribute one
// addition and at most 12 removals (4 axes x 3 companions, which are disjoint
// because the placed square is never its own companion), and the two piline
// channels contribute at most 64 toggles each, split across the two lists.
// Measured worst case over random play is 14 total.
struct FeatureDelta
{
    static constexpr int CAPACITY = 80;

    uint16_t added[CAPACITY];
    uint16_t removed[CAPACITY];
    int n_added;
    int n_removed;

    void clear() { n_added = 0; n_removed = 0; }
    void add(int id) { assert(n_added < CAPACITY); added[n_added++] = static_cast<uint16_t>(id); }
    void remove(int id) { assert(n_removed < CAPACITY); removed[n_removed++] = static_cast<uint16_t>(id); }
};

// --------------------------------------------------------------------------
// Zobrist keys, generated at compile time so there is no initialisation order
// to worry about. Only the pieces and the side to move are hashed - the piline
// and run masks are pure functions of those, so hashing them would be pure
// waste.
// --------------------------------------------------------------------------

struct ZobristTable
{
    uint64_t piece[4][64]; // [white migo, white yugo, black migo, black yugo][square]
    uint64_t side;
};

constexpr uint64_t splitmix64(uint64_t& state)
{
    state += 0x9e3779b97f4a7c15ULL;
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

constexpr ZobristTable make_zobrist()
{
    ZobristTable t{};
    uint64_t s = 0x243f6a8885a308d3ULL;
    for (int k = 0; k < 4; ++k)
        for (int sq = 0; sq < 64; ++sq)
            t.piece[k][sq] = splitmix64(s);
    t.side = splitmix64(s);
    return t;
}

inline constexpr ZobristTable ZOBRIST = make_zobrist();

inline uint64_t zob_migo(int color, int sq) { return ZOBRIST.piece[color * 2 + 0][sq]; }
inline uint64_t zob_yugo(int color, int sq) { return ZOBRIST.piece[color * 2 + 1][sq]; }

// --------------------------------------------------------------------------
// Undo record: everything do_move overwrites that cannot be cheaply rederived.
// `empty` is left out because it is one OR-and-complement away from the piece
// boards. POD, lives in the caller's recursion frame.
// --------------------------------------------------------------------------

struct Undo
{
    uint64_t cleared;          // Migos removed by the promotion (0 for a plain placement)
    uint64_t old_raw_illegal;  // mover's, before the move
    uint64_t old_raw_makes4;   // mover's, before the move
    uint64_t old_raw_igo;      // mover's, before the move
    uint64_t old_piline[2];
    uint64_t old_key;
    int sq;
    bool made_yugo;
};

// --------------------------------------------------------------------------
// The state
// --------------------------------------------------------------------------

class MigoyugoBB
{
public:
    uint64_t migo[2]{};
    uint64_t yugo[2]{};

    // Derived from migo[c] | yugo[c] - a move by one player never changes the
    // other player's copy, which halves the per-move work.
    uint64_t raw_illegal[2]{}; // placing here would make a run of 5+
    uint64_t raw_makes4[2]{};  // placing here completes a run of 4 (promotes)
    uint64_t raw_igo[2]{};     // placing here completes 4 Yugos in a row: instant win

    uint64_t piline[2]{};      // raw_illegal[c] & empty - the NNUE feature sets
    uint64_t empty{ ALL_SQ };
    uint64_t key{ 0 };

    int stm{ 0 };
    int ply{ 0 };

    MigoyugoBB() { recompute_all(); }

    static MigoyugoBB initial() { return MigoyugoBB(); }

    // Parses the FEN-ish encoding produced by MigoyugoState::to_short() /
    // MigoyugoLightState::to_short(): 8 slash-separated ranks of 'x' (White
    // Migo), 'X' (White Yugo), 'o' (Black Migo), 'O' (Black Yugo) and run
    // lengths of empty squares, then a space and the side to move.
    static MigoyugoBB from_short(const std::string& s)
    {
        MigoyugoBB b;
        b.migo[0] = b.migo[1] = b.yugo[0] = b.yugo[1] = 0;

        const size_t space_pos = s.find_last_of(' ');
        const std::string board_part = s.substr(0, space_pos);
        const int player = std::stoi(s.substr(space_pos + 1));

        int r = 0, c = 0;
        for (char ch : board_part)
        {
            if (ch == '/') { ++r; c = 0; continue; }
            if (std::isdigit(static_cast<unsigned char>(ch))) { c += (ch - '0'); continue; }

            if (r < 8 && c < 8)
            {
                const uint64_t bit = 1ULL << (r * 8 + c);
                switch (ch)
                {
                case 'x': b.migo[0] |= bit; break;
                case 'X': b.yugo[0] |= bit; break;
                case 'o': b.migo[1] |= bit; break;
                case 'O': b.yugo[1] |= bit; break;
                default: break;
                }
            }
            ++c;
        }

        b.stm = player;
        b.ply = 0;
        b.recompute_all();
        return b;
    }

    uint64_t occupancy(int c) const { return migo[c] | yugo[c]; }
    uint64_t legal_moves() const { return empty & ~raw_illegal[stm]; }
    uint64_t legal_moves_for(int c) const { return empty & ~raw_illegal[c]; }

    // Moves that promote to a Yugo, and the subset of those that win outright.
    uint64_t promoting_moves() const { return raw_makes4[stm] & legal_moves(); }
    uint64_t winning_moves() const { return raw_igo[stm] & legal_moves(); }

    int yugo_count(int c) const { return popcount64(yugo[c]); }
    int migo_count(int c) const { return popcount64(migo[c]); }

    // Estimated turn count used to pick the layer-stack bucket: each Migo on
    // the board cost one turn, each Yugo cost four (three Migos consumed plus
    // the placement). Derived from the board rather than tracked incrementally
    // so it cannot drift out of sync with the accumulator.
    int estimated_turns() const
    {
        return popcount64(migo[0] | migo[1]) + 4 * popcount64(yugo[0] | yugo[1]);
    }

    // The game is over for the side to move when it has no legal move (Wego).
    // An Igo win is detected as it happens inside do_move, so search never
    // needs this; it exists for a position handed in from outside.
    bool is_terminal_root() const
    {
        if (has_line_of_4(yugo[0]) || has_line_of_4(yugo[1])) return true;
        return legal_moves() == 0;
    }

    // Wego result from the side to move's point of view: +1 win, -1 loss, 0 draw.
    float wego_reward() const
    {
        const int a = yugo_count(stm);
        const int b = yugo_count(1 - stm);
        if (a > b) return 1.0f;
        if (a < b) return -1.0f;
        return 0.0f;
    }

    // Writes every active feature id (White's point of view) and returns the
    // count. At most 64 pieces + 128 piline cells.
    int active_features(uint16_t* out) const
    {
        int n = 0;
        for (int c = 0; c < 2; ++c)
        {
            for (uint64_t b = migo[c]; b; b &= b - 1) out[n++] = static_cast<uint16_t>(migo_feature(c, ctz64(b)));
            for (uint64_t b = yugo[c]; b; b &= b - 1) out[n++] = static_cast<uint16_t>(yugo_feature(c, ctz64(b)));
            for (uint64_t b = piline[c]; b; b &= b - 1) out[n++] = static_cast<uint16_t>(piline_feature(c, ctz64(b)));
        }
        return n;
    }

    // ----------------------------------------------------------------------
    // Make / unmake
    // ----------------------------------------------------------------------

    // Plays `sq`, which must be legal. Returns true if the move won the game
    // outright by forming four Yugos in a row (Igo).
    //
    // When EMIT is true, every NNUE feature that turned on or off is appended
    // to `delta`; perft and rule testing pass false and skip that work.
    template <bool EMIT>
    bool do_move(int sq, Undo& u, FeatureDelta* delta)
    {
        const int p = stm;
        const uint64_t sqbb = 1ULL << sq;
        const uint64_t own_before = occupancy(p);

        assert((legal_moves() >> sq) & 1ULL);

        u.sq = sq;
        u.old_key = key;
        u.old_raw_illegal = raw_illegal[p];
        u.old_raw_makes4 = raw_makes4[p];
        u.old_raw_igo = raw_igo[p];
        u.old_piline[0] = piline[0];
        u.old_piline[1] = piline[1];
        if (EMIT) delta->clear();

        bool igo = false;

        if (sqbb & raw_makes4[p])
        {
            // Promotion. Only now - on ~14% of moves - do we walk the rays, to
            // find which Migos the completed lines consume. An axis whose three
            // companions contain no Migo at all was already four Yugos, so the
            // placement wins by Igo.
            uint64_t cleared = 0;
            resolve_promotion(own_before, migo[p], sqbb, cleared, igo);

            yugo[p] |= sqbb;
            migo[p] &= ~cleared;
            u.cleared = cleared;
            u.made_yugo = true;

            key ^= zob_yugo(p, sq);
            if (EMIT) delta->add(yugo_feature(p, sq));
            for (uint64_t b = cleared; b; b &= b - 1)
            {
                const int s = ctz64(b);
                key ^= zob_migo(p, s);
                if (EMIT) delta->remove(migo_feature(p, s));
            }
        }
        else
        {
            migo[p] |= sqbb;
            u.cleared = 0;
            u.made_yugo = false;

            key ^= zob_migo(p, sq);
            if (EMIT) delta->add(migo_feature(p, sq));
        }

        empty = ~(migo[0] | yugo[0] | migo[1] | yugo[1]);

        compute_runs(occupancy(p), raw_illegal[p], raw_makes4[p]);
        if (u.made_yugo) raw_igo[p] = compute_makes4(yugo[p]);

        update_piline<EMIT>(delta);

        key ^= ZOBRIST.side;
        stm = 1 - p;
        ++ply;

        return igo;
    }

    bool do_move(int sq, Undo& u, FeatureDelta& delta) { return do_move<true>(sq, u, &delta); }
    bool do_move(int sq, Undo& u) { return do_move<false>(sq, u, nullptr); }

    void undo_move(const Undo& u)
    {
        --ply;
        stm = 1 - stm;

        const int p = stm;
        const uint64_t sqbb = 1ULL << u.sq;

        if (u.made_yugo)
        {
            yugo[p] &= ~sqbb;
            migo[p] |= u.cleared;
        }
        else
        {
            migo[p] &= ~sqbb;
        }

        empty = ~(migo[0] | yugo[0] | migo[1] | yugo[1]);
        raw_illegal[p] = u.old_raw_illegal;
        raw_makes4[p] = u.old_raw_makes4;
        raw_igo[p] = u.old_raw_igo;
        piline[0] = u.old_piline[0];
        piline[1] = u.old_piline[1];
        key = u.old_key;
    }

    void recompute_all()
    {
        empty = ~(migo[0] | yugo[0] | migo[1] | yugo[1]);
        key = 0;
        for (int c = 0; c < 2; ++c)
        {
            compute_runs(occupancy(c), raw_illegal[c], raw_makes4[c]);
            raw_igo[c] = compute_makes4(yugo[c]);
            piline[c] = raw_illegal[c] & empty;
            for (uint64_t b = migo[c]; b; b &= b - 1) key ^= zob_migo(c, ctz64(b));
            for (uint64_t b = yugo[c]; b; b &= b - 1) key ^= zob_yugo(c, ctz64(b));
        }
        if (stm == 1) key ^= ZOBRIST.side;
    }

private:
    template <bool EMIT>
    void update_piline(FeatureDelta* delta)
    {
        for (int c = 0; c < 2; ++c)
        {
            const uint64_t next = raw_illegal[c] & empty;
            if (EMIT)
            {
                const uint64_t on = next & ~piline[c];
                const uint64_t off = piline[c] & ~next;
                for (uint64_t b = on; b; b &= b - 1) delta->add(piline_feature(c, ctz64(b)));
                for (uint64_t b = off; b; b &= b - 1) delta->remove(piline_feature(c, ctz64(b)));
            }
            piline[c] = next;
        }
    }

    static void resolve_promotion(uint64_t own, uint64_t own_migo, uint64_t sqbb,
        uint64_t& cleared, bool& igo)
    {
        cleared = 0;
        igo = false;
        check_axis<1>(own, own_migo, sqbb, cleared, igo);
        check_axis<8>(own, own_migo, sqbb, cleared, igo);
        check_axis<9>(own, own_migo, sqbb, cleared, igo);
        check_axis<7>(own, own_migo, sqbb, cleared, igo);
    }

    template <int S>
    static void check_axis(uint64_t own, uint64_t own_migo, uint64_t sqbb,
        uint64_t& cleared, bool& igo)
    {
        const uint64_t run = axis_neighbours<S>(own, sqbb);
        if (popcount64(run) != 3) return;
        const uint64_t migos = run & own_migo;
        cleared |= migos;
        if (migos == 0) igo = true; // all three companions were Yugos
    }
};

} // namespace rl::games::mgbb

#endif
