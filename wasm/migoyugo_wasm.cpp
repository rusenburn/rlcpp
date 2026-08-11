// The single C ABI between the Migoyugo engine and the browser.
//
// Every rule decision - legality, promotion, capture, Igo, Wego - is made here
// by MigoyugoBB, the bitboard engine that run/bench_migoyugo_bb.cpp
// differentially tests against the reference implementation. JavaScript is
// pure presentation and never forms an opinion about the position.
//
// Two constraints shape this file:
//
//  1. Nothing here may throw. Emscripten's default build turns a throw into an
//     abort that kills the module permanently - not an error you can catch. So
//     MigoyugoState is never touched, MigoyugoBB::from_short (which calls
//     std::stoi) is never called, positions are built by replaying validated
//     moves, and every failure is a return code.
//
//  2. Nothing here may trust its caller. MigoyugoBB::do_move guards legality
//     with an assert only, which -DNDEBUG removes; an out-of-range or illegal
//     square would write a duplicate bit into migo[], desync key/empty, and
//     silently corrupt every later snapshot and search. Game::apply is the one
//     funnel that live moves, undo-replay and load_moves all pass through, and
//     it validates before every do_move.

// Built natively too, so run/test_wasm_api.cpp can exercise this exact API in
// a debugger before any browser is involved.
#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#else
#define EMSCRIPTEN_KEEPALIVE
#endif

#include <games/migoyugo_bb.hpp>
#include <nnue/nnue_layerstacks_player_v2.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

namespace mgbb = rl::games::mgbb;

namespace
{

// ---------------------------------------------------------------------------
// Wire formats
//
// Fixed size, every field a single byte, no multi-byte values at all, so no
// endianness or alignment question ever arises and JavaScript reads the whole
// position in one slice. The masks are plain 0/1 byte arrays rather than
// bitboards because 64 bytes is nothing and it keeps BigInt out of the JS.
// ---------------------------------------------------------------------------

constexpr uint8_t kSnapshotVersion = 1;

constexpr uint8_t kStatusPlaying = 0;
constexpr uint8_t kStatusIgo = 1; // someone made four Yugos in a line
constexpr uint8_t kStatusWego = 2; // side to move has no legal move; Yugo count decides
constexpr uint8_t kNoSquare = 255;

#pragma pack(push, 1)
struct Snapshot
{
    uint8_t version;
    uint8_t stm;      // side to move: 0 = White, 1 = Black
    uint8_t status;   // kStatusPlaying / kStatusIgo / kStatusWego
    uint8_t winner;   // 0, 1, or 255 for "no winner yet" and for a drawn Wego

    uint8_t yugo_count[2];
    uint8_t migo_count[2];

    uint8_t move_count;
    uint8_t last_move;          // 0..63, 255 if none
    uint8_t last_was_promotion; // the last move created a Yugo
    uint8_t n_cleared;          // Migos that promotion removed, 0..12

    uint8_t legal_count;
    uint8_t can_undo;
    uint8_t flags;    // bit 0: the last move was played by the bot
    uint8_t reserved;

    uint8_t cleared[12]; // squares cleared by last_move, 255-padded
    uint8_t win_line[4]; // the four Yugo squares of a winning Igo, 255-padded

    uint8_t cells[64];     // 0 empty, 1 W-Migo, 2 W-Yugo, 3 B-Migo, 4 B-Yugo
    uint8_t legal[64];     // 1 = the side to move may play here
    uint8_t promoting[64]; // 1 = legal and promotes to a Yugo
    uint8_t winning[64];   // 1 = legal and wins outright by Igo
    uint8_t piline_w[64];  // 1 = empty square White may not play (would build a line of 5+)
    uint8_t piline_b[64];  // 1 = same for Black
};

struct Info
{
    double nodes;
    double nps;
    int32_t depth;
    int32_t score;      // engine units; JS divides by 1024
    int32_t best_move;  // 0..63, -1 for none
    int32_t elapsed_ms;
};
#pragma pack(pop)

// These are the contract with wasm/web/snapshot.js. When a field is added here
// the assert fires and the JS gets fixed in the same commit.
static_assert(sizeof(Snapshot) == 416, "Snapshot layout must match wasm/web/snapshot.js");
static_assert(sizeof(Info) == 32, "Info layout must match wasm/web/snapshot.js");

// Return codes
constexpr int kOk = 0;
constexpr int kErrNoModel = -1;
constexpr int kErrRange = -2;
constexpr int kErrIllegal = -3;
constexpr int kErrGameOver = -4;
constexpr int kErrBadModel = -5;
constexpr int kErrAlignment = -6;
constexpr int kErrNothingToUndo = -7;

// ---------------------------------------------------------------------------
// Game: the authoritative position, plus the derived facts a UI needs that the
// bitboard alone does not carry (what the last move captured, which four Yugos
// won the game).
// ---------------------------------------------------------------------------

class Game
{
public:
    Game() { reset(); }

    void reset()
    {
        board_ = mgbb::MigoyugoBB::initial();
        moves_.clear();
        status_ = kStatusPlaying;
        winner_ = kNoSquare;
        cleared_.clear();
        win_line_.fill(kNoSquare);
        last_was_promotion_ = false;
        last_by_bot_ = false;
        refresh_terminal();
    }

    const mgbb::MigoyugoBB& board() const { return board_; }
    bool over() const { return status_ != kStatusPlaying; }
    const std::vector<uint8_t>& moves() const { return moves_; }

    // The only path by which the position ever changes.
    int apply(int sq, bool by_bot)
    {
        if (sq < 0 || sq >= 64) return kErrRange;
        if (over()) return kErrGameOver;
        if (((board_.legal_moves() >> sq) & 1ULL) == 0) return kErrIllegal;

        const int mover = board_.stm;
        const uint64_t migo_before = board_.migo[mover];

        mgbb::Undo undo;
        const bool igo = board_.do_move<false>(sq, undo, nullptr);

        moves_.push_back(static_cast<uint8_t>(sq));
        last_by_bot_ = by_bot;
        last_was_promotion_ = undo.made_yugo;

        cleared_.clear();
        for (uint64_t b = migo_before & ~board_.migo[mover]; b; b &= b - 1)
            cleared_.push_back(static_cast<uint8_t>(mgbb::ctz64(b)));

        win_line_.fill(kNoSquare);
        if (igo)
        {
            status_ = kStatusIgo;
            winner_ = static_cast<uint8_t>(mover);
            find_win_line(mover, sq);
        }
        else
        {
            refresh_terminal();
        }
        return kOk;
    }

    // Undo by replaying the move list from the start rather than by unwinding
    // Undo records. Undo records are built to pair with one do_move inside a
    // search frame; keeping a parallel stack alive across the JS event loop
    // would be a second source of truth that can desync, and it would not
    // restore status/winner/win_line anyway. A full replay is at most 64
    // do_move calls with feature emission off - tens of microseconds.
    int undo(int plies)
    {
        if (plies <= 0) return kErrRange;
        if (moves_.empty()) return kErrNothingToUndo;

        const int keep = std::max(0, static_cast<int>(moves_.size()) - plies);
        const std::vector<uint8_t> replay(moves_.begin(), moves_.begin() + keep);
        reset();
        for (uint8_t sq : replay)
        {
            const int rc = apply(sq, false);
            if (rc != kOk) { reset(); return rc; } // cannot happen; fail clean if it does
        }
        return kOk;
    }

    // Restores a game from a move list, which is what makes worker-crash
    // recovery and share-a-position possible. Validates every move.
    int load_moves(const uint8_t* data, int n)
    {
        if (n < 0 || n > 64 || (n > 0 && !data)) return kErrRange;
        const std::vector<uint8_t> wanted(data, data + n);
        reset();
        for (uint8_t sq : wanted)
        {
            const int rc = apply(sq, false);
            if (rc != kOk) { reset(); return rc; }
        }
        return kOk;
    }

    void write_snapshot(Snapshot& s) const
    {
        std::memset(&s, 0, sizeof(s));
        s.version = kSnapshotVersion;
        s.stm = static_cast<uint8_t>(board_.stm);
        s.status = status_;
        s.winner = winner_;

        for (int c = 0; c < 2; ++c)
        {
            s.yugo_count[c] = static_cast<uint8_t>(board_.yugo_count(c));
            s.migo_count[c] = static_cast<uint8_t>(board_.migo_count(c));
        }

        s.move_count = static_cast<uint8_t>(moves_.size());
        s.last_move = moves_.empty() ? kNoSquare : moves_.back();
        s.last_was_promotion = last_was_promotion_ ? 1 : 0;
        s.can_undo = moves_.empty() ? 0 : 1;
        s.flags = last_by_bot_ ? 1 : 0;

        s.n_cleared = static_cast<uint8_t>(cleared_.size());
        std::memset(s.cleared, kNoSquare, sizeof(s.cleared));
        for (size_t i = 0; i < cleared_.size() && i < sizeof(s.cleared); ++i)
            s.cleared[i] = cleared_[i];

        for (size_t i = 0; i < win_line_.size(); ++i) s.win_line[i] = win_line_[i];

        spread(board_.migo[0], s.cells, 1);
        spread(board_.yugo[0], s.cells, 2);
        spread(board_.migo[1], s.cells, 3);
        spread(board_.yugo[1], s.cells, 4);

        // A finished game has no moves to offer, whatever the bitboards say.
        const uint64_t legal = over() ? 0ULL : board_.legal_moves();
        s.legal_count = static_cast<uint8_t>(mgbb::popcount64(legal));
        spread(legal, s.legal, 1);
        spread(board_.raw_makes4[board_.stm] & legal, s.promoting, 1);
        spread(board_.raw_igo[board_.stm] & legal, s.winning, 1);
        spread(board_.piline[0], s.piline_w, 1);
        spread(board_.piline[1], s.piline_b, 1);
    }

private:
    static void spread(uint64_t bb, uint8_t* out, uint8_t value)
    {
        for (uint64_t b = bb; b; b &= b - 1) out[mgbb::ctz64(b)] = value;
    }

    void refresh_terminal()
    {
        if (board_.legal_moves() == 0)
        {
            status_ = kStatusWego;
            const int a = board_.yugo_count(0);
            const int b = board_.yugo_count(1);
            winner_ = (a == b) ? kNoSquare : static_cast<uint8_t>(a > b ? 0 : 1);
        }
        else
        {
            status_ = kStatusPlaying;
            winner_ = kNoSquare;
        }
    }

    // The four Yugos of the winning line, for the UI to draw through. C++ has
    // this for free at the moment the win happens; deriving it in JS would
    // mean reimplementing run detection.
    void find_win_line(int color, int sq)
    {
        const uint64_t y = board_.yugo[color];
        const uint64_t bit = 1ULL << sq;
        static constexpr int kSteps[4] = { 1, 8, 9, 7 };

        for (int axis = 0; axis < 4; ++axis)
        {
            uint64_t run = bit;
            switch (kSteps[axis])
            {
            case 1: run |= mgbb::axis_neighbours<1>(y, bit); break;
            case 8: run |= mgbb::axis_neighbours<8>(y, bit); break;
            case 9: run |= mgbb::axis_neighbours<9>(y, bit); break;
            default: run |= mgbb::axis_neighbours<7>(y, bit); break;
            }
            if (mgbb::popcount64(run) >= 4)
            {
                int n = 0;
                for (uint64_t b = run; b && n < 4; b &= b - 1)
                    win_line_[n++] = static_cast<uint8_t>(mgbb::ctz64(b));
                return;
            }
        }
    }

    mgbb::MigoyugoBB board_;
    std::vector<uint8_t> moves_;
    std::vector<uint8_t> cleared_;
    std::array<uint8_t, 4> win_line_{};
    uint8_t status_ = kStatusPlaying;
    uint8_t winner_ = kNoSquare;
    bool last_was_promotion_ = false;
    bool last_by_bot_ = false;
};

// ---------------------------------------------------------------------------
// Module state
// ---------------------------------------------------------------------------

Game g_game;
std::shared_ptr<const NNUELayerStacksModelV2> g_model;
std::unique_ptr<rl::players::NNUELayerStacksPlayerV2> g_engine;
Snapshot g_snapshot;
Info g_info;
int g_time_ms = 1000;

void publish_info(int chosen)
{
    std::memset(&g_info, 0, sizeof(g_info));
    if (!g_engine) return;
    const double secs = g_engine->last_elapsed_s();
    g_info.nodes = static_cast<double>(g_engine->nodes());
    g_info.nps = secs > 0 ? g_info.nodes / secs : 0.0;
    g_info.depth = g_engine->last_depth();
    g_info.score = g_engine->last_score();
    g_info.best_move = chosen;
    g_info.elapsed_ms = static_cast<int32_t>(secs * 1000.0);
}

int run_search()
{
    if (!g_engine) return kErrNoModel;
    if (g_game.over()) return kErrGameOver;

    g_engine->set_max_duration(std::chrono::duration<int, std::milli>(g_time_ms));
    const int sq = g_engine->search_position(g_game.board());
    publish_info(sq);

    // search_position answers -1 for "position is over", which collides with
    // kErrNoModel. over() is already checked above so this cannot fire, but
    // the codes must not be ambiguous if it ever does.
    if (sq < 0) return kErrGameOver;
    return sq;
}

} // namespace

extern "C" {

// Parses the weights buffer and builds the engine. Returns kOk or a negative
// code; the module is unusable until this succeeds.
EMSCRIPTEN_KEEPALIVE
int mgy_init(const uint8_t* model_data, int model_len, int tt_mb)
{
    if (!model_data || model_len <= 0) return kErrBadModel;

    auto model = load_nnue_layerstacks_v2_from_memory(
        model_data, static_cast<size_t>(model_len), "weights sent from the browser");
    if (!model) return kErrBadModel;

    // The header's aligned-load reasoning depends on this holding. Cheap to
    // check, and it turns a subtle class of wrongness into a clear failure.
    if (reinterpret_cast<uintptr_t>(model.get()) % 64 != 0) return kErrAlignment;

    g_model = std::move(model);
    g_engine = std::make_unique<rl::players::NNUELayerStacksPlayerV2>(
        g_model,
        std::chrono::duration<int, std::milli>(g_time_ms),
        static_cast<size_t>(std::clamp(tt_mb, 1, 128)),
        /*verbose=*/false);

    g_game.reset();
    return kOk;
}

EMSCRIPTEN_KEEPALIVE
void mgy_new_game()
{
    g_game.reset();
    if (g_engine) g_engine->clear_tt();
}

EMSCRIPTEN_KEEPALIVE
int mgy_play(int sq) { return g_game.apply(sq, /*by_bot=*/false); }

EMSCRIPTEN_KEEPALIVE
int mgy_undo(int plies) { return g_game.undo(plies); }

EMSCRIPTEN_KEEPALIVE
int mgy_load_moves(const uint8_t* moves, int n) { return g_game.load_moves(moves, n); }

// Searches and plays. Returns the square, or a negative code.
EMSCRIPTEN_KEEPALIVE
int mgy_bot_move()
{
    const int sq = run_search();
    if (sq < 0) return sq;
    const int rc = g_game.apply(sq, /*by_bot=*/true);
    return rc == kOk ? sq : rc;
}

// Searches without playing: the hint button and the analysis readout.
EMSCRIPTEN_KEEPALIVE
int mgy_bot_suggest() { return run_search(); }

EMSCRIPTEN_KEEPALIVE
void mgy_set_time_ms(int ms) { g_time_ms = std::clamp(ms, 10, 60000); }

// Resizing drops the table, so this is deliberately separate from the time
// budget - changing difficulty must not throw away what the engine has learned.
EMSCRIPTEN_KEEPALIVE
void mgy_set_tt_mb(int mb)
{
    if (g_engine) g_engine->resize_tt(static_cast<size_t>(std::clamp(mb, 1, 128)));
}

EMSCRIPTEN_KEEPALIVE
void mgy_clear_tt() { if (g_engine) g_engine->clear_tt(); }

// NOT a pure getter: this REWRITES the snapshot from the current position and
// then returns its (fixed) address. Callers must call it on every read.
// Caching the address once and reading that memory thereafter leaves the board
// frozen at whatever position was current the first time - which looks like a
// UI that has stopped updating rather than like an error.
EMSCRIPTEN_KEEPALIVE
const uint8_t* mgy_snapshot()
{
    g_game.write_snapshot(g_snapshot);
    return reinterpret_cast<const uint8_t*>(&g_snapshot);
}

// A compile-time constant, exported anyway so a browser-cached worker.js built
// against a different layout fails loudly instead of misparsing.
EMSCRIPTEN_KEEPALIVE
int mgy_snapshot_size() { return static_cast<int>(sizeof(Snapshot)); }

EMSCRIPTEN_KEEPALIVE
const uint8_t* mgy_info() { return reinterpret_cast<const uint8_t*>(&g_info); }

EMSCRIPTEN_KEEPALIVE
int mgy_info_size() { return static_cast<int>(sizeof(Info)); }

} // extern "C"
