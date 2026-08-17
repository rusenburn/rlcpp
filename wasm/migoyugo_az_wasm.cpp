// C ABI for the AlphaZero bot in the browser: Amcts2 over MigoyugoState, with
// the residual network evaluated by onnxruntime-web on WebGPU.
//
// This is a second, independent module alongside migoyugo_wasm.cpp. They share
// no code and no build. Two reasons they stay apart:
//
//  1. This one is linked with -sASYNCIFY, because onnxruntime-web's run()
//     returns a Promise and Amcts2::search() is a synchronous loop. Asyncify
//     instruments the module; the NNUE engine should not pay for that.
//
//  2. They speak different game representations. The NNUE engine is built on
//     MigoyugoBB (bitboards, alpha-beta). Amcts2 is written against IState, so
//     this module is built on MigoyugoState, the reference implementation.
//
// Like migoyugo_wasm.cpp this file is written to be throw-free - Emscripten's
// default build turns a throw into a permanent abort - so every entry point
// validates its arguments and returns a code.

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#else
#define EMSCRIPTEN_KEEPALIVE
#endif

#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include <games/migoyugo.hpp>
#include <onnxeval/onnx_evaluator.hpp>
#include <players/bandits/amcts2/amcts2.hpp>

#if defined(__EMSCRIPTEN__)
#include <onnxeval/onnx_session_web.hpp>
#endif

namespace
{
constexpr int kBoardCells = 64;
constexpr int kNActions = 64;

constexpr uint8_t kSnapshotVersion = 1;
constexpr uint8_t kStatusPlaying = 0;
constexpr uint8_t kStatusOver = 1;
constexpr uint8_t kNoSquare = 255;

constexpr int kOk = 0;
constexpr int kErrNoModel = -1;
constexpr int kErrRange = -2;
constexpr int kErrIllegal = -3;
constexpr int kErrGameOver = -4;
constexpr int kErrNothingToUndo = -7;

// Search defaults. Dirichlet noise is off: this is a bot playing a human, not
// self-play generating training data, and noise only makes it weaker.
constexpr float kCpuct = 2.0f;
constexpr float kTemperature = 1.0f;
constexpr float kDirichletEpsilon = 0.0f;
constexpr float kDirichletAlpha = 0.3f;
constexpr float kDefaultVisits = 1.0f;
constexpr float kDefaultWins = -1.0f;

// Floor on simulations per move, whatever the clock says. See mgy_az_bot_suggest.
constexpr int kMinSimulations = 1;

#pragma pack(push, 1)
// Deliberately smaller than migoyugo_wasm.cpp's Snapshot: single-byte fields,
// no multi-byte values, so JavaScript reads the position in one slice with no
// endianness question. See wasm/web/az_snapshot.js for the mirror.
struct AzSnapshot
{
    uint8_t version;
    uint8_t stm;     // side to move: 0 = White, 1 = Black
    uint8_t status;  // kStatusPlaying / kStatusOver
    uint8_t winner;  // 0, 1, or 255 for "none yet" and for a draw

    uint8_t move_count;
    uint8_t last_move; // 0..63, 255 if none
    uint8_t can_undo;
    uint8_t reserved;

    uint8_t cells[kBoardCells]; // 0 empty, 1 W-Migo, 2 W-Yugo, 3 B-Migo, 4 B-Yugo
    uint8_t legal[kBoardCells]; // 1 = the side to move may play here
};
#pragma pack(pop)
static_assert(sizeof(AzSnapshot) == 136, "AzSnapshot layout must match wasm/web/az_snapshot.js");

// ---------------------------------------------------------------------------
// Module state
// ---------------------------------------------------------------------------
std::unique_ptr<rl::common::IState> g_state;
std::vector<uint8_t> g_moves;
std::unique_ptr<rl::players::IEvaluator> g_evaluator;
AzSnapshot g_snapshot{};

// The MCTS visit distribution from the last search, so the UI can show what the
// bot was actually thinking rather than only its final choice.
float g_probs[kNActions] = { 0.0f };
float g_evaluation = 0.0f;

// Think time, not a simulation count. Amcts2::search takes both and stops when
// both are satisfied, so passing 0 simulations makes the clock the only limit -
// which is what a bot facing a human wants, and what makes the strength dial
// meaningful across machines whose GPUs differ by an order of magnitude.
int g_think_ms = 3000;
int g_batch = 8; // Amcts2's max_async_simulations: leaves per WebGPU call

// Leaf evaluations in the last search. The session is held separately from the
// evaluator so the counter stays readable; OnnxEvaluator only exposes it as an
// IOnnxSession, which has no counter on it.
#if defined(__EMSCRIPTEN__)
std::shared_ptr<rl::onnxeval::WebOnnxSession> g_session;
#endif
int g_last_evaluations = 0;

// The evaluator is rebuilt whenever the batch changes, because WebOnnxSession
// pads every run to one fixed shape and that shape is fixed at construction.
void rebuild_evaluator()
{
#if defined(__EMSCRIPTEN__)
    auto probe = rl::games::MigoyugoState::initialize();
    g_session = std::make_shared<rl::onnxeval::WebOnnxSession>(
        probe->get_observation_shape(), probe->get_n_actions(), g_batch);
    g_evaluator = std::make_unique<rl::onnxeval::OnnxEvaluator>(
        g_session, probe->get_n_actions(), probe->get_observation_shape());
#endif
}

bool legal_at(int sq)
{
    if (!g_state || sq < 0 || sq >= kNActions) return false;
    const auto mask = g_state->actions_mask();
    return sq < static_cast<int>(mask.size()) && mask.at(sq);
}

void reset_game()
{
    g_state = rl::games::MigoyugoState::initialize();
    g_moves.clear();
    std::memset(g_probs, 0, sizeof(g_probs));
    g_evaluation = 0.0f;
}

// Reconstructs absolute board contents from the observation. MigoyugoState keeps
// its board private, and get_observation() is perspective-relative - channels
// are {our migo, our yugo, their migo, their yugo} for the side to move - so the
// player index has to be folded back in to get absolute colors.
void write_cells(uint8_t* cells)
{
    std::memset(cells, 0, kBoardCells);
    const std::vector<float> obs = g_state->get_observation();
    if (obs.size() < static_cast<size_t>(4 * kBoardCells)) return;

    const int us = g_state->player_turn();
    const int them = 1 - us;
    // White (player 0) uses codes 1/2, Black uses 3/4.
    const uint8_t our_migo = us == 0 ? 1 : 3;
    const uint8_t our_yugo = us == 0 ? 2 : 4;
    const uint8_t their_migo = them == 0 ? 1 : 3;
    const uint8_t their_yugo = them == 0 ? 2 : 4;

    for (int i = 0; i < kBoardCells; i++)
    {
        if (obs[0 * kBoardCells + i] > 0.5f) cells[i] = our_migo;
        else if (obs[1 * kBoardCells + i] > 0.5f) cells[i] = our_yugo;
        else if (obs[2 * kBoardCells + i] > 0.5f) cells[i] = their_migo;
        else if (obs[3 * kBoardCells + i] > 0.5f) cells[i] = their_yugo;
    }
}

void refresh_snapshot()
{
    std::memset(&g_snapshot, 0, sizeof(g_snapshot));
    g_snapshot.version = kSnapshotVersion;
    if (!g_state) return;

    const bool over = g_state->is_terminal();
    g_snapshot.stm = static_cast<uint8_t>(g_state->player_turn());
    g_snapshot.status = over ? kStatusOver : kStatusPlaying;
    g_snapshot.move_count = static_cast<uint8_t>(g_moves.size());
    g_snapshot.last_move = g_moves.empty() ? kNoSquare : g_moves.back();
    g_snapshot.can_undo = g_moves.empty() ? 0 : 1;

    g_snapshot.winner = kNoSquare;
    if (over)
    {
        // get_reward() is relative to the side to move.
        const float reward = g_state->get_reward();
        if (reward > 0.0f) g_snapshot.winner = static_cast<uint8_t>(g_state->player_turn());
        else if (reward < 0.0f) g_snapshot.winner = static_cast<uint8_t>(1 - g_state->player_turn());
    }

    write_cells(g_snapshot.cells);

    if (!over)
    {
        const auto mask = g_state->actions_mask();
        for (int i = 0; i < kBoardCells && i < static_cast<int>(mask.size()); i++)
        {
            g_snapshot.legal[i] = mask.at(i) ? 1 : 0;
        }
    }
}

// The single funnel through which the position ever changes.
int apply_move(int sq)
{
    if (!g_state) return kErrNoModel;
    if (sq < 0 || sq >= kNActions) return kErrRange;
    if (g_state->is_terminal()) return kErrGameOver;
    if (!legal_at(sq)) return kErrIllegal;

    g_state = g_state->step(sq);
    g_moves.push_back(static_cast<uint8_t>(sq));
    return kOk;
}
} // namespace

extern "C"
{

/// @brief Builds the evaluator and the first position.
///
/// There is no model buffer to hand over: onnxruntime-web owns the weights on
/// the JS side, and az_worker.js must have installed Module.azRun before any
/// search runs. This only wires up the C++ half.
EMSCRIPTEN_KEEPALIVE int mgy_az_init(int think_ms, int batch)
{
    if (think_ms > 0) g_think_ms = think_ms;
    if (batch > 0) g_batch = batch;

#if defined(__EMSCRIPTEN__)
    rebuild_evaluator();
#else
    return kErrNoModel; // native builds have no onnxruntime-web to call
#endif

    reset_game();
    refresh_snapshot();
    return kOk;
}

EMSCRIPTEN_KEEPALIVE void mgy_az_new_game()
{
    reset_game();
    refresh_snapshot();
}

EMSCRIPTEN_KEEPALIVE int mgy_az_play(int sq)
{
    const int rc = apply_move(sq);
    if (rc == kOk) refresh_snapshot();
    return rc;
}

EMSCRIPTEN_KEEPALIVE int mgy_az_undo(int plies)
{
    if (plies <= 0) return kErrRange;
    if (g_moves.empty()) return kErrNothingToUndo;

    // Replay from the start rather than unwinding: MigoyugoState is immutable
    // and cheap to step, and a replay cannot desync from the move list.
    const int keep = static_cast<int>(g_moves.size()) > plies
        ? static_cast<int>(g_moves.size()) - plies : 0;
    const std::vector<uint8_t> replay(g_moves.begin(), g_moves.begin() + keep);

    reset_game();
    for (uint8_t sq : replay)
    {
        if (apply_move(sq) != kOk) { reset_game(); break; }
    }
    refresh_snapshot();
    return kOk;
}

/// @brief Restores a position by replaying a validated move list.
///
/// The counterpart of mgy_load_moves in the NNUE module, and the way index.html
/// keeps this module in step: that page's authoritative position lives in the
/// NNUE module, and this one is resynced from its move list before every search.
/// Replaying at most 64 MigoyugoState steps costs microseconds and cannot drift.
///
/// A rejected move here is worth surfacing rather than swallowing: it would mean
/// MigoyugoBB and MigoyugoState disagree about legality.
EMSCRIPTEN_KEEPALIVE int mgy_az_load_moves(const uint8_t* moves, int n)
{
    if (n < 0 || n > kNActions || (n > 0 && !moves)) return kErrRange;
    const std::vector<uint8_t> wanted(moves, moves + n);

    reset_game();
    for (uint8_t sq : wanted)
    {
        const int rc = apply_move(sq);
        if (rc != kOk) { reset_game(); refresh_snapshot(); return rc; }
    }
    refresh_snapshot();
    return kOk;
}

/// @brief Searches WITHOUT playing, and returns the best square.
///
/// Suspends across every WebGPU batch, so JS must reach it through
/// `ccall('mgy_az_bot_suggest', 'number', [], [], {async: true})` - a directly
/// called Asyncify export hands back a meaningless value at the first suspend.
EMSCRIPTEN_KEEPALIVE int mgy_az_bot_suggest()
{
    if (!g_evaluator || !g_state) return kErrNoModel;
    if (g_state->is_terminal()) return kErrGameOver;

#if defined(__EMSCRIPTEN__)
    if (g_session) g_session->reset_evaluations();
#endif

    auto tree = rl::players::Amcts2(kNActions, g_evaluator->copy(), kCpuct, kTemperature,
        g_batch, kDirichletEpsilon, kDirichletAlpha, kDefaultVisits, kDefaultWins);
    // Amcts2's loop runs while EITHER the count or the clock still allows, so the
    // clock is normally in charge - but never with fewer than kMinSimulations.
    //
    // That floor is a correctness guard, not a strength dial. Every child starts
    // at default_visits = 1, so a tree with fewer visits than it has legal moves
    // has near-identical counts everywhere and the argmax below degenerates to
    // "lowest square index" - the engine appears to play a1, b1, ... rather than
    // to play badly. A GPU hitch that eats the clock must not be able to produce
    // that.
    const std::vector<float> probs = tree.search(g_state.get(), kMinSimulations,
        std::chrono::duration<int, std::milli>(g_think_ms));

    std::memset(g_probs, 0, sizeof(g_probs));
    int best = -1;
    for (int a = 0; a < kNActions && a < static_cast<int>(probs.size()); a++)
    {
        g_probs[a] = probs[a];
        if (legal_at(a) && (best < 0 || probs[a] > probs[best])) best = a;
    }
    if (best < 0) return kErrIllegal;

    // get_evaluation() reads the root node, which only exists after a search.
    g_evaluation = tree.get_evaluation();
#if defined(__EMSCRIPTEN__)
    g_last_evaluations = g_session ? static_cast<int>(g_session->evaluations()) : 0;
#endif
    return best;
}

/// @brief Searches and plays. Async, exactly like mgy_az_bot_suggest.
EMSCRIPTEN_KEEPALIVE int mgy_az_bot_move()
{
    const int best = mgy_az_bot_suggest();
    if (best < 0) return best;

    const int rc = apply_move(best);
    if (rc != kOk) return rc;
    refresh_snapshot();
    return best;
}

EMSCRIPTEN_KEEPALIVE void mgy_az_set_time_ms(int think_ms)
{
    if (think_ms > 0) g_think_ms = think_ms;
}

/// @brief Changing the batch rebuilds the session, because the padded shape it
///        compiles WebGPU shaders for is fixed when it is constructed. Expect
///        the next search to pay compilation once.
EMSCRIPTEN_KEEPALIVE void mgy_az_set_batch(int batch)
{
    if (batch <= 0 || batch == g_batch) return;
    g_batch = batch;
    rebuild_evaluator();
}

/// @brief Rewrites the struct from the current position and returns its address.
///        Not a pure getter - call it on every read, like mgy_snapshot.
EMSCRIPTEN_KEEPALIVE const uint8_t* mgy_az_snapshot()
{
    refresh_snapshot();
    return reinterpret_cast<const uint8_t*>(&g_snapshot);
}

EMSCRIPTEN_KEEPALIVE int mgy_az_snapshot_size() { return static_cast<int>(sizeof(AzSnapshot)); }

/// @brief The MCTS visit distribution from the last search, 64 floats.
EMSCRIPTEN_KEEPALIVE const float* mgy_az_probs() { return g_probs; }

/// @brief Root evaluation from the last search, in [-1, 1].
EMSCRIPTEN_KEEPALIVE float mgy_az_evaluation() { return g_evaluation; }

/// @brief Positions the network evaluated during the last search.
///
/// Roughly the simulation count, and the basis for the speed readout. It is not
/// comparable to the NNUE engine's node count - report it as evaluations per
/// second, not nodes per second.
EMSCRIPTEN_KEEPALIVE int mgy_az_last_evaluations() { return g_last_evaluations; }

} // extern "C"
