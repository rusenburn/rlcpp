// Exercises the WebAssembly C ABI natively, so its semantics are tested in a
// real debugger before a browser is ever involved.
//
// It links wasm/migoyugo_wasm.cpp directly - the same translation unit emcc
// compiles - so anything this proves about legality validation, undo/replay
// and the snapshot layout is true of the shipped module.
//
//   test_wasm_api [weights.bin]

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <games/migoyugo_bb.hpp>

// The C ABI under test.
extern "C" {
int mgy_init(const uint8_t* model_data, int model_len, int tt_mb);
void mgy_new_game();
int mgy_play(int sq);
int mgy_undo(int plies);
int mgy_load_moves(const uint8_t* moves, int n);
int mgy_bot_move();
int mgy_bot_suggest();
void mgy_set_time_ms(int ms);
void mgy_set_tt_mb(int mb);
void mgy_clear_tt();
const uint8_t* mgy_snapshot();
int mgy_snapshot_size();
const uint8_t* mgy_info();
int mgy_info_size();
}

namespace
{

// Snapshot field offsets, transcribed from wasm/web/snapshot.js so that a
// three-way disagreement between C++, this test and the JS is impossible to
// miss.
constexpr int O_VERSION = 0, O_STM = 1, O_STATUS = 2, O_WINNER = 3;
constexpr int O_YUGO = 4, O_MIGO = 6, O_MOVE_COUNT = 8, O_LAST_MOVE = 9;
constexpr int O_LAST_PROMO = 10, O_N_CLEARED = 11, O_LEGAL_COUNT = 12, O_CAN_UNDO = 13;
constexpr int O_CLEARED = 16, O_WIN_LINE = 28;
constexpr int O_CELLS = 32, O_LEGAL = 96, O_PROMOTING = 160, O_WINNING = 224;
constexpr int O_PILINE_W = 288, O_PILINE_B = 352;
constexpr int SNAPSHOT_SIZE = 416;

constexpr uint8_t STATUS_PLAYING = 0, STATUS_IGO = 1, STATUS_WEGO = 2;

int failures = 0;

void check(bool ok, const char* what)
{
    if (!ok) { std::printf("  FAIL: %s\n", what); ++failures; }
}

std::mt19937_64 rng(0xc0ffee1234ULL);

std::vector<uint8_t> snap()
{
    const uint8_t* p = mgy_snapshot();
    return std::vector<uint8_t>(p, p + SNAPSHOT_SIZE);
}

// --- 1. layout -------------------------------------------------------------

void test_layout()
{
    std::printf("layout\n");
    check(mgy_snapshot_size() == SNAPSHOT_SIZE, "snapshot is 416 bytes");
    check(mgy_info_size() == 32, "info is 32 bytes");

    mgy_new_game();
    const auto s = snap();
    check(s[O_VERSION] == 1, "version byte is 1");
    check(s[O_STM] == 0, "White moves first");
    check(s[O_STATUS] == STATUS_PLAYING, "a new game is not over");
    check(s[O_MOVE_COUNT] == 0, "no moves played");
    check(s[O_CAN_UNDO] == 0, "nothing to undo");
    check(s[O_LEGAL_COUNT] == 64, "all 64 squares are legal on an empty board");
    for (int i = 0; i < 64; ++i) check(s[O_CELLS + i] == 0, "board starts empty");
}

// --- 2. the validating funnel ---------------------------------------------

void test_rejects_illegal()
{
    std::printf("illegal move rejection\n");
    mgy_new_game();

    check(mgy_play(-1) < 0, "negative square rejected");
    check(mgy_play(64) < 0, "square 64 rejected");
    check(mgy_play(9999) < 0, "wildly out of range rejected");

    check(mgy_play(27) == 0, "a legal move is accepted");
    check(mgy_play(27) < 0, "playing an occupied square is rejected");

    // Nothing above should have altered the position.
    const auto s = snap();
    check(s[O_MOVE_COUNT] == 1, "rejected moves did not change the move count");
    check(s[O_CELLS + 27] == 1, "the one legal move is on the board");

    // Every square the snapshot marks illegal must actually be refused, and
    // every square it marks legal must be accepted.
    mgy_new_game();
    for (int i = 0; i < 12; ++i)
    {
        const auto cur = snap();
        if (cur[O_STATUS] != STATUS_PLAYING) break;
        std::vector<int> legal;
        for (int sq = 0; sq < 64; ++sq)
        {
            if (cur[O_LEGAL + sq]) legal.push_back(sq);
            else check(mgy_play(sq) < 0, "square marked illegal is refused");
        }
        check(!legal.empty(), "a live position has at least one legal move");
        check(mgy_play(legal[rng() % legal.size()]) == 0, "square marked legal is accepted");
    }
}

// piline squares are empty squares that are illegal for that player: exactly
// the invariant the training data and the UI overlay both depend on.
void test_piline_consistency()
{
    std::printf("piline consistency\n");
    mgy_new_game();
    int checked = 0;
    for (int i = 0; i < 400; ++i)
    {
        const auto s = snap();
        if (s[O_STATUS] != STATUS_PLAYING) { mgy_new_game(); continue; }

        const int stm = s[O_STM];
        const int piline_off = stm == 0 ? O_PILINE_W : O_PILINE_B;
        for (int sq = 0; sq < 64; ++sq)
        {
            const bool empty = s[O_CELLS + sq] == 0;
            const bool legal = s[O_LEGAL + sq] != 0;
            const bool piline = s[piline_off + sq] != 0;
            check(!(piline && !empty), "a piline square is empty");
            if (empty) check(piline == !legal, "piline is exactly empty-and-not-legal");
            ++checked;
        }

        std::vector<int> legal;
        for (int sq = 0; sq < 64; ++sq) if (s[O_LEGAL + sq]) legal.push_back(sq);
        if (legal.empty()) { mgy_new_game(); continue; }
        mgy_play(legal[rng() % legal.size()]);
    }
    std::printf("  %d square-checks\n", checked);
}

// --- 3. undo and load_moves ------------------------------------------------

void test_undo_and_replay()
{
    std::printf("undo / load_moves round trip\n");
    mgy_new_game();

    std::vector<uint8_t> played;
    std::vector<std::vector<uint8_t>> history;

    for (int i = 0; i < 24; ++i)
    {
        history.push_back(snap());
        const auto& s = history.back();
        if (s[O_STATUS] != STATUS_PLAYING) { history.pop_back(); break; }
        std::vector<int> legal;
        for (int sq = 0; sq < 64; ++sq) if (s[O_LEGAL + sq]) legal.push_back(sq);
        if (legal.empty()) { history.pop_back(); break; }
        const int mv = legal[rng() % legal.size()];
        check(mgy_play(mv) == 0, "move accepted");
        played.push_back(static_cast<uint8_t>(mv));
    }

    // Undoing back to each earlier position must reproduce that position byte
    // for byte, apart from the last-move bookkeeping the snapshot carries.
    for (int back = 1; back <= static_cast<int>(played.size()); ++back)
    {
        // Rebuild, then undo, so each check starts from the full game.
        check(mgy_load_moves(played.data(), static_cast<int>(played.size())) == 0, "load_moves accepted");
        check(mgy_undo(back) == 0, "undo accepted");

        const auto got = snap();
        const auto& want = history[played.size() - back];
        bool same = true;
        for (int i = O_CELLS; i < SNAPSHOT_SIZE; ++i) if (got[i] != want[i]) same = false;
        check(same, "undo reproduces the earlier position exactly");
        check(got[O_MOVE_COUNT] == want[O_MOVE_COUNT], "move count matches after undo");
        check(got[O_STM] == want[O_STM], "side to move matches after undo");
    }

    check(mgy_load_moves(played.data(), static_cast<int>(played.size())) == 0, "reload");
    const auto full = snap();
    check(full[O_MOVE_COUNT] == played.size(), "load_moves replays every move");

    mgy_new_game();
    check(mgy_undo(1) < 0, "undo on an empty game is refused");

    // A corrupt move list must be refused wholesale, leaving a clean position.
    std::vector<uint8_t> bad = played;
    if (!bad.empty()) bad[bad.size() / 2] = 200; // out of range
    check(mgy_load_moves(bad.data(), static_cast<int>(bad.size())) < 0, "corrupt move list refused");
    const auto after_bad = snap();
    check(after_bad[O_MOVE_COUNT] == 0, "a refused load leaves a fresh game");
}

// --- 4. full games, terminal reporting ------------------------------------

void test_full_games()
{
    std::printf("random full games\n");
    int igos = 0, wegos = 0, draws = 0;

    for (int g = 0; g < 300; ++g)
    {
        mgy_new_game();
        for (int ply = 0; ply < 200; ++ply)
        {
            const auto s = snap();
            if (s[O_STATUS] != STATUS_PLAYING) break;
            std::vector<int> legal;
            for (int sq = 0; sq < 64; ++sq) if (s[O_LEGAL + sq]) legal.push_back(sq);
            check(!legal.empty(), "a playing position has a legal move");
            check(s[O_LEGAL_COUNT] == legal.size(), "legal_count matches the legal mask");
            if (legal.empty()) break;
            check(mgy_play(legal[rng() % legal.size()]) == 0, "move accepted");
        }

        const auto s = snap();
        check(s[O_STATUS] != STATUS_PLAYING, "the game ended");
        if (s[O_STATUS] == STATUS_IGO)
        {
            ++igos;
            check(s[O_WINNER] == 0 || s[O_WINNER] == 1, "an Igo has a winner");
            int line = 0;
            for (int i = 0; i < 4; ++i) if (s[O_WIN_LINE + i] != 255) ++line;
            check(line == 4, "an Igo reports four winning squares");
            const uint8_t want = s[O_WINNER] == 0 ? 2 : 4; // that colour's Yugo code
            for (int i = 0; i < 4; ++i)
                if (s[O_WIN_LINE + i] != 255)
                    check(s[O_CELLS + s[O_WIN_LINE + i]] == want, "each winning square holds the winner's Yugo");
            check(mgy_play(0) < 0, "no move is accepted after the game ends");
        }
        else
        {
            ++wegos;
            check(s[O_LEGAL_COUNT] == 0, "a Wego means no legal move");
            const int a = s[O_YUGO], b = s[O_YUGO + 1];
            if (a == b) { ++draws; check(s[O_WINNER] == 255, "equal Yugos is a draw"); }
            else check(s[O_WINNER] == (a > b ? 0 : 1), "more Yugos wins the Wego");
        }
    }
    std::printf("  %d Igo, %d Wego (%d drawn)\n", igos, wegos, draws);
}

// A promotion must report the Migos it cleared, for the capture animation.
void test_promotion_reporting()
{
    std::printf("promotion reporting\n");
    int promotions = 0, with_clears = 0, max_cleared = 0;

    for (int g = 0; g < 200 && promotions < 400; ++g)
    {
        mgy_new_game();
        for (int ply = 0; ply < 200; ++ply)
        {
            const auto before = snap();
            if (before[O_STATUS] != STATUS_PLAYING) break;
            std::vector<int> legal;
            for (int sq = 0; sq < 64; ++sq) if (before[O_LEGAL + sq]) legal.push_back(sq);
            if (legal.empty()) break;
            const int mv = legal[rng() % legal.size()];
            const bool expect_promo = before[O_PROMOTING + mv] != 0;
            const int mover = before[O_STM];
            if (mgy_play(mv) != 0) break;

            const auto after = snap();
            check((after[O_LAST_PROMO] != 0) == expect_promo,
                "the promoting mask predicted the promotion");
            check(after[O_LAST_MOVE] == mv, "last_move is the move just played");

            if (after[O_LAST_PROMO])
            {
                ++promotions;
                const uint8_t yugo_code = mover == 0 ? 2 : 4;
                check(after[O_CELLS + mv] == yugo_code, "the placed piece became a Yugo");
                const int n = after[O_N_CLEARED];
                if (n) ++with_clears;
                if (n > max_cleared) max_cleared = n;
                for (int i = 0; i < n; ++i)
                {
                    const int sq = after[O_CLEARED + i];
                    check(sq >= 0 && sq < 64, "a cleared square is on the board");
                    check(after[O_CELLS + sq] == 0, "a cleared square is now empty");
                    const uint8_t migo_code = mover == 0 ? 1 : 3;
                    check(before[O_CELLS + sq] == migo_code, "a cleared square held the mover's Migo");
                }
            }
            else
            {
                check(after[O_N_CLEARED] == 0, "a non-promotion clears nothing");
            }
        }
    }
    std::printf("  %d promotions, %d cleared Migos, max %d in one move\n",
        promotions, with_clears, max_cleared);
}

// --- 5. the engine ---------------------------------------------------------

void test_engine(const std::string& weights)
{
    std::printf("engine\n");

    std::vector<uint8_t> bytes;
    if (FILE* f = std::fopen(weights.c_str(), "rb"))
    {
        std::fseek(f, 0, SEEK_END);
        bytes.resize(static_cast<size_t>(std::ftell(f)));
        std::fseek(f, 0, SEEK_SET);
        if (std::fread(bytes.data(), 1, bytes.size(), f) != bytes.size()) bytes.clear();
        std::fclose(f);
    }
    if (bytes.empty())
    {
        std::printf("  SKIPPED: could not read %s\n", weights.c_str());
        return;
    }

    // Garbage in must be refused, not crash.
    check(mgy_init(nullptr, 0, 16) < 0, "a null model is refused");
    std::vector<uint8_t> truncated(bytes.begin(), bytes.begin() + 100);
    check(mgy_init(truncated.data(), static_cast<int>(truncated.size()), 16) < 0, "a truncated model is refused");
    std::vector<uint8_t> wrong_magic = bytes;
    wrong_magic[0] ^= 0xff;
    check(mgy_init(wrong_magic.data(), static_cast<int>(wrong_magic.size()), 16) < 0, "a bad magic is refused");

    check(mgy_init(bytes.data(), static_cast<int>(bytes.size()), 16) == 0, "the real model loads");

    mgy_set_time_ms(120);
    mgy_new_game();

    // The bot must only ever play a move the snapshot marks legal, and must
    // reach a terminal position without ever being refused.
    int plies = 0;
    for (; plies < 200; ++plies)
    {
        const auto s = snap();
        if (s[O_STATUS] != STATUS_PLAYING) break;
        const int sq = mgy_bot_move();
        check(sq >= 0 && sq < 64, "the bot returned a square");
        if (sq < 0) break;
        check(s[O_LEGAL + sq] != 0, "the bot played a legal move");
    }
    const auto end = snap();
    check(end[O_STATUS] != STATUS_PLAYING, "a bot-vs-bot game reaches a terminal position");
    std::printf("  self-play game ended after %d plies, status %d, winner %d\n",
        plies, end[O_STATUS], end[O_WINNER]);

    check(mgy_bot_move() < 0, "the bot refuses to move in a finished game");

    // suggest must not change the position.
    mgy_new_game();
    mgy_play(27);
    const auto before = snap();
    const int hint = mgy_bot_suggest();
    const auto after = snap();
    check(hint >= 0 && hint < 64, "suggest returned a square");
    check(before[O_LEGAL + hint] != 0, "suggest returned a legal square");
    check(std::memcmp(before.data(), after.data(), SNAPSHOT_SIZE) == 0, "suggest did not change the position");

    // The info block must be populated and self-consistent.
    const uint8_t* ip = mgy_info();
    double nodes, nps;
    int32_t depth, score, best, elapsed;
    std::memcpy(&nodes, ip + 0, 8);
    std::memcpy(&nps, ip + 8, 8);
    std::memcpy(&depth, ip + 16, 4);
    std::memcpy(&score, ip + 20, 4);
    std::memcpy(&best, ip + 24, 4);
    std::memcpy(&elapsed, ip + 28, 4);
    check(nodes > 0, "info reports nodes");
    check(depth > 0, "info reports a depth");
    check(best == hint, "info's best move matches the returned move");
    std::printf("  info: depth %d, score %.3f, nodes %.0f, nps %.0f, %d ms\n",
        depth, score / 1024.0, nodes, nps, elapsed);

    // Changing the budget must not disturb the table, and resizing must not crash.
    mgy_set_time_ms(30);
    check(mgy_bot_suggest() >= 0, "search works after a budget change");
    mgy_set_tt_mb(4);
    check(mgy_bot_suggest() >= 0, "search works after a TT resize");
    mgy_clear_tt();
    check(mgy_bot_suggest() >= 0, "search works after clearing the TT");
}

} // namespace

int main(int argc, char** argv)
{
    const std::string weights = argc > 1 ? argv[1] : "../checkpoints/nnue_layerstacks_v2_weights.bin";

    test_layout();
    test_rejects_illegal();
    test_piline_consistency();
    test_undo_and_replay();
    test_full_games();
    test_promotion_reporting();
    test_engine(weights);

    std::printf("\nwasm API test: %s (%d failures)\n", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}
