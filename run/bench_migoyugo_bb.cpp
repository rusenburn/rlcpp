// Correctness and speed harness for the bitboard Migoyugo environment.
//
//   bench_migoyugo_bb diff   [games]           differential test vs MigoyugoLightState
//   bench_migoyugo_bb perft  [depth]           node counts from the empty board, both engines
//   bench_migoyugo_bb speed  [depth]           make/unmake throughput of the bitboard engine
//   bench_migoyugo_bb search [depth] [weights] NNUE search nodes/second
//   bench_migoyugo_bb forced [depth] [weights] forced-move rule claims verified exhaustively
//   bench_migoyugo_bb determinism [depth] [weights] two identical searches must agree exactly
//   bench_migoyugo_bb match [ms] [games]       v1 vs v2 head to head at equal time
//   bench_migoyugo_bb all                      diff 20000, perft 4, speed 5
//
// MigoyugoLightState is the reference implementation of the rules. Nothing
// downstream should trust the bitboard engine until `diff` reports zero
// mismatches.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <games/migoyugo_bb.hpp>
#include <games/migoyugo_light.hpp>
#include <nnue/nnue_layerstacks_model.hpp>
#include <nnue/nnue_layerstacks_player.hpp>
#include <nnue/nnue_layerstacks_model_v2.hpp>
#include <nnue/nnue_layerstacks_player_v2.hpp>

using rl::games::MigoyugoLightState;
using namespace rl::games::mgbb;

namespace
{

std::mt19937_64 rng(0x5eed1234ULL);

// The reference state's board as four bitboards, for a field-by-field compare.
struct RefBoards
{
    uint64_t migo[2]{};
    uint64_t yugo[2]{};
};

RefBoards boards_of(const MigoyugoLightState& s)
{
    RefBoards r;
    // to_short() is the only public view of the board; parsing it back is
    // exactly how the players build their search state anyway.
    const std::string str = s.to_short();
    const size_t space_pos = str.find_last_of(' ');
    int row = 0, col = 0;
    for (size_t i = 0; i < space_pos; ++i)
    {
        const char ch = str[i];
        if (ch == '/') { ++row; col = 0; continue; }
        if (ch >= '0' && ch <= '9') { col += ch - '0'; continue; }
        const uint64_t bit = 1ULL << (row * 8 + col);
        switch (ch)
        {
        case 'x': r.migo[0] |= bit; break;
        case 'X': r.yugo[0] |= bit; break;
        case 'o': r.migo[1] |= bit; break;
        case 'O': r.yugo[1] |= bit; break;
        default: break;
        }
        ++col;
    }
    return r;
}

uint64_t legal_of(const MigoyugoLightState& s)
{
    uint64_t m = 0;
    const std::vector<int> mask = s.actions_mask_2();
    for (int a = 0; a < 64; ++a) if (mask[a]) m |= 1ULL << a;
    return m;
}

void dump(const char* what, const MigoyugoBB& bb, uint64_t got, uint64_t want)
{
    std::printf("  %s mismatch\n    bitboard 0x%016llx\n    reference 0x%016llx\n",
        what, (unsigned long long)got, (unsigned long long)want);
    std::printf("    stm=%d ply=%d wm=0x%llx wy=0x%llx bm=0x%llx by=0x%llx\n",
        bb.stm, bb.ply,
        (unsigned long long)bb.migo[0], (unsigned long long)bb.yugo[0],
        (unsigned long long)bb.migo[1], (unsigned long long)bb.yugo[1]);
}

// Plays random games, stepping both engines in lockstep and comparing
// everything observable after every ply.
int run_diff(int n_games)
{
    int failures = 0;
    long long plies = 0;
    long long promotions = 0;
    long long igos = 0, wegos = 0;
    long long piline_cells = 0;
    long long piline_toggles = 0;
    long long total_toggles = 0;
    int max_toggles = 0;
    int max_cleared = 0;

    for (int g = 0; g < n_games && failures < 10; ++g)
    {
        auto ref = MigoyugoLightState::initialize_state();
        MigoyugoBB bb = MigoyugoBB::initial();

        for (int step = 0; step < 200; ++step)
        {
            // --- board agreement ---
            const RefBoards rb = boards_of(*ref);
            if (rb.migo[0] != bb.migo[0] || rb.migo[1] != bb.migo[1] ||
                rb.yugo[0] != bb.yugo[0] || rb.yugo[1] != bb.yugo[1])
            {
                std::printf("game %d ply %d: board\n", g, step);
                dump("white migo", bb, bb.migo[0], rb.migo[0]);
                dump("white yugo", bb, bb.yugo[0], rb.yugo[0]);
                dump("black migo", bb, bb.migo[1], rb.migo[1]);
                dump("black yugo", bb, bb.yugo[1], rb.yugo[1]);
                ++failures;
                break;
            }
            if (ref->player_turn() != bb.stm)
            {
                std::printf("game %d ply %d: side to move %d vs %d\n",
                    g, step, bb.stm, ref->player_turn());
                ++failures;
                break;
            }

            // --- legality agreement ---
            const uint64_t ref_legal = legal_of(*ref);
            const uint64_t bb_legal = bb.legal_moves();
            if (ref_legal != bb_legal)
            {
                std::printf("game %d ply %d: legal moves\n", g, step);
                dump("legal", bb, bb_legal, ref_legal);
                ++failures;
                break;
            }

            // --- piline is exactly the empty squares that are not legal ---
            const uint64_t expect_piline = bb.empty & ~bb_legal;
            if (bb.piline[bb.stm] != expect_piline)
            {
                std::printf("game %d ply %d: piline[stm]\n", g, step);
                dump("piline", bb, bb.piline[bb.stm], expect_piline);
                ++failures;
                break;
            }
            if ((bb.piline[0] | bb.piline[1]) & ~bb.empty)
            {
                std::printf("game %d ply %d: piline overlaps an occupied square\n", g, step);
                ++failures;
                break;
            }

            // --- terminal agreement ---
            const bool ref_term = ref->is_terminal();
            const bool bb_term = bb.is_terminal_root();
            if (ref_term != bb_term)
            {
                std::printf("game %d ply %d: terminal %d vs %d\n", g, step, (int)bb_term, (int)ref_term);
                ++failures;
                break;
            }
            if (ref_term)
            {
                // Only compare rewards on a Wego; an Igo ends the game one ply
                // earlier in the bitboard engine (do_move reports it directly),
                // which is checked separately below.
                if (!has_line_of_4(bb.yugo[0]) && !has_line_of_4(bb.yugo[1]))
                {
                    ++wegos;
                    const float rr = ref->get_reward();
                    const float br = bb.wego_reward();
                    if (rr != br)
                    {
                        std::printf("game %d ply %d: wego reward %f vs %f\n", g, step, br, rr);
                        ++failures;
                    }
                }
                break;
            }

            // --- integrity of the incremental key ---
            MigoyugoBB fresh = bb;
            fresh.recompute_all();
            if (fresh.key != bb.key)
            {
                std::printf("game %d ply %d: zobrist key drift\n", g, step);
                ++failures;
                break;
            }
            if (fresh.raw_illegal[0] != bb.raw_illegal[0] || fresh.raw_illegal[1] != bb.raw_illegal[1] ||
                fresh.raw_makes4[0] != bb.raw_makes4[0] || fresh.raw_makes4[1] != bb.raw_makes4[1] ||
                fresh.raw_igo[0] != bb.raw_igo[0] || fresh.raw_igo[1] != bb.raw_igo[1])
            {
                std::printf("game %d ply %d: incremental mask drift\n", g, step);
                ++failures;
                break;
            }

            // --- pick a legal move and play it in both ---
            std::vector<int> moves;
            for (uint64_t b = bb_legal; b; b &= b - 1) moves.push_back(ctz64(b));
            const int action = moves[rng() % moves.size()];

            const uint64_t before_yugo = bb.yugo[bb.stm];
            const int mover = bb.stm;

            Undo u;
            FeatureDelta delta;
            const MigoyugoBB parent = bb;
            const bool igo = bb.do_move(action, u, delta);

            piline_cells += popcount64(parent.piline[0]) + popcount64(parent.piline[1]);
            total_toggles += delta.n_added + delta.n_removed;
            if (delta.n_added + delta.n_removed > max_toggles) max_toggles = delta.n_added + delta.n_removed;
            for (int i = 0; i < delta.n_added; ++i) if (delta.added[i] >= 256) ++piline_toggles;
            for (int i = 0; i < delta.n_removed; ++i) if (delta.removed[i] >= 256) ++piline_toggles;

            if (u.made_yugo)
            {
                ++promotions;
                const int nc = popcount64(u.cleared);
                if (nc > max_cleared) max_cleared = nc;
            }
            if (bb.yugo[mover] != (before_yugo | (u.made_yugo ? (1ULL << action) : 0ULL)))
            {
                std::printf("game %d ply %d: yugo board not updated as expected\n", g, step);
                ++failures;
                break;
            }

            // --- undo must restore the state exactly ---
            MigoyugoBB after = bb;
            bb.undo_move(u);
            if (std::memcmp(&bb, &parent, sizeof(MigoyugoBB)) != 0)
            {
                std::printf("game %d ply %d: undo_move did not restore the state\n", g, step);
                ++failures;
                break;
            }
            bb.do_move(action, u, delta);
            if (std::memcmp(&bb, &after, sizeof(MigoyugoBB)) != 0)
            {
                std::printf("game %d ply %d: redo produced a different state\n", g, step);
                ++failures;
                break;
            }

            ref = ref->step_state(action);
            ++plies;

            if (igo)
            {
                ++igos;
                // The mover just made four Yugos in a row, so the reference
                // must now see the position as terminal and lost for the side
                // to move.
                if (!ref->is_terminal() || ref->get_reward() != -1.0f)
                {
                    std::printf("game %d ply %d: igo claimed but reference says terminal=%d reward=%f\n",
                        g, step, (int)ref->is_terminal(), ref->get_reward());
                    ++failures;
                }
                break;
            }
        }
    }

    std::printf("\ndifferential test: %s (%d failures)\n", failures ? "FAILED" : "PASSED", failures);
    std::printf("  games %d, plies %lld\n", n_games, plies);
    std::printf("  promotions %lld (%.1f%% of moves), max migos cleared %d\n",
        promotions, plies ? 100.0 * promotions / plies : 0.0, max_cleared);
    std::printf("  igo endings %lld, wego endings %lld\n", igos, wegos);
    std::printf("  mean piline cells/position %.2f\n", plies ? (double)piline_cells / plies : 0.0);
    std::printf("  mean feature toggles/move %.2f (piline %.2f), max %d\n",
        plies ? (double)total_toggles / plies : 0.0,
        plies ? (double)piline_toggles / plies : 0.0, max_toggles);
    return failures;
}

// --- perft ---

uint64_t perft_bb(MigoyugoBB& s, int depth)
{
    const uint64_t legal = s.legal_moves();
    if (legal == 0) return 1;      // Wego: terminal, one leaf
    if (depth == 0) return 1;

    uint64_t nodes = 0;
    for (uint64_t b = legal; b; b &= b - 1)
    {
        const int sq = ctz64(b);
        Undo u;
        const bool igo = s.do_move(sq, u);
        nodes += igo ? 1 : perft_bb(s, depth - 1);
        s.undo_move(u);
    }
    return nodes;
}

uint64_t perft_ref(const MigoyugoLightState& s, int depth)
{
    if (s.is_terminal()) return 1;
    if (depth == 0) return 1;

    uint64_t nodes = 0;
    const std::vector<int> mask = s.actions_mask_2();
    for (int a = 0; a < 64; ++a)
    {
        if (!mask[a]) continue;
        auto child = s.step_state(a);
        nodes += perft_ref(*child, depth - 1);
    }
    return nodes;
}

int run_perft(int max_depth)
{
    int failures = 0;
    for (int d = 1; d <= max_depth; ++d)
    {
        MigoyugoBB bb = MigoyugoBB::initial();
        auto t0 = std::chrono::high_resolution_clock::now();
        const uint64_t got = perft_bb(bb, d);
        auto t1 = std::chrono::high_resolution_clock::now();
        const double secs = std::chrono::duration<double>(t1 - t0).count();

        auto ref = MigoyugoLightState::initialize_state();
        const uint64_t want = perft_ref(*ref, d);

        const bool ok = got == want;
        if (!ok) ++failures;
        std::printf("perft %d: bitboard %-12llu reference %-12llu %s  (%.3fs, %.0f nps)\n",
            d, (unsigned long long)got, (unsigned long long)want, ok ? "ok" : "MISMATCH",
            secs, secs > 0 ? got / secs : 0.0);
    }
    std::printf("\nperft: %s\n", failures ? "FAILED" : "PASSED");
    return failures;
}

// --- raw make/unmake throughput, with the NNUE feature delta being produced ---

uint64_t speed_walk(MigoyugoBB& s, int depth, FeatureDelta* deltas)
{
    const uint64_t legal = s.legal_moves();
    if (legal == 0 || depth == 0) return 1;

    uint64_t nodes = 0;
    for (uint64_t b = legal; b; b &= b - 1)
    {
        const int sq = ctz64(b);
        Undo u;
        const bool igo = s.do_move(sq, u, deltas[depth]);
        nodes += igo ? 1 : speed_walk(s, depth - 1, deltas);
        s.undo_move(u);
    }
    return nodes;
}

void run_speed(int depth)
{
    std::vector<FeatureDelta> deltas(depth + 1);
    MigoyugoBB bb = MigoyugoBB::initial();

    auto t0 = std::chrono::high_resolution_clock::now();
    const uint64_t nodes = speed_walk(bb, depth, deltas.data());
    auto t1 = std::chrono::high_resolution_clock::now();

    const double secs = std::chrono::duration<double>(t1 - t0).count();
    std::printf("\nmake/unmake + feature delta, depth %d: %llu nodes in %.3fs = %.2f M nodes/s\n",
        depth, (unsigned long long)nodes, secs, secs > 0 ? nodes / secs / 1e6 : 0.0);
}

// --- NNUE search: throughput, and the soundness of the forced-move rule ---

// A spread of positions reached by random play, so the search is measured on
// realistic middlegames rather than only on the empty board.
std::vector<MigoyugoBB> sample_positions(int count, int min_ply, int max_ply)
{
    std::vector<MigoyugoBB> out;
    while (static_cast<int>(out.size()) < count)
    {
        MigoyugoBB s = MigoyugoBB::initial();
        const int target = min_ply + static_cast<int>(rng() % (max_ply - min_ply + 1));
        bool ok = true;
        for (int i = 0; i < target; ++i)
        {
            const uint64_t legal = s.legal_moves();
            if (legal == 0) { ok = false; break; }
            std::vector<int> moves;
            for (uint64_t b = legal; b; b &= b - 1) moves.push_back(ctz64(b));
            Undo u;
            if (s.do_move(moves[rng() % moves.size()], u)) { ok = false; break; } // Igo ends it
        }
        if (ok && s.legal_moves() != 0 && s.winning_moves() == 0) out.push_back(s);
    }
    return out;
}

int run_search(int depth, const std::string& weights)
{
    auto model = load_nnue_layerstacks_v2(weights);
    if (!model) return 1;

    auto player = std::make_unique<rl::players::NNUELayerStacksPlayerV2>(
        model, std::chrono::duration<int, std::milli>(3600000), 64, false);

    const auto positions = sample_positions(24, 8, 40);

    uint64_t total_nodes = 0;
    double total_secs = 0;
    for (const auto& pos : positions)
    {
        player->clear_tt();
        const auto t0 = std::chrono::high_resolution_clock::now();
        player->search_fixed_depth(pos, depth);
        const auto t1 = std::chrono::high_resolution_clock::now();
        total_secs += std::chrono::duration<double>(t1 - t0).count();
        total_nodes += player->nodes();
    }

    std::printf("\nNNUE search, depth %d over %zu positions\n", depth, positions.size());
    std::printf("  %llu nodes in %.3fs = %.0f nodes/s\n",
        (unsigned long long)total_nodes, total_secs,
        total_secs > 0 ? total_nodes / total_secs : 0.0);
    return 0;
}

// The forced-move rule prunes on a game-specific argument, so it needs
// checking - but NOT by comparing search scores at a fixed depth. The rule
// returns proven results ("the opponent has two winning squares, so this is
// lost however deep you look"), which a depth-limited search legitimately
// cannot reproduce; the two disagreeing is the rule working, not failing.
//
// Instead, verify the rule's actual claims exhaustively and one ply deep,
// which is both exact and depth-independent:
//
//   two or more opponent winning squares -> EVERY legal move loses at once
//   exactly one, at square s             -> every legal move except s does
//
// "Loses at once" means: after the move, the opponent has a move that wins
// by Igo, or the game ends by Wego in their favour.
int run_forced(int depth, const std::string& weights)
{
    (void)depth; (void)weights; // this check needs no evaluation at all

    auto positions = sample_positions(4000, 6, 46);
    const auto deep = sample_positions(6000, 46, 90);
    positions.insert(positions.end(), deep.begin(), deep.end());

    int violations = 0;
    int fired_two = 0, fired_one = 0, considered = 0;

    for (size_t i = 0; i < positions.size(); ++i)
    {
        MigoyugoBB s = positions[i];
        const uint64_t legal = s.legal_moves();
        if (legal == 0 || s.winning_moves()) continue; // rule never applies here
        ++considered;

        const int opp = 1 - s.stm;
        const uint64_t threats = s.raw_igo[opp] & s.legal_moves_for(opp);
        if (threats == 0) continue;

        const bool two_or_more = (threats & (threats - 1)) != 0;
        const uint64_t block = threats & legal;
        const uint64_t exempt = (two_or_more || block == 0) ? 0ULL : block;
        if (two_or_more || block == 0) ++fired_two; else ++fired_one;

        // Every move the rule discards must lose on the spot.
        for (uint64_t b = legal & ~exempt; b; b &= b - 1)
        {
            const int sq = ctz64(b);
            Undo u;
            const bool igo = s.do_move(sq, u);

            bool loses_at_once;
            if (igo)
            {
                loses_at_once = false; // it wins outright, so discarding it is wrong
            }
            else if (s.legal_moves() == 0)
            {
                // Wego: the game ends now, with the opponent to move.
                loses_at_once = s.wego_reward() > 0.0f; // good for THEM
            }
            else
            {
                loses_at_once = s.winning_moves() != 0;
            }

            s.undo_move(u);

            if (!loses_at_once)
            {
                if (violations < 8)
                    std::printf("  position %zu: discarded move %d does not lose "
                        "(threats 0x%llx, exempt 0x%llx)\n",
                        i, sq, (unsigned long long)threats, (unsigned long long)exempt);
                ++violations;
                break;
            }
        }
    }

    std::printf("\nforced-move soundness over %zu positions: %s (%d violations)\n",
        positions.size(), violations ? "FAILED" : "PASSED", violations);
    std::printf("  positions where the rule applies: %d of %d considered "
        "(%d lost outright, %d reduced to a single move)\n",
        fired_two + fired_one, considered, fired_two, fired_one);
    return violations ? 1 : 0;
}

// Two identically configured searches must return identical scores and visit
// identical node counts; anything else means hidden state is leaking between
// searches and every other comparison in this file is meaningless.
int run_determinism(int depth, const std::string& weights)
{
    auto model = load_nnue_layerstacks_v2(weights);
    if (!model) return 1;

    auto a = std::make_unique<rl::players::NNUELayerStacksPlayerV2>(
        model, std::chrono::duration<int, std::milli>(3600000), 32, false);
    auto b = std::make_unique<rl::players::NNUELayerStacksPlayerV2>(
        model, std::chrono::duration<int, std::milli>(3600000), 32, false);

    const auto positions = sample_positions(200, 6, 60);
    int mismatches = 0;
    for (size_t i = 0; i < positions.size(); ++i)
    {
        a->clear_tt();
        b->clear_tt();
        const int sa = a->search_fixed_depth(positions[i], depth);
        const uint64_t na = a->nodes();
        const int sb = b->search_fixed_depth(positions[i], depth);
        const uint64_t nb = b->nodes();
        if (sa != sb || na != nb)
        {
            if (mismatches < 5)
                std::printf("  position %zu: %d/%llu vs %d/%llu\n", i, sa,
                    (unsigned long long)na, sb, (unsigned long long)nb);
            ++mismatches;
        }
    }
    std::printf("\ndeterminism at depth %d over %zu positions: %s (%d mismatches)\n",
        depth, positions.size(), mismatches ? "FAILED" : "PASSED", mismatches);
    return mismatches ? 1 : 0;
}


// Head to head between the old layer-stacks player and the new one, at equal
// time per move, colours alternating, from randomised short openings so the
// games differ. MigoyugoLightState drives the game because it is the reference
// implementation of the rules - neither engine gets to arbitrate its own game.
int run_match(int ms_per_move, int n_games, const std::string& v1_path, const std::string& v2_path)
{
    auto v2_model = load_nnue_layerstacks_v2(v2_path);
    if (!v2_model) return 1;

    auto v1_model = std::make_unique<NNUELayerStacksModel>();
    {
        FILE* f = std::fopen(v1_path.c_str(), "rb");
        if (!f) { std::printf("cannot open %s\n", v1_path.c_str()); return 1; }
        const size_t got = std::fread(v1_model.get(), sizeof(NNUELayerStacksModel), 1, f);
        std::fclose(f);
        if (got != 1) { std::printf("%s is truncated\n", v1_path.c_str()); return 1; }
    }

    const std::chrono::duration<int, std::milli> budget(ms_per_move);
    rl::players::NNUELayerStacksPlayer v1(*v1_model, budget);
    rl::players::NNUELayerStacksPlayerV2 v2(v2_model, budget, 64, false);

    int v2_wins = 0, v1_wins = 0, draws = 0;
    uint64_t v2_nodes = 0;
    double v2_secs = 0;

    for (int g = 0; g < n_games; ++g)
    {
        const int v2_seat = g % 2; // alternate who moves first

        std::unique_ptr<MigoyugoLightState> state;
        while (true) // random opening, retried if it ends the game
        {
            state = MigoyugoLightState::initialize_state();
            const int depth = static_cast<int>(rng() % 5);
            bool ok = true;
            for (int i = 0; i < depth; ++i)
            {
                const std::vector<int> mask = state->actions_mask_2();
                std::vector<int> moves;
                for (int a = 0; a < 64; ++a) if (mask[a]) moves.push_back(a);
                if (moves.empty()) { ok = false; break; }
                state = state->step_state(moves[rng() % moves.size()]);
                if (state->is_terminal()) { ok = false; break; }
            }
            if (ok) break;
        }

        int plies = 0;
        while (!state->is_terminal() && plies < 200)
        {
            std::unique_ptr<rl::common::IState> as_istate = state->clone();
            int action;
            if (state->player_turn() == v2_seat)
            {
                const auto t0 = std::chrono::high_resolution_clock::now();
                action = v2.choose_action(as_istate);
                v2_secs += std::chrono::duration<double>(
                    std::chrono::high_resolution_clock::now() - t0).count();
                v2_nodes += v2.nodes();
            }
            else
            {
                action = v1.choose_action(as_istate);
            }
            if (action < 0) break;
            state = state->step_state(action);
            ++plies;
        }

        // get_reward() is from the point of view of whoever is to move in the
        // terminal position. That is the player who just got mated by an Igo,
        // or the one with no legal move left - who can still WIN a Wego on Yugo
        // count, so this is a perspective, not a verdict.
        const float reward = state->is_terminal() ? state->get_reward() : 0.0f;
        const int final_seat = state->player_turn();
        const float v2_result = (reward == 0.0f) ? 0.0f
            : ((final_seat == v2_seat) ? reward : -reward);

        if (v2_result > 0) ++v2_wins; else if (v2_result < 0) ++v1_wins; else ++draws;

        std::printf("[match] game %d/%d  v2 %s  (running v2 %d - v1 %d - draw %d)\n",
            g + 1, n_games, v2_result > 0 ? "win " : (v2_result < 0 ? "loss" : "draw"),
            v2_wins, v1_wins, draws);
        std::fflush(stdout);
    }

    const double score = (v2_wins + 0.5 * draws) / n_games;
    std::printf("\nmatch at %d ms/move over %d games: v2 %d wins, %d losses, %d draws (%.1f%%)\n",
        ms_per_move, n_games, v2_wins, v1_wins, draws, 100.0 * score);
    std::printf("  v2 in-game throughput: %.0f nodes/s over %llu nodes\n",
        v2_secs > 0 ? v2_nodes / v2_secs : 0.0, (unsigned long long)v2_nodes);
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    const std::string mode = argc > 1 ? argv[1] : "all";
    const int arg = argc > 2 ? std::atoi(argv[2]) : 0;
    const std::string weights = argc > 3 ? argv[3] : "../checkpoints/nnue_layerstacks_v2_weights.bin";

    int failures = 0;
    if (mode == "diff") failures += run_diff(arg ? arg : 20000);
    else if (mode == "perft") failures += run_perft(arg ? arg : 4);
    else if (mode == "speed") run_speed(arg ? arg : 5);
    else if (mode == "search") failures += run_search(arg ? arg : 6, weights);
    else if (mode == "forced") failures += run_forced(arg ? arg : 5, weights);
    else if (mode == "determinism") failures += run_determinism(arg ? arg : 5, weights);
    else if (mode == "match")
    {
        const int games = argc > 3 ? std::atoi(argv[3]) : 40;
        const std::string v1 = argc > 4 ? argv[4] : "../checkpoints/nnue_layerstacks_weights.bin";
        const std::string v2 = argc > 5 ? argv[5] : "../checkpoints/nnue_layerstacks_v2_weights.bin";
        failures += run_match(arg ? arg : 1000, games, v1, v2);
    }
    else
    {
        failures += run_diff(20000);
        failures += run_perft(4);
        run_speed(5);
    }
    return failures ? 1 : 0;
}
