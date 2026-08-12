// Correctness, throughput and strength harness for MigoyugoGravePlayer.
//
//   bench_migoyugo_grave selftest   [positions] [weights]  legality and tactics
//   bench_migoyugo_grave invariants [positions] [weights]  tree/back-up bookkeeping
//   bench_migoyugo_grave speed      [ms]        [weights]  sims/s, all configurations
//   bench_migoyugo_grave match      [ms] [games] [weights] configurations head to head
//   bench_migoyugo_grave leafsweep  [ms] [games] [weights] tune the leaf-blend weight
//   bench_migoyugo_grave vsgplayer  [ms] [games]           new bot vs the old GPlayer
//   bench_migoyugo_grave all        [ms]        [weights]  selftest + invariants + speed
//
// Deliberately Torch-free, like bench_migoyugo_bb: this is the gate the new
// search has to pass, and it should stay quick to build and quick to run.
//
// `vsgplayer` is a separate mode because GPlayer prints a line per move from
// inside players/src/bandits/grave/g.cpp and there is no way to quieten it
// without editing that file.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <games/migoyugo_bb.hpp>
#include <games/migoyugo_light.hpp>
#include <nnue/migoyugo_grave_player.hpp>
#include <nnue/nnue_layerstacks_model_v2.hpp>
#include <players/g_player.hpp>

using rl::games::MigoyugoLightState;
using rl::players::MigoyugoGravePlayer;
using namespace rl::games::mgbb;

namespace
{

using Milliseconds = std::chrono::duration<int, std::milli>;

std::mt19937_64 rng(0x9a7beef5ULL);

// ---------------------------------------------------------------- helpers ---

// The inverse of MigoyugoBB::from_short, which is the only bridge to the
// IState world that GPlayer lives in.
std::string bb_to_short(const MigoyugoBB& board)
{
    std::string out;
    for (int row = 0; row < 8; ++row)
    {
        int empty_run = 0;
        for (int col = 0; col < 8; ++col)
        {
            const uint64_t bit = 1ULL << (row * 8 + col);
            char piece = 0;
            if (board.migo[0] & bit) piece = 'x';
            else if (board.yugo[0] & bit) piece = 'X';
            else if (board.migo[1] & bit) piece = 'o';
            else if (board.yugo[1] & bit) piece = 'O';

            if (!piece) { ++empty_run; continue; }
            if (empty_run) { out += std::to_string(empty_run); empty_run = 0; }
            out += piece;
        }
        if (empty_run) out += std::to_string(empty_run);
        if (row < 7) out += '/';
    }
    out += ' ';
    out += std::to_string(board.stm);
    return out;
}

// A spread of positions reached by random play, so everything is measured on
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
        if (ok && s.legal_moves() != 0) out.push_back(s);
    }
    return out;
}

// Random play almost never produces four-Yugos-in-a-row threats - Yugos are
// rare and the game usually ends first - so the two tactical assertions in the
// selftest would run over zero cases and prove nothing. These positions are
// built directly instead: three Yugos on an axis with the fourth square empty,
// then a few Migos sprinkled on.
//
// Everything is gated by the environment's own predicates afterwards, so a
// position that could not arise from legal play (a run of four or more, which
// migoyugo_bb.hpp relies on being impossible) is discarded rather than tested.
std::vector<MigoyugoBB> sample_igo_threat_positions(int count, bool threatener_to_move)
{
    static const int STEPS[4] = { 1, 8, 9, 7 };
    static const int COLUMN_DELTA[4] = { 1, 0, 1, -1 };

    std::vector<MigoyugoBB> out;
    for (int attempt = 0; static_cast<int>(out.size()) < count && attempt < count * 4000; ++attempt)
    {
        MigoyugoBB board = MigoyugoBB::initial();
        board.migo[0] = board.migo[1] = board.yugo[0] = board.yugo[1] = 0;

        const int colour = static_cast<int>(rng() % 2);
        const int axis = static_cast<int>(rng() % 4);
        const int step = STEPS[axis];
        const int column_delta = COLUMN_DELTA[axis];

        // Four squares along the axis, rejecting any that would wrap a file.
        int squares[4];
        int square = static_cast<int>(rng() % 64);
        bool on_board = true;
        for (int i = 0; i < 4; ++i)
        {
            if (i)
            {
                const int previous_column = square % 8;
                square += step;
                if (square < 0 || square >= 64) { on_board = false; break; }
                if (square % 8 != previous_column + column_delta) { on_board = false; break; }
            }
            squares[i] = square;
        }
        if (!on_board) continue;

        const int hole = static_cast<int>(rng() % 4);
        for (int i = 0; i < 4; ++i)
            if (i != hole) board.yugo[colour] |= 1ULL << squares[i];

        board.stm = threatener_to_move ? colour : 1 - colour;
        board.recompute_all();

        // A few Migos, placed only where the environment says a placement is
        // legal and does not complete a run of four.
        const int migo_count = static_cast<int>(rng() % 8);
        for (int i = 0; i < migo_count; ++i)
        {
            const int migo_colour = static_cast<int>(rng() % 2);
            const uint64_t safe = board.legal_moves_for(migo_colour) & ~board.raw_makes4[migo_colour];
            if (!safe) continue;
            const int bits = popcount64(safe);
            uint64_t pick = safe;
            for (int skip = static_cast<int>(rng() % bits); skip > 0; --skip) pick &= pick - 1;
            board.migo[migo_colour] |= pick & (~pick + 1);
            board.recompute_all();
        }

        // Gates: the position must look like one legal play could reach.
        if (has_line_of_4(board.occupancy(0)) || has_line_of_4(board.occupancy(1))) continue;
        if (board.legal_moves() == 0) continue;

        // ...and it must still contain the threat we built it for.
        const uint64_t threat = board.raw_igo[colour] & board.legal_moves_for(colour);
        if (!threat) continue;

        out.push_back(board);
    }
    return out;
}

std::shared_ptr<const NNUELayerStacksModelV2> load_model(const std::string& weights)
{
    auto model = load_nnue_layerstacks_v2(weights);
    if (!model)
        std::printf("  (no network at %s - NNUE configurations skipped)\n", weights.c_str());
    return model;
}

// The four configurations worth telling apart.
enum class Config { even_uniform, even_tactical, leaf_blend, action_value };

const char* config_name(Config c)
{
    switch (c)
    {
    case Config::even_uniform:  return "even-game, uniform rollouts";
    case Config::even_tactical: return "even-game, tactical rollouts";
    case Config::leaf_blend:    return "leaf-value blend w=0.5";
    case Config::action_value:  return "NNUE action-value priors";
    }
    return "?";
}

bool config_needs_network(Config c)
{
    return c == Config::leaf_blend || c == Config::action_value;
}

const Config ALL_CONFIGS[] = { Config::even_uniform, Config::even_tactical,
    Config::leaf_blend, Config::action_value };

std::unique_ptr<MigoyugoGravePlayer> make_player(Config config, Milliseconds budget,
    std::shared_ptr<const NNUELayerStacksModelV2> model)
{
    auto player = std::make_unique<MigoyugoGravePlayer>(budget, 2, std::move(model));
    player->set_verbose(false);

    switch (config)
    {
    case Config::even_uniform:
        player->set_rollout_policy(MigoyugoGravePlayer::RolloutPolicy::uniform_random);
        break;
    case Config::even_tactical:
        break;
    case Config::leaf_blend:
        player->set_leaf_value_weight(0.5f);
        break;
    case Config::action_value:
        player->set_heuristic_mode(MigoyugoGravePlayer::HeuristicMode::nnue_action_value);
        break;
    }
    return player;
}

// ---------------------------------------------------------------- selftest ---

// Three ground truths taken straight from the environment, so a descent or a
// back-up that has gone off the rails shows up as a wrong move rather than as
// a plausible-looking but quietly weaker bot.
int run_selftest(int positions, const std::string& weights)
{
    auto model = load_model(weights);

    std::printf("\nselftest over %d positions per configuration\n", positions);

    int failures = 0;
    for (Config config : ALL_CONFIGS)
    {
        if (config_needs_network(config) && !model) continue;

        auto player = make_player(config, Milliseconds(0), model);
        player->set_fixed_simulations(4000);
        player->set_random_seed(0x1234abcdULL);

        int illegal = 0, missed_win = 0, missed_block = 0;
        int win_cases = 0, block_cases = 0;

        std::vector<MigoyugoBB> cases = sample_positions(positions, 6, 45);
        for (const auto& p : sample_igo_threat_positions(positions / 2, true)) cases.push_back(p);
        for (const auto& p : sample_igo_threat_positions(positions / 2, false)) cases.push_back(p);

        for (const auto& base : cases)
        {
            const int move = player->search_position(base);

            if (move < 0 || ((base.legal_moves() >> move) & 1ULL) == 0) { ++illegal; continue; }

            // 1. An Igo available right now must be played.
            const uint64_t winning = base.winning_moves();
            if (winning)
            {
                ++win_cases;
                if (((winning >> move) & 1ULL) == 0) ++missed_win;
                continue;
            }

            // 2. If the opponent has exactly one winning square and we can
            //    occupy it, that is the only move that does not lose at once.
            //    A move never changes the opponent's own no-long-lines mask,
            //    so their winning square stays winning unless we take it.
            const int opponent = 1 - base.stm;
            const uint64_t threats = base.raw_igo[opponent] & base.legal_moves_for(opponent);
            if (threats && (threats & (threats - 1)) == 0)
            {
                const uint64_t block = threats & base.legal_moves();
                if (block)
                {
                    ++block_cases;
                    if (((block >> move) & 1ULL) == 0) ++missed_block;
                }
            }
        }

        // Uniform rollouts are not required to find the block. The punishment
        // for not blocking is the opponent taking the square in the playout,
        // and a uniform playout usually walks past it - that is the whole
        // reason the tactical policy is the default, not a defect in the tree.
        const bool blocks_required = (config != Config::even_uniform);

        std::printf("  %-30s illegal %d | forced wins %d/%d | forced blocks %d/%d%s\n",
            config_name(config), illegal,
            win_cases - missed_win, win_cases,
            block_cases - missed_block, block_cases,
            blocks_required ? "" : "  (informational)");

        failures += illegal + missed_win + (blocks_required ? missed_block : 0);
    }

    std::printf("%s\n", failures ? "  FAILED" : "  ok");
    return failures ? 1 : 0;
}

// -------------------------------------------------------------- invariants ---

int run_invariants(int positions, const std::string& weights)
{
    auto model = load_model(weights);

    std::printf("\ninvariants over %d positions per configuration\n", positions);

    int failures = 0;
    for (Config config : ALL_CONFIGS)
    {
        if (config_needs_network(config) && !model) continue;

        auto player = make_player(config, Milliseconds(0), model);
        player->set_fixed_simulations(3000);
        player->set_check_invariants(true);
        player->set_random_seed(0xfeed5678ULL);

        int ordering = 0, audits = 0;
        std::string first_error;

        for (const auto& base : sample_positions(positions, 6, 45))
        {
            player->search_position(base);
            ordering += player->invariant_violations();

            std::string error;
            if (!player->audit_tree(error))
            {
                ++audits;
                if (first_error.empty()) first_error = error;
            }
        }

        std::printf("  %-30s back-up ordering %d | tree audit %d%s%s\n",
            config_name(config), ordering, audits,
            first_error.empty() ? "" : "  <- ", first_error.c_str());

        failures += ordering + audits;
    }

    std::printf("%s\n", failures ? "  FAILED" : "  ok");
    return failures ? 1 : 0;
}

// ------------------------------------------------------------------- speed ---

void run_speed_comparison(const std::vector<MigoyugoBB>& positions, int simulations_per_move);

void run_speed(int ms_per_move, const std::string& weights)
{
    auto model = load_model(weights);
    // One set of positions for every configuration and for GPlayer, so the
    // numbers are comparable to each other and not just to themselves.
    const auto positions = sample_positions(12, 8, 40);
    const Milliseconds budget(ms_per_move);

    std::printf("\nthroughput at %d ms/move over %zu positions\n", ms_per_move, positions.size());

    for (Config config : ALL_CONFIGS)
    {
        if (config_needs_network(config) && !model) continue;

        auto player = make_player(config, budget, model);

        double seconds = 0;
        uint64_t simulations = 0;
        size_t nodes = 0, expanded = 0;
        for (const auto& pos : positions)
        {
            player->search_position(pos);
            seconds += player->last_elapsed_s();
            simulations += player->last_simulation_count();
            nodes += player->node_count();
            expanded += player->expanded_node_count();
        }

        std::printf("  %-30s %10.0f sims/s   %7zu nodes/move (%zu expanded)\n",
            config_name(config),
            seconds > 0 ? simulations / seconds : 0.0,
            nodes / positions.size(), expanded / positions.size());
    }

    run_speed_comparison(positions, 20000);
}

// The old bot on the same positions, at a fixed simulation count.
//
// Timing GPlayer by its own printed GSims is worthless: its simulation rate
// spans three orders of magnitude across a game (1.4k to 4.6M per move in a
// 30-game match), because a rollout from a nearly full board is one or two
// plies. Any average over a game is dominated by trivial endgames. Fixing the
// count and timing it is the only comparison that means anything - and it has
// to be on the same positions the configurations above were measured on.
void run_speed_comparison(const std::vector<MigoyugoBB>& positions, int simulations_per_move)
{
    std::printf("\nsame positions, same work: %d simulations/move for both bots\n",
        simulations_per_move);
    std::printf("(GPlayer prints a line per search from g.cpp; the two numbers below are the point)\n");

    const double total = static_cast<double>(simulations_per_move) * positions.size();

    // A zero duration means G::search runs exactly minimum_no_simulations: its
    // timed loop finds the deadline already passed.
    rl::players::GPlayer old_player(simulations_per_move, Milliseconds(0), 15, 0.04f);
    double old_seconds = 0;
    for (const auto& position : positions)
    {
        std::unique_ptr<rl::common::IState> state =
            MigoyugoLightState::from_short(bb_to_short(position));
        const auto start = std::chrono::high_resolution_clock::now();
        old_player.choose_action(state);
        old_seconds += std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - start).count();
    }

    auto new_player = make_player(Config::even_tactical, Milliseconds(0), nullptr);
    new_player->set_fixed_simulations(simulations_per_move);
    double new_seconds = 0;
    for (const auto& position : positions)
    {
        new_player->search_position(position);
        new_seconds += new_player->last_elapsed_s();
    }

    const double old_rate = old_seconds > 0 ? total / old_seconds : 0.0;
    const double new_rate = new_seconds > 0 ? total / new_seconds : 0.0;

    std::printf("  %-30s %10.0f sims/s\n", "GPlayer (IState)", old_rate);
    std::printf("  %-30s %10.0f sims/s   %.1fx\n", "GRAVE-BB (same algorithm)",
        new_rate, old_rate > 0 ? new_rate / old_rate : 0.0);
}

// ------------------------------------------------------------------- games ---

using Agent = std::function<int(const MigoyugoBB&)>;

// +1 if player 0 wins, -1 if player 1 wins, 0 for a draw.
int play_game(const Agent& player_0, const Agent& player_1, const MigoyugoBB& opening)
{
    MigoyugoBB board = opening;
    for (;;)
    {
        const uint64_t legal = board.legal_moves();
        if (legal == 0)
        {
            // Wego: no legal move, so the Yugo count decides it.
            const int yugos_0 = board.yugo_count(0);
            const int yugos_1 = board.yugo_count(1);
            return yugos_0 > yugos_1 ? 1 : (yugos_0 < yugos_1 ? -1 : 0);
        }

        const int mover = board.stm;
        const int move = (mover == 0 ? player_0 : player_1)(board);

        if (move < 0 || ((legal >> move) & 1ULL) == 0)
        {
            std::printf("  ILLEGAL move %d by player %d\n", move, mover);
            return mover == 0 ? -1 : 1; // an illegal move forfeits
        }

        Undo undo;
        if (board.do_move(move, undo)) return mover == 0 ? 1 : -1; // Igo
    }
}

// Random but legal openings, so a deterministic pair of bots does not play the
// same game every time. Both seats get each opening once.
std::vector<MigoyugoBB> make_openings(int count)
{
    return sample_positions(count, 2, 6);
}

struct MatchResult { int wins = 0, losses = 0, draws = 0; };

MatchResult play_match(const Agent& challenger, const Agent& defender, int games)
{
    MatchResult result;
    const auto openings = make_openings((games + 1) / 2);

    for (int g = 0; g < games; ++g)
    {
        const MigoyugoBB& opening = openings[g / 2];
        const bool challenger_is_player_0 = (g % 2) == 0;

        const int outcome = challenger_is_player_0
            ? play_game(challenger, defender, opening)
            : play_game(defender, challenger, opening);

        const int from_challenger = challenger_is_player_0 ? outcome : -outcome;
        if (from_challenger > 0) ++result.wins;
        else if (from_challenger < 0) ++result.losses;
        else ++result.draws;

        std::printf("\r  game %d/%d: +%d =%d -%d   ",
            g + 1, games, result.wins, result.draws, result.losses);
        std::fflush(stdout);
    }
    std::printf("\n");
    return result;
}

void report(const char* challenger, const char* defender, const MatchResult& r, int games)
{
    const double score = (r.wins + 0.5 * r.draws) / games;
    std::printf("  %s vs %s: +%d =%d -%d  (%.1f%%)\n\n",
        challenger, defender, r.wins, r.draws, r.losses, 100.0 * score);
}

int run_match(int ms_per_move, int games, const std::string& weights)
{
    auto model = load_model(weights);
    const Milliseconds budget(ms_per_move);

    std::printf("\nmatch at %d ms/move, %d games per pairing\n\n", ms_per_move, games);

    // Each pairing answers one question, so each is worth running alone.
    struct Pairing { Config challenger, defender; };
    const Pairing pairings[] = {
        { Config::even_tactical, Config::even_uniform },  // does the tactical rollout help?
        { Config::leaf_blend,    Config::even_tactical }, // does the leaf blend help?
        { Config::action_value,  Config::even_tactical }, // do the NNUE priors help?
    };

    for (const Pairing& pairing : pairings)
    {
        if ((config_needs_network(pairing.challenger) || config_needs_network(pairing.defender))
            && !model)
            continue;

        auto challenger = make_player(pairing.challenger, budget, model);
        auto defender = make_player(pairing.defender, budget, model);

        const MatchResult result = play_match(
            [&](const MigoyugoBB& b) { return challenger->search_position(b); },
            [&](const MigoyugoBB& b) { return defender->search_position(b); },
            games);

        report(config_name(pairing.challenger), config_name(pairing.defender), result, games);
    }
    return 0;
}

// The leaf blend is the configuration that actually won, so how much network
// to mix in is the one parameter worth tuning. Every weight plays the same
// reference (w = 0.5), which is what the UI ships with.
int run_leaf_sweep(int ms_per_move, int games, const std::string& weights)
{
    auto model = load_model(weights);
    if (!model) return 1;

    const Milliseconds budget(ms_per_move);
    const float reference_weight = 0.5f;

    std::printf("\nleaf-value weight sweep at %d ms/move, %d games each, all vs w=%.2f\n\n",
        ms_per_move, games, reference_weight);

    for (float weight : { 0.0f, 0.25f, 0.75f, 1.0f })
    {
        auto challenger = make_player(Config::leaf_blend, budget, model);
        challenger->set_leaf_value_weight(weight);

        auto defender = make_player(Config::leaf_blend, budget, model);
        defender->set_leaf_value_weight(reference_weight);

        const MatchResult result = play_match(
            [&](const MigoyugoBB& b) { return challenger->search_position(b); },
            [&](const MigoyugoBB& b) { return defender->search_position(b); },
            games);

        const double score = (result.wins + 0.5 * result.draws) / games;
        std::printf("  w=%.2f vs w=%.2f: +%d =%d -%d  (%.1f%%)\n\n",
            weight, reference_weight, result.wins, result.draws, result.losses, 100.0 * score);
    }
    return 0;
}

int run_vs_gplayer(int ms_per_move, int games)
{
    const Milliseconds budget(ms_per_move);

    std::printf("\nGRAVE-BB vs GPlayer at %d ms/move, %d games\n", ms_per_move, games);
    std::printf("(GPlayer prints a line per move from g.cpp; redirect if it is in the way)\n\n");

    auto challenger = make_player(Config::even_tactical, budget, nullptr);
    rl::players::GPlayer defender(2, budget, 15, 0.04f);

    const MatchResult result = play_match(
        [&](const MigoyugoBB& b) { return challenger->search_position(b); },
        [&](const MigoyugoBB& b)
        {
            std::unique_ptr<rl::common::IState> state =
                MigoyugoLightState::from_short(bb_to_short(b));
            return defender.choose_action(state);
        },
        games);

    report("GRAVE-BB (even-game, tactical)", "GPlayer", result, games);
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    const std::string mode = argc > 1 ? argv[1] : "all";
    const int arg = argc > 2 ? std::atoi(argv[2]) : 0;

    if (mode == "selftest")
        return run_selftest(arg ? arg : 400,
            argc > 3 ? argv[3] : "../checkpoints/nnue_layerstacks_v2_weights.bin");

    if (mode == "invariants")
        return run_invariants(arg ? arg : 60,
            argc > 3 ? argv[3] : "../checkpoints/nnue_layerstacks_v2_weights.bin");

    if (mode == "speed")
    {
        run_speed(arg ? arg : 1000,
            argc > 3 ? argv[3] : "../checkpoints/nnue_layerstacks_v2_weights.bin");
        return 0;
    }

    if (mode == "match")
    {
        const int games = argc > 3 ? std::atoi(argv[3]) : 40;
        const std::string weights = argc > 4 ? argv[4]
            : "../checkpoints/nnue_layerstacks_v2_weights.bin";
        return run_match(arg ? arg : 1000, games, weights);
    }

    if (mode == "leafsweep")
    {
        const int games = argc > 3 ? std::atoi(argv[3]) : 20;
        const std::string weights = argc > 4 ? argv[4]
            : "../checkpoints/nnue_layerstacks_v2_weights.bin";
        return run_leaf_sweep(arg ? arg : 200, games, weights);
    }

    if (mode == "vsgplayer")
    {
        const int games = argc > 3 ? std::atoi(argv[3]) : 40;
        return run_vs_gplayer(arg ? arg : 1000, games);
    }

    const std::string weights = argc > 3 ? argv[3]
        : "../checkpoints/nnue_layerstacks_v2_weights.bin";
    int failures = 0;
    failures += run_selftest(400, weights);
    failures += run_invariants(60, weights);
    run_speed(arg ? arg : 1000, weights);
    return failures ? 1 : 0;
}
