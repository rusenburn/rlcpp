#ifndef RL_NNUE_MIGOYUGO_GRAVE_PLAYER_HPP_
#define RL_NNUE_MIGOYUGO_GRAVE_PLAYER_HPP_

// GRAVE (Generalized Rapid Action Value Estimation) for Migoyugo, on the
// bitboard environment.
//
// The algorithm is the one in players/src/bandits/grave/g_node.cpp - which is
// the only correct GRAVE in this repository, since GraveNode reads its own
// AMAF tables rather than the reference node's and is therefore plain
// per-node RAVE. What changes here is everything underneath it:
//
//   * No IState. games/include/games/migoyugo_bb.hpp answers every rule query
//     with a handful of 64-bit shifts and maintains the legality, promotion
//     and instant-win masks incrementally. GNode paid a vector<bool> mask
//     allocation, a heap-allocated successor state and two linear scans over
//     64 actions for every single rollout ply.
//   * No state per node, and no per-node heap allocation. Nodes live in an
//     arena and address each other by index; the board is a 120-byte struct
//     that each simulation copies once from the root, so the descent needs no
//     undo records at all - a struct copy is cheaper than unwinding one.
//   * Node statistics are allocated only when a node first selects a move.
//     The frontier of a million-simulation tree is mostly nodes visited once,
//     and a 1.8 KB statistics block for each of them is what would otherwise
//     put the memory ceiling out of reach.
//
// Two optional uses of the layer-stacked NNUE v2 network, independently
// switchable and both off by default:
//
//   * heuristic_mode_ == nnue_action_value is Heuristic MC-RAVE exactly as in
//     Gelly & Silver: Q(s,a) <- H(s,a), N(s,a) <- C(s,a) when a node is
//     expanded. The network only scores positions, not moves, but that is all
//     their strongest heuristic (Q_rlgo, a linear model over local shape
//     features) did either - a state evaluator becomes a state-action value by
//     one-ply lookahead, H(s,a) = -V(s.a). Costs one evaluation per legal move
//     per expansion.
//   * leaf_value_weight_ > 0 blends the network's opinion of the leaf into the
//     simulation's return. Costs one evaluation per simulation, and keeps the
//     rollout, so RAVE keeps its full AMAF signal.
//
// With both off this is GNode's algorithm on a fast environment, which is the
// A/B that isolates the speedup.

#include <common/player.hpp>
#include <common/random.hpp>
#include <games/migoyugo_bb.hpp>

#include "nnue_layerstacks_eval_v2.hpp"
#include "nnue_layerstacks_model_v2.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

namespace rl::players
{

namespace mgbb = rl::games::mgbb;

class MigoyugoGravePlayer : public rl::common::IPlayer
{
public:
    enum class HeuristicMode
    {
        // W~(s,a) = 0, N~(s,a) = C(s,a): every move starts at the value of a
        // drawn game, which is what gives first-play urgency.
        even_game,
        // Q(s,a) = -V(s.a) from the network, weighted by C(s,a).
        nnue_action_value,
    };

    enum class RolloutPolicy
    {
        // Win now if an Igo is available, else block the opponent's Igo
        // threat, else uniform. The masks are already maintained by
        // MigoyugoBB, so this is two extra ANDs per ply, and it removes the
        // largest source of rollout noise: games decided by a random move
        // overlooking an instant win.
        tactical,
        // The default policy of both papers, and of GNode.
        uniform_random,
    };

    static constexpr int N_ACTIONS = 64;

    // Deep enough that no real game reaches it; a simulation that does is
    // treated as a leaf rather than growing the tree further.
    static constexpr int MAX_TREE_DEPTH = 128;

    // Cap on the actions one simulation may contribute to AMAF. A Migoyugo
    // game is far shorter than this - occupancy strictly increases except on a
    // promotion, and promotions are capped by the 64 possible Yugos - but the
    // arrays are fixed size, so the recording stops rather than overruns.
    static constexpr int MAX_SIMULATION_ACTIONS = 1024;

    MigoyugoGravePlayer(std::chrono::duration<int, std::milli> minimum_duration,
        int minimum_simulations = 2,
        std::shared_ptr<const NNUELayerStacksModelV2> model = nullptr)
        : model_(std::move(model)),
        minimum_duration_(minimum_duration),
        minimum_simulations_(minimum_simulations),
        random_state_(seed_from_global_generator())
    {
        set_memory_budget_mb(256);
        nodes_.reserve(4096);
        node_statistics_.reserve(512);
    }

    // ------------------------------------------------------------ config ---

    void set_heuristic_mode(HeuristicMode mode)
    {
        // Refusing rather than pretending: with no network there is nothing to
        // compute H(s,a) from, and silently running even-game while reporting
        // otherwise is the kind of thing that wastes an afternoon of matches.
        heuristic_mode_ = (mode == HeuristicMode::nnue_action_value && !model_)
            ? HeuristicMode::even_game : mode;
        refresh_accumulator_requirement();
    }

    // C(s,a) in the paper: how many simulations the prior is worth. 50 is what
    // GNode hard-codes for its even-game prior.
    void set_equivalent_experience(int simulations)
    {
        equivalent_experience_ = std::max(1, simulations);
    }

    // 0 = pure rollout, 1 = pure network. The rollout still runs at 1 because
    // its action list is what feeds AMAF.
    void set_leaf_value_weight(float weight)
    {
        leaf_value_weight_ = model_ ? std::clamp(weight, 0.0f, 1.0f) : 0.0f;
        refresh_accumulator_requirement();
    }

    void set_rollout_policy(RolloutPolicy policy) { rollout_policy_ = policy; }

    // This is Silver's 4b^2, not b. 0.04 is what every call site in the repo
    // uses for GPlayer.
    void set_rave_bias(float bias) { rave_bias_ = bias; }

    // GRAVE's `ref`: the AMAF statistics used at a node are those of the
    // closest ancestor-or-self with more than this many visits.
    void set_amaf_reference_threshold(int visits) { amaf_reference_threshold_ = visits; }

    void set_minimum_duration(std::chrono::duration<int, std::milli> duration) { minimum_duration_ = duration; }

    // A floor and a ceiling on the simulation count. The time budget can only
    // stop the search between the two.
    void set_minimum_simulations(int simulations) { minimum_simulations_ = std::max(1, simulations); }
    void set_max_simulations(int simulations) { max_simulations_ = std::max(1, simulations); }

    // Exactly this many simulations, whatever the clock says - a reproducible
    // search for tests and for comparing configurations at equal work. Setting
    // only the ceiling is not enough: the time check still fires below it, and
    // with a zero budget that caps every search at 64 simulations.
    void set_fixed_simulations(int simulations)
    {
        set_minimum_simulations(simulations);
        set_max_simulations(simulations);
    }

    void set_verbose(bool on) { verbose_ = on; }

    // Splits between node headers and statistics blocks. Headers are 32 bytes
    // and statistics blocks are 1.8 KB, and only a minority of nodes ever need
    // a block, so the split is deliberately lopsided.
    void set_memory_budget_mb(int megabytes)
    {
        const size_t budget = static_cast<size_t>(std::max(8, megabytes)) * 1024u * 1024u;
        max_nodes_ = std::max<size_t>(1024, (budget / 8) / sizeof(SearchNode));
        max_statistics_ = std::max<size_t>(256, (budget - budget / 8) / sizeof(NodeStatistics));
    }

    // Deterministic runs for the test driver.
    void set_random_seed(uint64_t seed) { random_state_ = seed ? seed : 0x9e3779b97f4a7c15ULL; }

    // ------------------------------------------------------------- entry ---

    int choose_action(const std::unique_ptr<rl::common::IState>& state_ptr) override
    {
        return search_position(mgbb::MigoyugoBB::from_short(state_ptr->to_short()));
    }

    // Callers already holding a board skip the to_short round trip.
    // Returns the chosen square, or -1 if the position is already over.
    int search_position(const mgbb::MigoyugoBB& position)
    {
        const auto start_time = std::chrono::high_resolution_clock::now();

        last_move_ = -1;
        last_root_value_ = 0.0f;
        last_simulation_count_ = 0;
        last_elapsed_ = 0.0;

        root_board_ = position;
        nodes_.clear();
        node_statistics_.clear();
        invariant_violations_ = 0;
        simulation_actions_overflowed_ = false;

        const int32_t root = create_node(root_board_);
        if (root < 0 || nodes_[root].is_terminal) return -1;

        const uint64_t root_legal = nodes_[root].legal_moves;

        // Only one legal move, or an Igo available right now: neither is worth
        // a search, and playing the win is not a judgement call.
        const uint64_t winning = root_board_.winning_moves();
        if (winning) { last_move_ = mgbb::ctz64(winning); last_root_value_ = 1.0f; return last_move_; }
        if ((root_legal & (root_legal - 1)) == 0) { last_move_ = mgbb::ctz64(root_legal); return last_move_; }

        if (accumulator_live_)
        {
            // Once per search, not once per simulation: the root never moves,
            // and rebuilding it costs more than an entire rollout.
            rl::nnue::build_accumulator(*model_, root_board_, accumulator_stack_[0]);
        }

        // Both budgets are floors, and both must be satisfied before the search
        // stops - the same contract as G::search, which runs its
        // minimum_no_simulations loop first and only then loops on the clock.
        // Whichever is the slower to be met is the one that decides.
        //
        // max_simulations_ is the one exception: a hard ceiling that overrides
        // both floors, so a test can ask for a reproducible amount of work.
        const auto deadline = start_time + minimum_duration_;
        int simulations = 0;
        for (;;)
        {
            run_simulation(root);
            ++simulations;

            if (simulations >= max_simulations_) break;

            // The clock is polled in batches: at a couple of microseconds per
            // simulation, one call to now() per simulation is real overhead.
            if (simulations >= minimum_simulations_ && (simulations & 63) == 0
                && std::chrono::high_resolution_clock::now() >= deadline)
                break;
        }

        // The move played is the argmax of the same GRAVE value used inside
        // the tree, with the root as its own reference node - not max-visits.
        float best_value = 0.0f;
        last_move_ = (nodes_[root].statistics_index >= 0)
            ? select_action(root, root, &best_value)
            : mgbb::ctz64(root_legal); // only reachable if the arenas were exhausted immediately
        last_root_value_ = best_value;
        last_simulation_count_ = simulations;
        last_elapsed_ = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - start_time).count();

        if (verbose_)
        {
            std::cout << "GRAVE-BB  sims " << last_simulation_count_
                << "\tvalue " << last_root_value_
                << "\tnodes " << nodes_.size()
                << "\texpanded " << node_statistics_.size()
                << "\tsims/s " << static_cast<uint64_t>(last_elapsed_ > 0 ? simulations / last_elapsed_ : 0)
                << "\tmove " << last_move_ << std::endl;
        }

        return last_move_;
    }

    // ------------------------------------------------------------- stats ---

    int last_move() const { return last_move_; }
    int last_simulation_count() const { return last_simulation_count_; }
    float last_root_value() const { return last_root_value_; }
    double last_elapsed_s() const { return last_elapsed_; }
    size_t node_count() const { return nodes_.size(); }
    size_t expanded_node_count() const { return node_statistics_.size(); }

    // -------------------------------------------------------- verification ---

    // Checks, during back-up, that each node's AMAF update sees exactly the
    // actions played strictly below it. Off by default; see back_up() for why
    // this particular ordering is the thing worth policing.
    void set_check_invariants(bool on) { check_invariants_ = on; }
    int invariant_violations() const { return invariant_violations_; }

    // Post-search walk of the arena. Returns false and fills `error` on the
    // first inconsistency found.
    bool audit_tree(std::string& error) const
    {
        const bool nnue_priors = (heuristic_mode_ == HeuristicMode::nnue_action_value);

        for (size_t i = 0; i < nodes_.size(); ++i)
        {
            const SearchNode& node = nodes_[i];

            if (node.is_terminal && node.statistics_index >= 0)
                return fail(error, "terminal node was expanded", i);

            if (node.statistics_index < 0)
            {
                if (node.visit_count != 0)
                    return fail(error, "unexpanded node has visits", i);
                continue;
            }

            const NodeStatistics& statistics = node_statistics_[node.statistics_index];

            // Every real visit is one increment of exactly one action counter,
            // plus whatever the prior seeded.
            int64_t counted = 0;
            for (int action = 0; action < N_ACTIONS; ++action)
            {
                const bool legal = ((node.legal_moves >> action) & 1ULL) != 0;
                if (!legal && statistics.action_visit_count[action] != 0)
                    return fail(error, "illegal action was selected", i);
                if (!legal && statistics.child_index[action] >= 0)
                    return fail(error, "illegal action has a child", i);
                counted += statistics.action_visit_count[action];

                // Priors are only ever added to, never removed.
                if (statistics.amaf_visit_count[0][action] < equivalent_experience_
                    || statistics.amaf_visit_count[1][action] < equivalent_experience_)
                    return fail(error, "AMAF count fell below its prior", i);

                const int32_t child = statistics.child_index[action];
                if (child >= 0)
                {
                    if (static_cast<size_t>(child) >= nodes_.size())
                        return fail(error, "child index out of range", i);
                    if (nodes_[child].side_to_move == node.side_to_move)
                        return fail(error, "child has the same side to move", i);
                }
            }

            const int64_t prior = nnue_priors
                ? static_cast<int64_t>(equivalent_experience_) * mgbb::popcount64(node.legal_moves)
                : 0;
            if (counted != static_cast<int64_t>(node.visit_count) + prior)
                return fail(error, "N(s,a) does not sum to N(s)", i);
        }

        error.clear();
        return true;
    }

private:
    // -------------------------------------------------------------- tree ---

    // 32 bytes. Every node created gets one of these; most of them are
    // frontier nodes that are visited once and never select a move.
    struct SearchNode
    {
        uint64_t legal_moves;
        int32_t statistics_index; // -1 until this node first selects a move
        int32_t visit_count;      // N(s), counting only visits that selected
        float terminal_value;     // Wego result, from side_to_move's point of view
        int8_t side_to_move;
        bool needs_playout;       // true until this node has had its own rollout
        bool is_terminal;
    };

    // 1792 bytes, allocated only on expansion.
    struct NodeStatistics
    {
        float action_value_sum[N_ACTIONS];        // W(s,a)
        int32_t action_visit_count[N_ACTIONS];    // N(s,a)
        float amaf_value_sum[2][N_ACTIONS];       // W~(s,a), by ABSOLUTE colour
        int32_t amaf_visit_count[2][N_ACTIONS];   // N~(s,a)
        int32_t child_index[N_ACTIONS];           // -1 until the child exists
    };

    struct PathEntry
    {
        int32_t node_index;
        int32_t statistics_index;
        int8_t action;
        int8_t side_to_move;
    };

    // Two AMAF tables split by absolute colour are not optional: a reference
    // node is routinely consulted by a descendant of the opposite colour, so a
    // single table would be read with the wrong sign half the time.
    int32_t create_node(const mgbb::MigoyugoBB& board)
    {
        if (nodes_.size() >= max_nodes_) return -1;

        SearchNode node{};
        node.legal_moves = board.legal_moves();
        node.statistics_index = -1;
        node.visit_count = 0;
        node.side_to_move = static_cast<int8_t>(board.stm);
        node.needs_playout = true;
        node.is_terminal = (node.legal_moves == 0);
        // No legal move is a Wego: the game ends now and the Yugo count
        // decides it. An Igo is detected as it is played, so a node is never
        // created on a position already won.
        node.terminal_value = node.is_terminal ? board.wego_reward() : 0.0f;

        nodes_.push_back(node);
        return static_cast<int32_t>(nodes_.size() - 1);
    }

    // Returns the statistics index, or -1 if the arena is full (in which case
    // the caller treats the node as a permanent leaf and simply plays out).
    // `depth` is the node's depth, needed for the accumulator when the NNUE
    // action-value heuristic is on.
    int32_t expand_node(int32_t node_index, int depth)
    {
        if (node_statistics_.size() >= max_statistics_) return -1;

        node_statistics_.emplace_back();
        const int32_t index = static_cast<int32_t>(node_statistics_.size() - 1);

        NodeStatistics& statistics = node_statistics_[index];
        std::memset(statistics.action_value_sum, 0, sizeof(statistics.action_value_sum));
        std::memset(statistics.action_visit_count, 0, sizeof(statistics.action_visit_count));
        std::memset(statistics.amaf_value_sum, 0, sizeof(statistics.amaf_value_sum));
        for (int action = 0; action < N_ACTIONS; ++action)
        {
            statistics.child_index[action] = -1;
            // The even-game prior, on both colours and on every square,
            // including illegal ones - AMAF records every action played
            // anywhere in the subtree, not only the ones legal here.
            statistics.amaf_visit_count[0][action] = equivalent_experience_;
            statistics.amaf_visit_count[1][action] = equivalent_experience_;
        }

        nodes_[node_index].statistics_index = index;

        if (heuristic_mode_ == HeuristicMode::nnue_action_value)
            apply_nnue_action_value_priors(node_index, index, depth);

        return index;
    }

    // Heuristic MC-RAVE's NewNode: Q(s,a) <- H(s,a), N(s,a) <- C(s,a), with
    // H(s,a) = -V(s.a) from a one-ply lookahead. Called with `board_` sitting
    // on this node's position and accumulator_stack_[depth] matching it.
    //
    // This is the one place an Undo record is needed, because we come back to
    // the same position once per legal move.
    void apply_nnue_action_value_priors(int32_t node_index, int32_t statistics_index, int depth)
    {
        const uint64_t legal = nodes_[node_index].legal_moves;
        const int mover = nodes_[node_index].side_to_move;
        const float weight = static_cast<float>(equivalent_experience_);

        for (uint64_t remaining = legal; remaining; remaining &= remaining - 1)
        {
            const int action = mgbb::ctz64(remaining);

            mgbb::Undo undo;
            mgbb::FeatureDelta delta;
            const bool igo = board_.do_move(action, undo, delta);

            float heuristic_value;
            if (igo)
            {
                // A move that wins outright needs no network.
                heuristic_value = 1.0f;
            }
            else
            {
                rl::nnue::accumulator_apply_delta(*model_,
                    accumulator_stack_[depth + 1], accumulator_stack_[depth], delta);
                // The network scores the child from the child's point of view,
                // which is the opponent's; negate to get ours.
                heuristic_value = -std::clamp(
                    rl::nnue::evaluate_position(*model_, accumulator_stack_[depth + 1], board_),
                    -1.0f, 1.0f);
            }

            board_.undo_move(undo);

            NodeStatistics& statistics = node_statistics_[statistics_index];
            statistics.action_value_sum[action] = heuristic_value * weight;
            statistics.action_visit_count[action] = equivalent_experience_;
            statistics.amaf_value_sum[mover][action] = heuristic_value * weight;
            statistics.amaf_visit_count[mover][action] = equivalent_experience_;
            // The opponent's AMAF table keeps its even-game prior. Seeding it
            // with -H is tempting, but it is a different quantity - "what the
            // opponent scored when they played this square somewhere in this
            // subtree" - and nothing in the paper supports guessing it.
        }
    }

    // ---------------------------------------------------------- selection ---

    // (1 - beta) * Q(s,a) + beta * Q~(s,a), with the real statistics taken
    // from `node` and the AMAF statistics from the reference node but indexed
    // by `node`'s colour. No UCT exploration term, deliberately: Gelly &
    // Silver found the optimal exploration rate for heuristic MC-RAVE to be
    // zero, and GNode has none either.
    int select_action(int32_t node_index, int32_t reference_index, float* out_value = nullptr) const
    {
        const SearchNode& node = nodes_[node_index];
        const NodeStatistics& statistics = node_statistics_[node.statistics_index];
        const NodeStatistics& reference = node_statistics_[nodes_[reference_index].statistics_index];
        const int colour = node.side_to_move;

        int best_action = -1;
        float best_value = -std::numeric_limits<float>::infinity();

        // Set bits only. With a near-empty board this loop runs 60 times per
        // node per simulation and is the hottest code in the search.
        for (uint64_t remaining = node.legal_moves; remaining; remaining &= remaining - 1)
        {
            const int action = mgbb::ctz64(remaining);

            const float real_visits = static_cast<float>(statistics.action_visit_count[action]) + 1e-8f;
            const float real_mean = statistics.action_value_sum[action] / real_visits;

            const float amaf_visits = static_cast<float>(reference.amaf_visit_count[colour][action]) + 1e-8f;
            const float amaf_mean = reference.amaf_value_sum[colour][action] / amaf_visits;

            const float beta = amaf_visits
                / (amaf_visits + real_visits + rave_bias_ * amaf_visits * real_visits);

            const float value = (1.0f - beta) * real_mean + beta * amaf_mean;
            if (value > best_value)
            {
                best_value = value;
                best_action = action;
            }
        }

        if (out_value) *out_value = best_value;
        return best_action;
    }

    // --------------------------------------------------------- simulation ---

    void run_simulation(int32_t root_index)
    {
        board_ = root_board_;
        path_length_ = 0;
        simulation_action_count_[0] = 0;
        simulation_action_count_[1] = 0;

        int32_t node_index = root_index;
        int32_t reference_index = root_index;
        int depth = 0;

        float value = 0.0f;   // relative to `player`
        int player = 0;

        for (;;)
        {
            if (nodes_[node_index].is_terminal)
            {
                value = nodes_[node_index].terminal_value;
                player = nodes_[node_index].side_to_move;
                break;
            }

            if (nodes_[node_index].needs_playout)
            {
                nodes_[node_index].needs_playout = false;
                run_leaf_evaluation(depth, value, player);
                break;
            }

            int32_t statistics_index = nodes_[node_index].statistics_index;
            if (statistics_index < 0)
            {
                statistics_index = expand_node(node_index, depth);
                if (statistics_index < 0)
                {
                    // Statistics arena exhausted: this node stays a leaf and
                    // just plays out. The search keeps improving the
                    // statistics it already has instead of dying.
                    run_leaf_evaluation(depth, value, player);
                    break;
                }
            }

            if (nodes_[node_index].visit_count > amaf_reference_threshold_)
                reference_index = node_index;

            const int action = select_action(node_index, reference_index);
            const int mover = nodes_[node_index].side_to_move;

            path_[path_length_].node_index = node_index;
            path_[path_length_].statistics_index = statistics_index;
            path_[path_length_].action = static_cast<int8_t>(action);
            path_[path_length_].side_to_move = static_cast<int8_t>(mover);
            ++path_length_;

            const bool igo = play_in_tree(action, depth);
            ++depth;

            if (igo)
            {
                value = 1.0f;
                player = mover;
                break;
            }

            int32_t child_index = node_statistics_[statistics_index].child_index[action];
            if (child_index < 0)
            {
                // create_node can reallocate nodes_, so nothing may hold a
                // SearchNode reference across this call.
                child_index = create_node(board_);
                if (child_index < 0)
                {
                    // Node arena exhausted: play out from here without
                    // recording a new node.
                    run_leaf_evaluation(depth, value, player);
                    break;
                }
                node_statistics_[statistics_index].child_index[action] = child_index;
            }

            node_index = child_index;

            if (depth >= MAX_TREE_DEPTH)
            {
                run_leaf_evaluation(depth, value, player);
                break;
            }
        }

        back_up(value, player);
    }

    // One ply of the descent, keeping the accumulator stack in step when the
    // network is in use. The child accumulator is written while the parent's
    // is read, so there is nothing to undo.
    bool play_in_tree(int action, int depth)
    {
        mgbb::Undo undo;
        if (!accumulator_live_) return board_.do_move(action, undo);

        mgbb::FeatureDelta delta;
        const bool igo = board_.do_move(action, undo, delta);
        rl::nnue::accumulator_apply_delta(*model_,
            accumulator_stack_[depth + 1], accumulator_stack_[depth], delta);
        return igo;
    }

    // The value a simulation returns from a leaf: the rollout, optionally
    // blended with the network's opinion of the leaf position.
    void run_leaf_evaluation(int depth, float& out_value, int& out_player)
    {
        if (leaf_value_weight_ <= 0.0f)
        {
            run_playout(out_value, out_player);
            return;
        }

        // Read the leaf before the rollout moves off it.
        const int leaf_player = board_.stm;
        const float leaf_value = std::clamp(
            rl::nnue::evaluate_position(*model_, accumulator_stack_[depth], board_), -1.0f, 1.0f);

        float rollout_value = 0.0f;
        int rollout_player = leaf_player;
        run_playout(rollout_value, rollout_player);

        const float rollout_from_leaf = (rollout_player == leaf_player) ? rollout_value : -rollout_value;

        out_value = leaf_value_weight_ * leaf_value + (1.0f - leaf_value_weight_) * rollout_from_leaf;
        out_player = leaf_player;
    }

    // Termination is guaranteed by the environment: (yugo_count, occupancy)
    // rises lexicographically every ply, so no ply cap is needed.
    void run_playout(float& out_value, int& out_player)
    {
        for (;;)
        {
            const uint64_t legal = board_.legal_moves();
            if (legal == 0)
            {
                out_value = board_.wego_reward();
                out_player = board_.stm;
                return;
            }

            int action;
            if (rollout_policy_ == RolloutPolicy::tactical)
            {
                const uint64_t winning = board_.winning_moves();
                if (winning)
                {
                    action = mgbb::ctz64(winning);
                }
                else
                {
                    const int opponent = 1 - board_.stm;
                    const uint64_t block = board_.raw_igo[opponent]
                        & board_.legal_moves_for(opponent)
                        & legal;
                    action = block ? mgbb::ctz64(block) : random_set_bit(legal);
                }
            }
            else
            {
                action = random_set_bit(legal);
            }

            const int mover = board_.stm;
            record_simulation_action(mover, action);

            mgbb::Undo undo;
            if (board_.do_move(action, undo))
            {
                out_value = 1.0f;
                out_player = mover;
                return;
            }
        }
    }

    // ------------------------------------------------------------ back-up ---

    // Walks the path from the leaf back to the root. The ordering matters and
    // is easy to get wrong: a node's AMAF tables are updated with the actions
    // played STRICTLY BELOW it, and only then does its own selected action
    // join the list for its ancestors. Doing it the other way round leaks a
    // node's own move into its own AMAF table - silently, with no crash, just
    // a weaker bot.
    //
    // The leaf itself is not in `path_`, so its own tables miss this one
    // simulation. That is deliberate: a leaf has no statistics block yet, and
    // a node needs more than amaf_reference_threshold_ visits before its
    // tables are ever read, by which point one missing sample is noise.
    void back_up(float value, int player)
    {
        // Everything recorded so far came from the rollout; the path's own
        // moves are appended one at a time as we walk back up.
        const int rollout_actions = simulation_action_count_[0] + simulation_action_count_[1];

        for (int i = path_length_ - 1; i >= 0; --i)
        {
            const PathEntry& entry = path_[i];
            NodeStatistics& statistics = node_statistics_[entry.statistics_index];

            // The expected count is passed in and checked inside update_amaf
            // rather than tested here, so that it measures what update_amaf
            // actually saw. Checked at this point it would pass whatever order
            // the two calls below are written in, which is the one thing it
            // exists to catch.
            update_amaf(statistics, value, player, rollout_actions + (path_length_ - 1 - i));

            record_simulation_action(entry.side_to_move, entry.action);

            nodes_[entry.node_index].visit_count += 1;
            statistics.action_visit_count[entry.action] += 1;
            statistics.action_value_sum[entry.action] +=
                (entry.side_to_move == player) ? value : -value;
        }
    }

    // Every action played anywhere below this node contributes, whether or not
    // it is legal here - this is GNode's save_illegal_amaf_actions = true, the
    // only setting any call site in the repository uses, and dropping the mask
    // test keeps this loop branch-free.
    void update_amaf(NodeStatistics& statistics, float value, int player, int expected_actions)
    {
        // A node must see the rollout plus exactly the moves of the path
        // entries below it - never its own.
        if (check_invariants_ && !simulation_actions_overflowed_
            && simulation_action_count_[0] + simulation_action_count_[1] != expected_actions)
            ++invariant_violations_;

        const float score_for_player_0 = (player == 0) ? value : -value;
        const float score_for_player_1 = -score_for_player_0;

        for (int i = 0; i < simulation_action_count_[0]; ++i)
        {
            const int action = simulation_actions_[0][i];
            statistics.amaf_visit_count[0][action] += 1;
            statistics.amaf_value_sum[0][action] += score_for_player_0;
        }
        for (int i = 0; i < simulation_action_count_[1]; ++i)
        {
            const int action = simulation_actions_[1][i];
            statistics.amaf_visit_count[1][action] += 1;
            statistics.amaf_value_sum[1][action] += score_for_player_1;
        }
    }

    void record_simulation_action(int colour, int action)
    {
        int& count = simulation_action_count_[colour];
        if (count < MAX_SIMULATION_ACTIONS)
            simulation_actions_[colour][count++] = static_cast<uint8_t>(action);
        else
            simulation_actions_overflowed_ = true;
    }

    static bool fail(std::string& error, const char* what, size_t node)
    {
        error = std::string(what) + " (node " + std::to_string(node) + ")";
        return false;
    }

    // -------------------------------------------------------------- misc ---

    // xorshift64*, seeded once. rl::common::get() constructs a
    // std::uniform_int_distribution on every call, which is real time when it
    // runs tens of millions of times per move, and it shares one global engine
    // with every other bot in the process.
    uint32_t next_random_below(uint32_t bound)
    {
        random_state_ ^= random_state_ >> 12;
        random_state_ ^= random_state_ << 25;
        random_state_ ^= random_state_ >> 27;
        const uint64_t scrambled = random_state_ * 0x2545f4914f6cdd1dULL;
        return static_cast<uint32_t>(((scrambled >> 32) * bound) >> 32);
    }

    int random_set_bit(uint64_t mask)
    {
        uint32_t skip = next_random_below(static_cast<uint32_t>(mgbb::popcount64(mask)));
        while (skip--) mask &= mask - 1;
        return mgbb::ctz64(mask);
    }

    static uint64_t seed_from_global_generator()
    {
        const uint64_t high = static_cast<uint64_t>(rl::common::mt());
        const uint64_t low = static_cast<uint64_t>(rl::common::mt());
        const uint64_t seed = (high << 32) ^ low;
        return seed ? seed : 0x9e3779b97f4a7c15ULL;
    }

    void refresh_accumulator_requirement()
    {
        accumulator_live_ = model_
            && (heuristic_mode_ == HeuristicMode::nnue_action_value || leaf_value_weight_ > 0.0f);
    }

    // ----------------------------------------------------------- members ---

    // Written at depth + 1 while depth is read, so index MAX_TREE_DEPTH + 1
    // must exist. 133 KB, and untouched unless the network is in use.
    alignas(64) int16_t accumulator_stack_[MAX_TREE_DEPTH + 2][2][256];

    std::shared_ptr<const NNUELayerStacksModelV2> model_;
    std::chrono::duration<int, std::milli> minimum_duration_;
    int minimum_simulations_;

    mgbb::MigoyugoBB root_board_;
    mgbb::MigoyugoBB board_;

    std::vector<SearchNode> nodes_;
    std::vector<NodeStatistics> node_statistics_;
    size_t max_nodes_{ 0 };
    size_t max_statistics_{ 0 };

    PathEntry path_[MAX_TREE_DEPTH + 1]{};
    int path_length_{ 0 };

    uint8_t simulation_actions_[2][MAX_SIMULATION_ACTIONS]{};
    int simulation_action_count_[2]{};

    bool simulation_actions_overflowed_{ false };

    HeuristicMode heuristic_mode_{ HeuristicMode::even_game };
    RolloutPolicy rollout_policy_{ RolloutPolicy::tactical };
    int equivalent_experience_{ 50 };
    float leaf_value_weight_{ 0.0f };
    float rave_bias_{ 0.04f };
    int amaf_reference_threshold_{ 15 };
    int max_simulations_{ std::numeric_limits<int>::max() };
    bool accumulator_live_{ false };
    bool verbose_{ true };
    bool check_invariants_{ false };
    int invariant_violations_{ 0 };

    uint64_t random_state_;

    int last_move_{ -1 };
    int last_simulation_count_{ 0 };
    float last_root_value_{ 0.0f };
    double last_elapsed_{ 0.0 };
};

} // namespace rl::players

#endif
