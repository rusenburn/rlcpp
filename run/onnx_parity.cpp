// Proves the ONNX evaluator is a drop-in replacement for the libtorch one.
//
// scripts/export_az_onnx.py already checks that the exported graph matches
// PyTorch. That is not the same question as this one: the graph could be perfect
// and the C++ wrapper around it still disagree with NetworkEvaluator, because
// the parts that matter to the search - masking the policy with actions_mask(),
// renormalizing it, and reducing wdl to a scalar - happen outside the graph.
// OnnxEvaluator reimplements them without torch, so they need checking against
// the original.
//
// Three sections:
//   1. parity      - both evaluators over the same self-played game
//   2. throughput  - both at the batch sizes Amcts2 actually produces
//   3. end to end  - Amcts2 driven by each, with no changes to Amcts2
//
// Run it from build/Release/bin, like the other tools here, so that
// "../checkpoints" resolves.

#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <common/random.hpp>
#include <deeplearning/alphazero/networks/shared_res_nn.hpp>
#include <deeplearning/network_evaluator.hpp>
#include <games/migoyugo.hpp>
#include <onnxeval/onnx_evaluator.hpp>
#include <onnxeval/onnx_session_ort.hpp>
#include <players/bandits/amcts2/amcts2.hpp>

namespace
{
constexpr float CPUCT = 2.0f;
constexpr float TEMPERATURE = 1.0f;
constexpr int MAX_ASYNC_SIMULATIONS = 16;
// Zero on purpose: dirichlet noise is sampled from the global RNG, and the point
// of section 3 is to compare two searches that differ only in their evaluator.
constexpr float DIRICHLET_EPSILON = 0.0f;
constexpr float DIRICHLET_ALPHA = 0.3f;
constexpr float DEFAULT_VISITS = 1.0f;
constexpr float DEFAULT_WINS = -1.0f;

int choose_action(const std::vector<float>& probs, int n_game_actions)
{
    float remaining = rl::common::get();
    int action = 0;
    const int last = n_game_actions - 1;
    while ((action < last) && ((remaining -= probs.at(action)) >= 0)) ++action;
    return action;
}

int argmax(const std::vector<float>& v, int offset, int count)
{
    int best = 0;
    for (int i = 1; i < count; i++)
    {
        if (v.at(offset + i) > v.at(offset + best)) best = i;
    }
    return best;
}

// Section 1 --------------------------------------------------------------------
void compare_positions(rl::players::IEvaluator& reference,
    rl::players::IEvaluator& candidate,
    const rl::common::IState& start,
    int n_actions)
{
    std::cout << "\n--- parity over a self-played game ---\n";

    double sum_prob_diff = 0.0, sum_value_diff = 0.0;
    float max_prob_diff = 0.0f, max_value_diff = 0.0f;
    int samples = 0, argmax_agreements = 0;

    auto state = start.clone();
    while (!state->is_terminal())
    {
        auto [ref_probs, ref_values] = reference.evaluate(state);
        auto [got_probs, got_values] = candidate.evaluate(state);

        for (int a = 0; a < n_actions; a++)
        {
            const float d = std::fabs(ref_probs.at(a) - got_probs.at(a));
            sum_prob_diff += d;
            if (d > max_prob_diff) max_prob_diff = d;
        }
        const float dv = std::fabs(ref_values.at(0) - got_values.at(0));
        sum_value_diff += dv;
        if (dv > max_value_diff) max_value_diff = dv;

        if (argmax(ref_probs, 0, n_actions) == argmax(got_probs, 0, n_actions))
        {
            ++argmax_agreements;
        }
        ++samples;

        state = state->step(choose_action(ref_probs, n_actions));
    }

    if (samples == 0)
    {
        std::cout << "no positions visited\n";
        return;
    }

    std::cout << std::scientific << std::setprecision(3)
        << "probs  mean |diff| " << sum_prob_diff / (samples * n_actions)
        << "   max " << max_prob_diff << '\n'
        << "value  mean |diff| " << sum_value_diff / samples
        << "   max " << max_value_diff << '\n'
        << std::defaultfloat
        << "best move agreed on " << argmax_agreements << " / " << samples << " positions\n";
}

// Section 2 --------------------------------------------------------------------
void time_batches(rl::players::IEvaluator& reference,
    rl::players::IEvaluator& candidate,
    const rl::common::IState& start,
    int n_actions)
{
    std::cout << "\n--- throughput (ms per batch, mean of 20) ---\n";

    // A pool of distinct, reachable positions rather than the same one repeated,
    // so nothing downstream can benefit from a cache.
    std::vector<std::unique_ptr<rl::common::IState>> pool;
    auto state = start.clone();
    while (pool.size() < 128)
    {
        if (state->is_terminal()) state = start.clone();
        pool.emplace_back(state->clone());
        auto [probs, values] = reference.evaluate(state);
        state = state->step(choose_action(probs, n_actions));
    }

    std::cout << std::setw(8) << "batch" << std::setw(14) << "torch"
        << std::setw(14) << "onnx" << '\n';

    for (int batch : { 1, 8, 32, 128 })
    {
        std::vector<const rl::common::IState*> states;
        states.reserve(batch);
        for (int i = 0; i < batch; i++) states.push_back(pool.at(i).get());

        auto measure = [&states](rl::players::IEvaluator& ev) {
            ev.evaluate(states); // warm up: first call allocates and, for ORT, plans
            const auto t0 = std::chrono::high_resolution_clock::now();
            for (int rep = 0; rep < 20; rep++) ev.evaluate(states);
            const auto t1 = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double, std::milli>(t1 - t0).count() / 20.0;
        };

        const double torch_ms = measure(reference);
        const double onnx_ms = measure(candidate);
        std::cout << std::fixed << std::setprecision(3)
            << std::setw(8) << batch << std::setw(14) << torch_ms
            << std::setw(14) << onnx_ms << '\n';
    }
    std::cout << std::defaultfloat;
}

// Section 3 --------------------------------------------------------------------
std::vector<float> search_with(std::unique_ptr<rl::players::IEvaluator> evaluator,
    const rl::common::IState& start,
    int n_actions,
    int simulations)
{
    auto tree = rl::players::Amcts2(n_actions, std::move(evaluator), CPUCT, TEMPERATURE,
        MAX_ASYNC_SIMULATIONS, DIRICHLET_EPSILON, DIRICHLET_ALPHA,
        DEFAULT_VISITS, DEFAULT_WINS);
    return tree.search(&start, simulations, std::chrono::duration<int, std::milli>(0));
}

void drive_amcts2(std::unique_ptr<rl::players::IEvaluator> reference,
    std::unique_ptr<rl::players::IEvaluator> candidate,
    const rl::common::IState& start,
    int n_actions)
{
    constexpr int SIMULATIONS = 400;
    std::cout << "\n--- Amcts2 end to end (" << SIMULATIONS << " simulations, unmodified) ---\n";

    const auto onnx_probs = search_with(std::move(candidate), start, n_actions, SIMULATIONS);
    const auto torch_probs = search_with(std::move(reference), start, n_actions, SIMULATIONS);

    const int onnx_move = argmax(onnx_probs, 0, n_actions);
    const int torch_move = argmax(torch_probs, 0, n_actions);

    float max_diff = 0.0f;
    for (int a = 0; a < n_actions; a++)
    {
        max_diff = std::max(max_diff, std::fabs(onnx_probs.at(a) - torch_probs.at(a)));
    }

    std::cout << "torch picks " << torch_move << " (visit share " << torch_probs.at(torch_move) << ")\n"
        << "onnx  picks " << onnx_move << " (visit share " << onnx_probs.at(onnx_move) << ")\n"
        << "max |visit share difference| " << max_diff << '\n'
        << (onnx_move == torch_move ? "same move\n" : "DIFFERENT MOVE\n");
}
} // namespace

int main(int argc, char** argv)
{
    const std::filesystem::path folder("../checkpoints");
    const std::string nn_name = argc > 1 ? argv[1] : "migoyugo_strongest_900.pt";
    const std::string onnx_name = argc > 2 ? argv[2] : "migoyugo_az.onnx";

    auto state = rl::games::MigoyugoState::initialize();
    const auto observation_shape = state->get_observation_shape();
    const int n_actions = state->get_n_actions();

    // Reference: the existing libtorch path, built exactly as every other tool
    // here builds it. SharedResNetwork::save() stores no hyperparameters, so the
    // 128/512/5 must be restated.
    auto network_ptr = std::make_unique<rl::deeplearning::alphazero::SharedResNetwork>(
        observation_shape, n_actions, 128, 512, 5, true);
    network_ptr->load((folder / nn_name).string());
    network_ptr->to(torch::kCPU); // CPU both sides, so section 2 compares like with like
    auto reference = std::make_unique<rl::deeplearning::NetworkEvaluator>(
        std::move(network_ptr), n_actions, observation_shape);

    // Candidate: the same weights through ONNX Runtime.
    auto session = std::make_shared<rl::onnxeval::OrtOnnxSession>(
        (folder / onnx_name).string(), observation_shape, n_actions);
    std::cout << "\nONNX Runtime provider: " << session->provider_name() << std::endl;
    auto candidate = std::make_unique<rl::onnxeval::OnnxEvaluator>(
        session, n_actions, observation_shape);

    compare_positions(*reference, *candidate, *state, n_actions);
    time_batches(*reference, *candidate, *state, n_actions);
    drive_amcts2(reference->copy(), candidate->copy(), *state, n_actions);

    return 0;
}
