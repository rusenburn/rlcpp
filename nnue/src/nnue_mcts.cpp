#include <cassert>
#include <math.h>
#include <common/random.hpp>
#include <common/exceptions.hpp>
#include <nnue/nnue_mcts.hpp>
#include <iostream>
#include <cmath>
#include <memory>
#include <algorithm>

namespace rl::nnue
{

void MCTS::apply_update(const NNUEModel& model, std::array<int16_t, 256>& aw, std::array<int16_t, 256>& ab, const rl::games::NNUEUpdate& u)
{
    for (int f : u.white_added)   for (int j = 0; j < 256; ++j) aw[j] += model.l1_weights[j][f];
    for (int f : u.white_removed) for (int j = 0; j < 256; ++j) aw[j] -= model.l1_weights[j][f];
    for (int f : u.black_added)   for (int j = 0; j < 256; ++j) ab[j] += model.l1_weights[j][f];
    for (int f : u.black_removed) for (int j = 0; j < 256; ++j) ab[j] -= model.l1_weights[j][f];
}

MCTS::MCTS(int n_game_actions, float cpuct, float temperature)
    : n_game_actions_{ n_game_actions }, cpuct_{ cpuct }, temperature_{ temperature }
{
}
MCTS::~MCTS() = default;

std::vector<float> MCTS::search(const rl::games::MigoyugoLightState* state_ptr, const NNUEModel& model, int minimum_no_simulations, std::chrono::duration<int, std::milli> minimum_duration)
{
    assert(state_ptr->is_terminal() == false);

    std::array<int16_t, 256> acc_w, acc_b;
    rl::games::NNUEUpdate initial_features;
    state_ptr->get_active_features(initial_features);
    apply_update(model, acc_w, acc_b, initial_features);

    auto root_node = std::make_unique<rl::nnue::MCTSNode>(std::move(state_ptr->clone_state()), state_ptr->get_n_actions(), cpuct_, acc_w, acc_b);
    return root_node->search_and_get_probs(model, minimum_no_simulations, minimum_duration, temperature_);
}




MCTSNode::MCTSNode(
    std::unique_ptr<rl::games::MigoyugoLightState> state_ptr,
    int n_game_actions,
    float cpuct,
    const  std::array<int16_t, 256>& acc_w,
    const std::array<int16_t, 256>& acc_b)
    : state_ptr_{ std::move(state_ptr) },
    n_game_actions_{ n_game_actions },
    cpuct_{ cpuct },
    acc_w_{ acc_w },
    acc_b_{ acc_b },
    probs_(n_game_actions, 0.0f),
    n_visits{ 0 },
    actions_visits_(n_game_actions, 0.0f),
    delta_wins_(n_game_actions, 0.0f),
    is_terminal_{},
    game_result_{},
    children_{}
{
}

MCTSNode::~MCTSNode() = default;


std::pair<float, int> MCTSNode::search(const NNUEModel& model)
{
    if (is_terminal_.has_value() == false)
    {
        is_terminal_.emplace(state_ptr_->is_terminal());
    }

    if (is_terminal_.value())
    {
        if (game_result_.has_value() == false)
        {
            game_result_.emplace(state_ptr_->get_reward());
        }
        return std::make_pair(game_result_.value(), state_ptr_->player_turn());
    }

    if (actions_mask_.size() == 0)
    {
        // first visit => rollout
        actions_mask_ = state_ptr_->actions_mask_2();
        children_.resize(n_game_actions_);
        // for (int action{ 0 }; action < n_game_actions_; action++)
        // {
        //     children_.push_back(std::unique_ptr<MCTSNode>(nullptr));
        // }

        auto& current_acc = (state_ptr_->player_turn() == 0) ? acc_w_ : acc_b_;
        float wins = evaluate_nnue_simd(current_acc, model);
        int n_legal_actions = 0;
        for (int action = 0; action < n_game_actions_; action++)
        {
            n_legal_actions += actions_mask_[action];
        }

        for (int action = 0; action < n_game_actions_; action++)
        {
            if (actions_mask_[action])
            {
                probs_[action] = 1 / static_cast<float>(n_legal_actions);
            }
        }

        return std::make_pair(wins, state_ptr_->player_turn());
    }

    int best_action = get_best_action();

    if (children_[best_action].get() == nullptr)
    {
        rl::games::NNUEUpdate update;
        auto new_state_ptr = state_ptr_->step_state_light(best_action, update);

        std::array<int16_t, 256> acc_w = acc_w_;
        std::array<int16_t, 256> acc_b = acc_b_;

        apply_update(model, acc_w, acc_b, update);

        children_[best_action] = std::make_unique<MCTSNode>(std::move(new_state_ptr), n_game_actions_, cpuct_, acc_w, acc_b);
    }
    const auto& new_node = children_[best_action];
    assert(new_node.get() != nullptr);

    auto [next_result, new_player] = new_node->search(model);

    if (new_player != state_ptr_->player_turn())
    {
        next_result = -next_result;
    }
    delta_wins_[best_action] += next_result;
    actions_visits_[best_action] += 1;
    n_visits += 1;
    return std::make_pair(next_result, state_ptr_->player_turn());
}

float MCTSNode::evaluate_nnue_simd(const std::array<int16_t, 256>& accumulator, const NNUEModel& model) {
    // 1. Pre-calculate Activated Layer 1 (Clipped ReLU)
    // We do this once so we don't have to clamp inside the nested loops.
    alignas(32) std::array<int16_t, 256> activated_l1;
    for (size_t i = 0; i < 256; ++i) {
        activated_l1[i] = static_cast<int16_t>(std::clamp<int32_t>(accumulator[i], 0, 127));
    }

    // --- Layer 2: 256 -> 16 ---
    alignas(32) std::array<int32_t, 16> l2_out;
    for (size_t i = 0; i < 16; ++i) {
        // Initialize sum with bias
        __m128i sum_v = _mm_setzero_si128();

        // Process 256 inputs in chunks of 8 (128-bit SSE)
        for (size_t j = 0; j < 256; j += 8) {
            // Load 8 weights and 8 activated inputs
            __m128i weights = _mm_load_si128((__m128i*) & model.l2_weights[i][j]);
            __m128i inputs = _mm_load_si128((__m128i*) & activated_l1[j]);

            // _mm_madd_epi16: (w0*i0 + w1*i1), (w2*i2 + w3*i3)... 
            // Result is 4 int32s in one register
            __m128i madd = _mm_madd_epi16(weights, inputs);
            sum_v = _mm_add_epi32(sum_v, madd);
        }

        // Horizontal sum of the 4 ints in sum_v
        alignas(16) int32_t temp_sums[4];
        _mm_store_si128((__m128i*)temp_sums, sum_v);
        int32_t total_sum = model.l2_bias[i] + temp_sums[0] + temp_sums[1] + temp_sums[2] + temp_sums[3];

        l2_out[i] = std::clamp(total_sum >> 7, 0, 127);
    }

    // --- Layer 3: 16 -> 32 ---
    // (16 inputs is small, but we can still SSE it)
    alignas(32) std::array<int32_t, 32> l3_out;
    for (size_t i = 0; i < 32; ++i) {
        __m128i sum_v = _mm_setzero_si128();

        // Convert l2_out (int32) to int16 temporarily for madd instruction
        // or just do standard math since 16 iterations is tiny.
        int32_t total_sum = model.l3_bias[i];
        for (size_t j = 0; j < 16; ++j) {
            total_sum += model.l3_weights[i][j] * l2_out[j];
        }
        l3_out[i] = std::clamp(total_sum >> 7, 0, 127);
    }

    // --- Output Layer: 32 -> 1 ---
    int32_t final_sum = model.out_bias;
    for (size_t i = 0; i < 32; i += 4) {
        // Simple 32-bit math for the final layer
        for (size_t k = 0; k < 4; ++k) {
            final_sum += model.out_weights[i + k] * l3_out[i + k];
        }
    }

    return static_cast<float>(final_sum) / (128.0f * 128.0f);
}

void MCTSNode::apply_update(const NNUEModel& model, std::array<int16_t, 256>& aw, std::array<int16_t, 256>& ab, const rl::games::NNUEUpdate& u) {
    for (int f : u.white_added)   for (int j = 0; j < 256; ++j) aw[j] += model.l1_weights[j][f];
    for (int f : u.white_removed) for (int j = 0; j < 256; ++j) aw[j] -= model.l1_weights[j][f];
    for (int f : u.black_added)   for (int j = 0; j < 256; ++j) ab[j] += model.l1_weights[j][f];
    for (int f : u.black_removed) for (int j = 0; j < 256; ++j) ab[j] -= model.l1_weights[j][f];
}

int MCTSNode::get_best_action()
{
    assert(actions_mask_.size() != 0);
    if (priority_.size() == 0)
    {
        priority_ = state_ptr_->detect_threats();
    }
    float max_u = -INFINITY;
    int best_a = -1;
    std::vector<int> used(state_ptr_->get_n_actions(), 0);
    for (int action : priority_)
    {
        if (actions_mask_[action] == false)
        {
            continue;
        }
        used[action] = 1;
        float action_visits = actions_visits_[action];
        float qsa = 0;
        if (action_visits > 0)
        {
            qsa = delta_wins_[action] / action_visits;
        }
        float u = qsa + cpuct_ * probs_[action] * sqrtf(n_visits + 1e-8f) / (1 + action_visits);
        if (u > max_u)
        {
            max_u = u;
            best_a = action;
        }
    }
    for (int action = 0; action < n_game_actions_; action++)
    {
        if (actions_mask_[action] == false || used[action])
        {
            continue;
        }
        float action_visits = actions_visits_[action];
        float qsa = 0;
        if (action_visits > 0)
        {
            qsa = delta_wins_[action] / action_visits;
        }
        float u = qsa + cpuct_ * probs_[action] * sqrtf(n_visits + 1e-8f) / (1 + action_visits);
        u += 0.05f;

        if (u > max_u)
        {
            max_u = u;
            best_a = action;
        }
    }

    if (best_a == -1)
    {
        std::vector<int> legal_actions;
        for (int i = 0; i < this->n_game_actions_; i++)
        {
            if (actions_mask_[i])
            {
                legal_actions.push_back(i);
            }
        }
        int action_index = rl::common::get(static_cast<int>(legal_actions.size()));
        best_a = legal_actions[action_index];
    }
    return best_a;
}

// int MCTSNode::get_best_action() {
//     float max_u = -1e9f;
//     int best_a = -1;

//     // Calculate FPU: The "baseline" value for a node with 0 visits.
//     // If the current node is already bad (e.g., -0.5), we expect its children to be bad.
//     float parent_q = (n_visits > 0) ? (get_evaluation()) : 0.0f;
//     float fpu_penalty = 0.5f; // Adjust this: higher = more selective, lower = more exploratory
//     float fpu_value = parent_q - fpu_penalty;

//     for (int action = 0; action < n_game_actions_; action++) {
//         if (!actions_mask_[action]) continue;

//         float action_visits = actions_visits_[action];
//         float qsa;

//         if (action_visits > 0) {
//             qsa = delta_wins_[action] / action_visits;
//         }
//         else {
//             // APPLY FPU: Instead of 0, assume it's slightly worse than the parent
//             qsa = fpu_value;
//         }

//         // UCB Formula
//         float u = qsa + cpuct_ * probs_[action] * sqrtf(n_visits + 1e-8f) / (1 + action_visits);

//         if (u > max_u) {
//             max_u = u;
//             best_a = action;
//         }
//     }

//     if (best_a == -1)
//     {
//         std::vector<int> legal_actions;
//         for (int i = 0; i < this->n_game_actions_; i++)
//         {
//             if (actions_mask_[i])
//             {
//                 legal_actions.push_back(i);
//             }
//         }
//         int action_index = rl::common::get(static_cast<int>(legal_actions.size()));
//         best_a = legal_actions[action_index];
//     }
//     return best_a;
// }

std::vector<float> MCTSNode::get_probs(float temperature)
{
    const auto& actions_visits = actions_visits_;
    if (temperature == 0.0f)
    {
        // find max action visits
        float max_action_visits = -1.0f;
        for (float action_visits : actions_visits)
        {
            if (action_visits > max_action_visits)
            {
                max_action_visits = action_visits;
            }
        }
        std::vector<float> probs{};
        probs.reserve(n_game_actions_);
        float sum_probs = 0;
        for (int action{ 0 }; action < n_game_actions_; action++)
        {
            if (actions_visits[action] == max_action_visits)
            {
                sum_probs++;
                probs.emplace_back(1.0f);
            }
            else
            {
                probs.emplace_back(0.0f);
            }
        }
        if (sum_probs == 0.0f)
        {
            throw std::runtime_error("mcts no legal actions were provided");
        }
        // normalize probs
        for (int action{ 0 }; action < n_game_actions_; action++)
        {
            probs[action] /= sum_probs;
        }
        return probs;
    }
    std::vector<float> probs_with_temperature{};
    probs_with_temperature.reserve(n_game_actions_);
    float sum_action_visits{ 0.0f };
    float sum_actions_with_temperature{ 0.0f };
    for (auto& a : actions_visits)
    {
        sum_action_visits += a;
    }
    for (auto& a : actions_visits)
    {
        float p = powf(a / sum_action_visits, 1.0f / temperature);
        sum_actions_with_temperature += p;
        probs_with_temperature.emplace_back(p);
    }

    for (auto& prob : probs_with_temperature)
    {
        prob /= sum_actions_with_temperature;
    }
    // just assert that actions probs sums to 1 almost
    sum_actions_with_temperature = 0.0f;
    for (const auto& prob : probs_with_temperature)
    {
        sum_actions_with_temperature += prob;
    }
    if (sum_actions_with_temperature < 1.0f - 1e-3f || sum_actions_with_temperature > 1.0f + 1e-3f)
    {
        std::runtime_error("action probabilities do not equal to one");
    }
    return probs_with_temperature;
}


std::vector<float> MCTSNode::search_and_get_probs(const NNUEModel& model, int n_sims, std::chrono::duration<int, std::milli> minimum_duration, float temperature)
{
    if (is_terminal_.has_value() == false)
    {
        is_terminal_.emplace(state_ptr_->is_terminal());
    }
    if (is_terminal_.value())
    {
        throw rl::common::SteppingTerminalStateException("MCTS Node is searching a terminal state which cannot be stepped.");
    }

    auto t_start = std::chrono::high_resolution_clock::now();
    auto t_end = t_start + minimum_duration;

    int simulation_count{ 0 };

    while (simulation_count <= n_sims)
    {
        search(model);
        simulation_count++;
    }

    while (t_end > std::chrono::high_resolution_clock::now())
    {
        for (int i = 0;i < 100;i++)
        {
            search(model);
            simulation_count++;
        }
    }


    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_seconds = end_time - t_start;
    double nps = (duration_seconds.count() > 0) ? (simulation_count / duration_seconds.count()) : 0;
    auto value = get_evaluation();
    std::cout << "NNUE_MCTS Score: " << value
        << "\tSim/S: " << static_cast<uint64_t>(nps) << std::endl;

    return get_probs(temperature);
}

float MCTSNode::get_evaluation()const
{
    if (!state_ptr_)
    {
        std::runtime_error("Trying to get evaluation of nullptr state");
    }

    float sum_delta_wins = 0.0f;
    for (auto delta_win : delta_wins_)
    {
        sum_delta_wins += delta_win;
    }

    return sum_delta_wins / n_visits;
}

} // namespace rl::search_trees
