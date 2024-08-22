#include <cassert>
#include <math.h>
#include <common/random.hpp>
#include <common/exceptions.hpp>
#include <players/mcts.hpp>
#include <iostream>

namespace rl::players
{

MCTSNode::MCTSNode(std::unique_ptr<rl::common::IState> state_ptr, int n_game_actions, float cpuct)
    : state_ptr_{ std::move(state_ptr) },
    n_game_actions_{ n_game_actions },
    cpuct_{ cpuct },
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

std::pair<float, int> MCTSNode::search(std::unique_ptr<IEvaluator>& evaluator_ptr)
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
        actions_mask_ = state_ptr_->actions_mask();
        for (int action{ 0 }; action < n_game_actions_; action++)
        {
            children_.push_back(std::unique_ptr<MCTSNode>(nullptr));
        }

        auto [probs, wdl] = evaluator_ptr->evaluate(state_ptr_);
        float wins = wdl.at(0);
        probs_ = probs;
        return std::make_pair(wins, state_ptr_->player_turn());
    }

    int best_action = get_best_action();

    if (children_.at(best_action).get() == nullptr)
    {
        auto new_state_ptr = state_ptr_->step(best_action);
        children_.at(best_action) = std::make_unique<MCTSNode>(std::move(new_state_ptr), n_game_actions_, cpuct_);
    }
    const auto& new_node = children_.at(best_action);
    assert(new_node.get() != nullptr);

    auto [next_result, new_player] = new_node->search(evaluator_ptr);

    if (new_player != state_ptr_->player_turn())
    {
        next_result = -next_result;
    }
    delta_wins_.at(best_action) += next_result;
    actions_visits_.at(best_action) += 1;
    n_visits += 1;
    return std::make_pair(next_result, state_ptr_->player_turn());
}

int MCTSNode::get_best_action()
{
    assert(actions_mask_.size() != 0);
    float max_u = -INFINITY;
    int best_a = -1;
    for (int action{ 0 }; action < n_game_actions_; action++)
    {
        if (actions_mask_.at(action) == false)
        {
            continue;
        }
        float action_visits = actions_visits_.at(action);
        float qsa = 0;
        if (action_visits > 0)
        {
            qsa = delta_wins_.at(action) / action_visits;
        }
        float u = qsa + cpuct_ * probs_.at(action) * sqrtf(n_visits + 1e-8f) / (1 + action_visits);
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
            if (actions_mask_.at(i))
            {
                legal_actions.push_back(i);
            }
        }
        int action_index = rl::common::get(static_cast<int>(legal_actions.size()));
        best_a = legal_actions[action_index];
    }
    return best_a;
}

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
            if (actions_visits.at(action) == max_action_visits)
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
            probs.at(action) /= sum_probs;
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
std::vector<float> MCTSNode::search_and_get_probs(std::unique_ptr<IEvaluator>& evaluator_ptr, int n_sims, std::chrono::duration<int, std::milli> minimum_duration, float temperature)
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
        search(evaluator_ptr);
        simulation_count++;
    }

    while (t_end > std::chrono::high_resolution_clock::now())
    {
        search(evaluator_ptr);
        simulation_count++;
    }

    // std::cout << "MCTS " << simulation_count << std::endl;
    return get_probs(temperature);
}

MCTS::MCTS(std::unique_ptr<IEvaluator> evaluator_ptr, int n_game_actions, float cpuct, float temperature)
    : evaluator_ptr_{ std::move(evaluator_ptr) }, n_game_actions_{ n_game_actions }, cpuct_{ cpuct }, temperature_{ temperature }
{
}
MCTS::~MCTS() = default;

std::vector<float> MCTS::search(const rl::common::IState* state_ptr, int minimum_no_simulations, std::chrono::duration<int, std::milli> minimum_duration)
{
    assert(state_ptr->is_terminal() == false);
    root_node = std::make_unique<MCTSNode>(state_ptr->clone(), state_ptr->get_n_actions(), cpuct_);
    return root_node->search_and_get_probs(evaluator_ptr_, minimum_no_simulations, minimum_duration, temperature_);
}

} // namespace rl::search_trees
