#include <cassert>
#include <players/bandits/amcts2/amcts2.hpp>
#include <iostream>
#include <stdexcept>



namespace rl::players
{
constexpr float EPS = 1e-8f;

Amcts2::Amcts2(int n_game_actions, std::unique_ptr<IEvaluator> evaluator_ptr, float cpuct, float temperature, int max_async_simulations, float default_visits, float default_wins)
    : n_game_actions_{ n_game_actions },
    evaluator_ptr_{ std::move(evaluator_ptr) },
    cpuct_{ cpuct },
    temperature_{ temperature },
    max_async_simulations_{ max_async_simulations },
    default_n_{ default_visits },
    default_w_{ default_wins }
{
}
Amcts2::~Amcts2() = default;

std::vector<float> Amcts2::search(const rl::common::IState* state_ptr, int minimum_no_simulations, std::chrono::duration<int, std::milli> minimum_duration)
{
    if (state_ptr == nullptr)
    {
        throw std::runtime_error("root is null ,but being searched by search tree");
    }

    // check if it is the state has only 1 legal action return legal actions

    auto legal_actions = state_ptr->actions_mask();
    int n_legal_actions = 0;
    for (bool m : legal_actions)
    {
        n_legal_actions += m ? 1 : 0;
    }
    if (n_legal_actions == 1)
    {
        std::vector<float> res;
        res.reserve(n_game_actions_);
        for (bool m : legal_actions)
        {
            res.emplace_back(float(m));
        }
        return res;
    }


    set_root(state_ptr);

    auto t_start = std::chrono::high_resolution_clock::now();
    auto t_end = t_start + minimum_duration;

    int simulations_count{ 0 };

    while (simulations_count <= minimum_no_simulations)
    {
        roll(false);
        simulations_count++;
        if (simulations_count % max_async_simulations_ == 0)
        {
            auto rollouts = get_rollouts();

            auto evaluations = evaluator_ptr_->evaluate(rollouts);

            evaluate_collected_states(evaluations);
            clear_rollout();
        }

    }

    while (t_end > std::chrono::high_resolution_clock::now())
    {
        roll(false);
        simulations_count++;
        if (simulations_count % max_async_simulations_ == 0)
        {
            auto rollouts = get_rollouts();
            auto evaluations = evaluator_ptr_->evaluate(rollouts);
            evaluate_collected_states(evaluations);
            clear_rollout();
        }
    }

    auto rollouts = get_rollouts();
    auto evaluations = evaluator_ptr_->evaluate(rollouts);
    evaluate_collected_states(evaluations);
    clear_rollout();
    // std::cout << "Amcts2: " << simulations_count << std::endl;
    return get_probs();
}

void players::Amcts2::set_root(const rl::common::IState* state_ptr)
{
    rollouts_.clear();
    assert(state_ptr->is_terminal() == false);
    root_node_ = std::make_unique<Amcts2Node>(state_ptr->clone(), state_ptr->get_n_actions(), cpuct_);
    // root_node_->expand_node();
}
void players::Amcts2::roll(bool use_dirichlet_noise)
{
    rollouts_.push_back(std::make_pair<rl::common::IState*, std::vector<Amcts2Info>>(nullptr, {}));
    auto& rollout_info = rollouts_.back();
    root_node_->simulate_once(rollout_info, use_dirichlet_noise, default_n_, default_w_,root_node_.get());
    if (rollout_info.first == nullptr)
    {
        rollouts_.pop_back();
    }
}
std::vector<const rl::common::IState*> players::Amcts2::get_rollouts()
{
    std::vector<const rl::common::IState*> res{};
    for (int i = 0; i < rollouts_.size(); i++)
    {
        auto state_ptr = std::get<0>(rollouts_.at(i));
        assert(state_ptr != nullptr);
        res.push_back(std::get<0>(rollouts_.at(i)));
    }
    return res;
}

void players::Amcts2::evaluate_collected_states(std::tuple<std::vector<float>, std::vector<float>>& evaluations_tuple)
{
    int n_states = static_cast<int>(rollouts_.size());
    std::vector<const rl::common::IState*> states_ptrs(n_states, nullptr);
    for (int i{ 0 }; i < n_states; i++)
    {
        states_ptrs.at(i) = std::get<0>(rollouts_.at(i));
    }
    std::vector<float>& probs = std::get<0>(evaluations_tuple);
    std::vector<float>& values = std::get<1>(evaluations_tuple);

    // backprob values
    for (int i{ 0 }; i < n_states; i++)
    {
        auto& visited_path = std::get<1>(rollouts_[i]);
        auto& state_ptr = std::get<0>(rollouts_[i]);
        float value = values.at(i);
        std::vector<float> state_probs(n_game_actions_, 0);
        int probs_start = i * n_game_actions_;
        for (int j = 0;j < n_game_actions_;j++)
        {
            state_probs.at(j) = probs.at(probs_start + j);
        }

        backpropogate(visited_path, value, states_ptrs.at(i)->player_turn(), state_probs);
    }
}

std::vector<float> players::Amcts2::get_probs()
{
    assert(root_node_ != nullptr);
    return root_node_->get_probs(temperature_);
}

float players::Amcts2::get_evaluation()
{
    assert(root_node_ != nullptr);
    return root_node_->get_evaluation();
}

void players::Amcts2::clear_rollout()
{
    rollouts_.clear();
}

void players::Amcts2::backpropogate(std::vector<Amcts2Info>& visited_path, float final_result, int final_player, std::vector<float>& probs)
{
    root_node_->backpropogate(visited_path, 0, final_result, final_player, probs, default_n_, default_w_);
}
} // namespace rl::players


