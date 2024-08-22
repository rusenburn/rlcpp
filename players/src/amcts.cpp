#include <players/amcts.hpp>

#include <cmath>
#include <stdexcept>
#include <players/amcts.hpp>
#include <common/random.hpp>
#include <iostream>

namespace rl::players
{
constexpr float EPS = 1e-8f;
Amcts::Amcts(int n_game_actions,
    std::unique_ptr<IEvaluator> evaluator_ptr,
    float cpuct,
    float temperature,
    int max_async_simulations,
    float default_visits,
    float default_wins)
    : n_game_actions_{ n_game_actions },
    max_async_simulations_{ max_async_simulations },
    default_n_{ default_visits },
    default_w_{ default_wins },
    evaluator_ptr_{ std::move(evaluator_ptr) },
    cpuct_{ cpuct },
    temperature_{ temperature },
    states_{},
    edges_{},
    ns_{},
    nsa_{},
    wsa_{},
    psa_{},
    masks_{},
    root_ptr{ nullptr },
    root_player_{ -1 },
    rollouts_{}
{
}
Amcts::~Amcts() = default;
std::vector<float> Amcts::search(const rl::common::IState* state_ptr, int minimum_no_simulations, std::chrono::duration<int, std::milli> minimum_duration)
{
    // TODO if should be clearing previous data
    // states_.clear();
    // edges_.clear();
    // ns_.clear();
    // nsa_.clear();
    // wsa_.clear();
    // psa_.clear();
    // masks_.clear();
    // rollouts_.clear();
    // root_ptr = state_ptr->clone();
    return search_root(state_ptr, minimum_no_simulations, minimum_duration);
}

std::vector<float> Amcts::search_root(const rl::common::IState* root_ptr, int minimum_no_simulations, std::chrono::duration<int, std::milli> minimum_duration)
{
    if (root_ptr == nullptr)
    {
        throw std::runtime_error("root is null ,but being searched by search tree");
    }
    root_player_ = root_ptr->player_turn();

    // check if it is the root state has only 1 legal action return legal actions

    auto legal_actions = root_ptr->actions_mask();
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

    auto t_start = std::chrono::high_resolution_clock::now();
    auto t_end = t_start + minimum_duration;

    int simulations_count{ 0 };

    while (simulations_count <= minimum_no_simulations)
    {
        std::vector<AmctsInfo> visited_path{};
        simulate_once(root_ptr, visited_path);
        simulations_count++;
        if (simulations_count % max_async_simulations_ == 0)
        {
            evaluate_collected_states();
        }
    }

    while (t_end > std::chrono::high_resolution_clock::now())
    {
        std::vector<AmctsInfo> visited_path{};
        simulate_once(root_ptr, visited_path);
        simulations_count++;
        if (simulations_count % max_async_simulations_ == 0)
        {
            evaluate_collected_states();
        }
    }

    evaluate_collected_states();
    // std::cout << "Amsts: " <<simulations_count << std::endl;
    return get_probs(root_ptr);
}

void Amcts::simulate_once(const rl::common::IState* state_ptr, std::vector<AmctsInfo>& visited_path)
{
    if (state_ptr->is_terminal())
    {
        // state is terminal get result

        float result = state_ptr->get_reward();
        int player = state_ptr->player_turn();

        backpropogate(visited_path, result, player);
        return;
    }

    auto short_state = state_ptr->to_short();
    int player = state_ptr->player_turn();

    if (states_.find(short_state) == states_.end())
    {
        // first visit ( short states are not in state )
        expand_state(state_ptr, short_state);
        std::vector<bool> actions_mask = masks_.at(short_state);

        // check if it has more than  1 legal action
        int n_legal_actions{ 0 };
        for (bool m : actions_mask)
        {
            n_legal_actions += int(m);
        }
        if (n_legal_actions == 1)
        {
            // forced move , only 1 legal action , change probs to have this action as 1 and the rest are 0 , skip rollouts/evaluation and continue down the tree
            std::vector<float> probs;
            probs.reserve(n_game_actions_);
            for (bool m : actions_mask)
            {
                probs.emplace_back(float(m));
            }
            psa_.at(short_state) = probs;
        }
        else
        {
            add_to_rollouts(state_ptr, visited_path);
            return;
        }
    }

    // continue down the tree

    int best_action = find_best_action(short_state);

    if (!edges_[short_state].at(best_action)) // check if edge of best action is null
    {
        edges_[short_state].at(best_action) = state_ptr->step(best_action);
    }

    auto new_state_ptr = edges_[short_state].at(best_action).get();

    if (new_state_ptr == nullptr)
    {
        throw std::runtime_error("new state was found to be null");
    }

    visited_path.push_back({ short_state, best_action, player });
    nsa_.at(short_state).at(best_action) += default_n_;
    ns_.at(short_state) += default_n_;
    wsa_.at(short_state).at(best_action) += default_w_;
    simulate_once(new_state_ptr, visited_path);
}

int Amcts::find_best_action(std::string& short_state)
{
    float max_u = -INFINITY;
    int best_action = -1;

    const auto& wsa_vec = wsa_.at(short_state);
    const auto& nsa_vec = nsa_.at(short_state);
    const auto& psa_vec = psa_.at(short_state);
    const auto& masks = masks_.at(short_state);
    float current_state_visis = ns_.at(short_state);

    for (int action{ 0 }; action < masks.size(); action++)
    {
        bool is_legal = masks.at(action);
        if (!is_legal)
        {
            continue;
        }
        float action_prob = psa_vec.at(action);
        float action_visits = nsa_vec.at(action);
        float qsa = 0.0f;

        if (action_visits > 0)
        {
            qsa = wsa_vec.at(action) / (action_visits + EPS);
        }
        float u = qsa + cpuct_ * action_prob * sqrtf(current_state_visis + EPS) / (1.0f + action_visits);

        if (u > max_u)
        {
            max_u = u;
            best_action = action;
        }
    }
    if (best_action == -1)
    {
        std::vector<int> legal_actions;
        for (int i = 0; i < this->n_game_actions_; i++)
        {
            if (masks.at(i))
            {
                legal_actions.push_back(i);
            }
        }
        // int action_index = rand() % legal_actions.size();
        int action_index = rl::common::get(static_cast<int>(legal_actions.size()));
        best_action = legal_actions[action_index];
        // should not happen but somehow it did;
    }
    return best_action;
}
void Amcts::expand_state(const rl::common::IState* state_ptr, std::string& short_state)
{
    if (state_ptr->is_terminal())
    {
        throw std::runtime_error("Expanding a terminal state");
    }
    states_.insert(short_state);

    std::vector<bool> action_mask = state_ptr->actions_mask();
    masks_[short_state] = action_mask;
    ns_[short_state] = 0;
    nsa_[short_state] = std::vector<float>(n_game_actions_, 0.0f);
    wsa_[short_state] = std::vector<float>(n_game_actions_, 0.0f);
    // edges_[short_state] = std::move(std::vector<std::unique_ptr<rl::common::State>>(n_game_actions_, std::unique_ptr<rl::common::State>(nullptr)));
    auto& vec = edges_[short_state];
    vec.clear();
    for (int i{ 0 }; i < n_game_actions_; i++)
    {
        vec.push_back(std::unique_ptr<rl::common::IState>(nullptr));
    }
    std::vector<float> probs{};
    probs.reserve(n_game_actions_);
    float n_legal_actions = 0.0f;
    for (bool m : action_mask)
    {
        n_legal_actions += m;
    }
    for (bool m : action_mask)
    {
        probs.emplace_back(float(m) / n_legal_actions);
    }
    psa_[short_state] = probs;
}

void Amcts::add_to_rollouts(const rl::common::IState* state_ptr, std::vector<AmctsInfo> visited_path)
{
    rollouts_.push_back(std::make_tuple(state_ptr, visited_path));
}

std::vector<float> Amcts::get_probs(const rl::common::IState* root_ptr)
{
    if (!root_ptr)
    {
        std::runtime_error("Trying to get probabilities with nullptr state");
    }
    std::string state_short = root_ptr->to_short();
    const auto& actions_visits = nsa_.at(state_short);
    if (temperature_ == 0.0f)
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
            throw std::runtime_error("Amcts no legal actions were provided");
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
        float p = powf(a / sum_action_visits, 1.0f / temperature_);
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

void Amcts::evaluate_collected_states()
{
    if (rollouts_.size() == 0)
    {
        return;
    }
    int n_states = static_cast<int>(rollouts_.size());
    std::vector<const rl::common::IState*> states_ptrs(n_states, nullptr);
    for (int i{ 0 }; i < n_states; i++)
    {
        // states_ptrs.push_back(std::get<0>(rollouts_.at(i)));
        states_ptrs.at(i) = std::get<0>(rollouts_.at(i));
    }
    auto evaluations_tuple = evaluator_ptr_->evaluate(states_ptrs);
    std::vector<float>& probs = std::get<0>(evaluations_tuple);
    std::vector<float>& values = std::get<1>(evaluations_tuple);

    for (int i{ 0 }; i < n_states; i++)
    {
        auto& visited_path = std::get<1>(rollouts_[i]);
        float value = values.at(i);
        backpropogate(visited_path, value, states_ptrs.at(i)->player_turn());
    }

    for (int i{ 0 }; i < n_states; i++)
    {
        int start = i * n_game_actions_;
        auto state_ptr = states_ptrs.at(i);

        // check if we can use the visited_path;
        auto state_short = state_ptr->to_short();
        auto actions_mask = masks_.at(state_short);

        std::vector<float>& tree_probs_ref = psa_.at(state_short);
        float probs_sum{ 0.0f };
        for (int j{ 0 }; j < n_game_actions_; j++)
        {
            float p = probs.at(start + j) * actions_mask.at(j);
            tree_probs_ref.at(j) = p;
            probs_sum += p;
        }
        // normalize tree probs

        for (int j{ 0 }; j < n_game_actions_; j++)
        {
            tree_probs_ref.at(j) /= probs_sum;
        }
    }

    rollouts_.clear();
}

void Amcts::backpropogate(std::vector<AmctsInfo>& visited_path, float final_result, int final_player)
{
    while (visited_path.size() != 0)
    {
        auto& info = visited_path.at(visited_path.size() - 1);
        auto& state_short = info.state_short;
        float score = info.player == final_player ? final_result : -final_result;
        ns_.at(state_short) += 1 - default_n_;
        nsa_.at(state_short).at(info.action) += 1 - default_n_;
        wsa_.at(state_short).at(info.action) += score - default_w_;
        visited_path.pop_back();
    }
}
} // namespace rl::search_trees
