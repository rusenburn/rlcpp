#include <cmath>
#include <stdexcept>
#include <iostream>
#include <common/random.hpp>
#include <deeplearning/alphazero/alphazero_sub_tree2.hpp>

namespace rl::deeplearning
{
    constexpr float EPS = 1e-8f;
    AmctsSubTree2::AmctsSubTree2(int n_game_actions,
                               float cpuct,
                               float temperature,
                               float default_visits,
                               float default_wins)
        : n_game_actions_{n_game_actions},
          default_n_{default_visits},
          default_w_{default_wins},
          cpuct_{cpuct},
          temperature_{temperature},
          states_{},
          values_{},
          rollouts_{}
    {
    }
    AmctsSubTree2::~AmctsSubTree2() = default;

    void AmctsSubTree2::roll(const rl::common::IState *root_ptr)
    {
        std::vector<AmctsInfo2> visited_path{};
        simulate_once(root_ptr, visited_path);
    }

    std::vector<const rl::common::IState *> AmctsSubTree2::get_rollouts()
    {
        std::vector<const rl::common::IState *> res{};
        for (int i = 0; i < rollouts_.size(); i++)
        {
            res.push_back(std::get<0>(rollouts_.at(i)));
        }
        return res;
    }

    void AmctsSubTree2::simulate_once(const rl::common::IState *state_ptr, std::vector<AmctsInfo2> &visited_path)
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
            MapValues &state_values = values_.at(short_state);
            const std::vector<bool> &actions_mask = state_values.masks;

            // check if it has more than  1 legal action
            int n_legal_actions{0};
            for (bool m : actions_mask)
            {
                n_legal_actions += static_cast<int>(m);
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
                state_values.psa = probs;
                // psa_.at(short_state) = probs;
            }
            else
            {
                add_to_rollouts(state_ptr, visited_path);
                return;
            }
        }

        // continue down the tree

        MapValues &state_value = values_.at(short_state);
        int best_action = find_best_action(state_value);

        if (!state_value.edges.at(best_action)) // check if edge of best action is null
        {
            state_value.edges.at(best_action) = state_ptr->step(best_action);
        }

        auto new_state_ptr = state_value.edges.at(best_action).get();

        if (new_state_ptr == nullptr)
        {
            throw std::runtime_error("new state was found to be null");
        }

        visited_path.push_back({short_state, best_action, player});

        state_value.nsa.at(best_action) += default_n_;
        state_value.ns += default_n_;
        state_value.ws += default_w_;
        state_value.wsa.at(best_action) += default_w_;
        
        simulate_once(new_state_ptr, visited_path);
    }

    int AmctsSubTree2::find_best_action(MapValues &state_values)
    {
        float max_u = -INFINITY;
        int best_action = -1;

        const auto &wsa_vec = state_values.wsa;
        const auto &nsa_vec = state_values.nsa;
        const auto &psa_vec = state_values.psa;
        const auto &masks = state_values.masks;
        float current_state_visis = state_values.ns;

        for (int action{0}; action < masks.size(); action++)
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
    void AmctsSubTree2::expand_state(const rl::common::IState *state_ptr, std::string &short_state)
    {
        if (state_ptr->is_terminal())
        {
            throw std::runtime_error("Expanding a terminal state");
        }
        states_.insert(short_state);

        std::vector<bool> action_mask = state_ptr->actions_mask();
        auto ns = 0;
        auto nsa = std::vector<float>(n_game_actions_, 0.0f);
        auto wsa = std::vector<float>(n_game_actions_, 0.0f);
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
        auto& a = (values_[short_state] = {});
        a.masks = action_mask;
        a.ns = ns;
        a.ws = 0;
        a.nsa = nsa;
        a.wsa = wsa;
        a.psa = probs;

        auto &vec = a.edges;
        vec.clear();
        for (int i{0}; i < n_game_actions_; i++)
        {
            vec.push_back(std::unique_ptr<rl::common::IState>(nullptr));
        }
    }

    void AmctsSubTree2::add_to_rollouts(const rl::common::IState *state_ptr, std::vector<AmctsInfo2> visited_path)
    {
        rollouts_.push_back(std::make_tuple(state_ptr, visited_path));
    }

    std::vector<float> AmctsSubTree2::get_probs(const rl::common::IState *root_ptr)
    {
        if (!root_ptr)
        {
            std::runtime_error("Trying to get probabilities with nullptr state");
        }
        std::string state_short = root_ptr->to_short();
        auto& state_values = values_.at(state_short);
        const auto &actions_visits = state_values.nsa;
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
            for (int action{0}; action < n_game_actions_; action++)
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
            for (int action{0}; action < n_game_actions_; action++)
            {
                probs.at(action) /= sum_probs;
            }
            return probs;
        }
        std::vector<float> probs{};
        probs.reserve(n_game_actions_);
        float sum_visits{0.0f};

        for (auto &a : actions_visits)
        {
            sum_visits += a;
        }
        for (auto &a : actions_visits)
        {
            float p = a / sum_visits;
            probs.emplace_back(p);
        }
        std::vector<float> probs_with_temp{};

        probs_with_temp.reserve(n_game_actions_);
        float sum_probs_with_temp{0.0f};

        for (auto p : probs)
        {
            float p_t = powf(p, 1 / temperature_);
            probs_with_temp.emplace_back(p_t);
            sum_probs_with_temp += p_t;
        }
        for (auto &p_t : probs_with_temp)
        {
            p_t = p_t / sum_probs_with_temp;
        }

        float sum_probs{0.0};
        for (auto p_t : probs_with_temp)
        {
            sum_probs += p_t;
        }

        if (sum_probs < 1.0f - 1e-3f || sum_probs > 1.0f + 1e-3f)
        {
            std::runtime_error("action probabilities do not equal to one");
        }

        if (sum_probs == NAN)
        {
            std::cout << "NAN probs are found returning probs";
            return probs;
        }
        else
        {
            return probs_with_temp;
        }
    }

    float AmctsSubTree2::get_evaluation(const rl::common::IState *root_ptr)
    {
        if (!root_ptr)
        {
            std::runtime_error("Trying to get evaluation of nullptr state");
        }
        
        std::string state_short = root_ptr->to_short();
        auto &state_value_ref = values_.at(state_short);
        float n = state_value_ref.ns;
        if (n <= 0)
        {
            std::runtime_error("Trying to get evaluation of with zero or negative total visits");
        }
        float w = state_value_ref.ws;
        // float win_ratio = (w + n) / (2 * n);
        return w/n;
    }
    void AmctsSubTree2::evaluate_collected_states(std::tuple<std::vector<float>, std::vector<float>> &evaluations_tuple)
    {
        int n_states = static_cast<int>(rollouts_.size());
        std::vector<const rl::common::IState *> states_ptrs(n_states, nullptr);
        for (int i{0}; i < n_states; i++)
        {
            // states_ptrs.push_back(std::get<0>(rollouts_.at(i)));
            states_ptrs.at(i) = std::get<0>(rollouts_.at(i));
        }
        std::vector<float> &probs = std::get<0>(evaluations_tuple);
        std::vector<float> &values = std::get<1>(evaluations_tuple);

        for (int i{0}; i < n_states; i++)
        {
            auto &visited_path = std::get<1>(rollouts_[i]);
            float value = values.at(i);
            backpropogate(visited_path, value, states_ptrs.at(i)->player_turn());
        }

        for (int i{0}; i < n_states; i++)
        {
            int start = i * n_game_actions_;
            auto state_ptr = states_ptrs.at(i);

            // check if we can use the visited_path;
            auto state_short = state_ptr->to_short();
            auto &state_value_ref = values_.at(state_short);
            const auto actions_mask = state_value_ref.masks;

            std::vector<float> &tree_probs_ref = state_value_ref.psa;
            float probs_sum{0.0f};
            for (int j{0}; j < n_game_actions_; j++)
            {
                float p = probs.at(start + j) * actions_mask.at(j);
                tree_probs_ref.at(j) = p;
                probs_sum += p;
            }
            // normalize tree probs

            for (int j{0}; j < n_game_actions_; j++)
            {
                tree_probs_ref.at(j) /= probs_sum;
            }
        }

        rollouts_.clear();
    }

    void AmctsSubTree2::backpropogate(std::vector<AmctsInfo2> &visited_path, float final_result, int final_player)
    {
        while (visited_path.size() != 0)
        {
            auto &info = visited_path.at(visited_path.size() - 1);
            auto &state_short = info.state_short;
            float score = info.player == final_player ? final_result : -final_result;
            auto &state_value_ref = values_.at(state_short);

            state_value_ref.ns += 1 - default_n_;
            state_value_ref.nsa.at(info.action) += 1 - default_n_;
            state_value_ref.wsa.at(info.action) += score - default_w_;
            state_value_ref.ws +=score - default_w_;
            visited_path.pop_back();
        }
    }
} // namespace rl::search_trees
