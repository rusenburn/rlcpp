#include <cassert>
#include <iostream>
#include <common/exceptions.hpp>
#include <players/bandits/lm_mcts/lm_mcts_node.hpp>
#include <common/utils.hpp>
#include <common/random.hpp>
namespace rl::players
{
    LMMctsNode::LMMctsNode(int n_game_actions, float cpuct)
        : n_game_actions_{n_game_actions}, cpuct_{cpuct},
          legal_actions_{},
          children_{},
          probs_{},
          action_visits_{},
          delta_wins_{},
          is_terminal_{},
          game_result_{}
    {
    }

    LMMctsNode::~LMMctsNode()
    {
        legal_actions_.clear();
        legal_actions_.shrink_to_fit();
        children_.clear();
        children_.shrink_to_fit();
        probs_.clear();
        probs_.shrink_to_fit();
        delta_wins_.clear();
        delta_wins_.shrink_to_fit();
    }

    std::vector<float> LMMctsNode::search_and_get_probs(std::unique_ptr<rl::common::IState> &state_ptr, std::unique_ptr<IEvaluator> &evaluator_ptr, int n_sims, std::chrono::duration<int, std::milli> minimum_duration, float temperature)
    {
        if (is_terminal_.has_value() == false)
        {
            is_terminal_.emplace(state_ptr->is_terminal());
        }
        if (is_terminal_.value())
        {
            throw rl::common::SteppingTerminalStateException("MCTS Node is searching a terminal state which cannot be stepped.");
        }

        // check if it is the root state has only 1 legal action return legal actions

        auto legal_actions = state_ptr->actions_mask();
        int n_legal_actions = 0;
        for (bool m : legal_actions)
        {
            n_legal_actions += m ? 1 : 0;
        }
        if (n_legal_actions == 1)
        {
            std::vector<float> res{};
            res.reserve(n_game_actions_);
            for (bool m : legal_actions)
            {
                res.emplace_back(static_cast<float>(m));
            }
            return res;
        }
        auto t_start = std::chrono::high_resolution_clock::now();
        auto t_end = t_start + minimum_duration;

        int simulation_count{0};

        while (simulation_count <= n_sims)
        {
            search(state_ptr, evaluator_ptr);
            simulation_count++;
        }

        while (t_end > std::chrono::high_resolution_clock::now())
        {
            search(state_ptr, evaluator_ptr);
            simulation_count++;
        }
        std::cout << "LMMCTS " << simulation_count << std::endl;
        return get_probs(temperature);
    }

    std::pair<float, int> LMMctsNode::search(std::unique_ptr<rl::common::IState> &state_ptr, std::unique_ptr<IEvaluator> &evaluator_ptr)
    {
        if (is_terminal_.has_value() == false)
        {
            is_terminal_.emplace(state_ptr->is_terminal());
        }

        if (is_terminal_.value())
        {
            if (game_result_.has_value() == false)
            {
                game_result_.emplace(state_ptr->get_reward());
            }
            return std::make_pair(game_result_.value(), state_ptr->player_turn());
        }

        if (legal_actions_.size() == 0)
        {
            // first visit => rollout
            std::vector<bool> mask = state_ptr->actions_mask();
            for (int a = 0; a < mask.size(); a++)
            {
                if (mask[a])
                {
                    legal_actions_.push_back(a);
                }
            }
            auto [probs_result, wdl] = evaluator_ptr->evaluate(state_ptr);
            auto wins = wdl.at(0);
            float sum_probs = 1e-8f;
            for (int a : legal_actions_)
            {
                children_.push_back(std::unique_ptr<LMMctsNode>(nullptr));
                auto p = probs_result.at(a);
                probs_.push_back(p);
                action_visits_.push_back(0.0f);
                delta_wins_.push_back(0.0f);
                sum_probs += p;
            }
            // normalize probs
            for (auto &p : probs_)
            {
                p /= sum_probs;
            }
            return std::make_pair(wins, state_ptr->player_turn());
        }

        // already expanded and not a terminal state
        auto [best_action, best_action_index] = get_best_action_and_index();

        if (children_.at(best_action_index) == nullptr)
        {
            children_.at(best_action_index) = std::make_unique<LMMctsNode>(n_game_actions_, cpuct_);
        }

        const auto &next_node = children_.at(best_action_index);
        assert(next_node.get() != nullptr);

        auto next_state = state_ptr->step(best_action);
        auto [next_result, next_player] = next_node->search(next_state, evaluator_ptr);

        float relative_result;
        if (next_player != state_ptr->player_turn())
        {
            relative_result = -next_result;
        }
        else
        {
            relative_result = next_result;
        }

        delta_wins_.at(best_action_index) += relative_result;
        action_visits_.at(best_action_index) += 1;
        n_visits_++;
        return std::make_pair(relative_result, state_ptr->player_turn());
    }

    std::vector<float> LMMctsNode::get_probs(float temperature)
    {
        std::vector<float> probs_with_temperature(n_game_actions_, 0.0f);

        float sum_action_visits{0.0f};
        for (int index = 0; index < legal_actions_.size(); index++)
        {
            int action = legal_actions_.at(index);
            float visits = action_visits_.at(index);
            probs_with_temperature.at(action) = static_cast<float>(visits);
        }
        rl::common::utils::apply_temperature(probs_with_temperature, temperature);
        return probs_with_temperature;
    }

    std::pair<int, int> LMMctsNode::get_best_action_and_index()
    {
        assert(legal_actions_.size() != 0);

        float max_u = -INFINITY;
        int best_action_index = -1;
        int best_action = -1;
        const int n_legal_actions = legal_actions_.size();
        for (int index = 0; index < n_legal_actions; index++)
        {
            float a_visits = action_visits_.at(index);
            float prob = probs_.at(index);
            float qsa = 0.0f;
            if (a_visits > 0)
            {
                qsa = delta_wins_.at(index) / a_visits;
            }
            float u = qsa + cpuct_ * prob * sqrtf(n_visits_ + 1e-8f) / (1 + a_visits);

            if (u > max_u)
            {
                int action = legal_actions_.at(index);
                max_u = u;
                best_action = action;
                best_action_index = index;
            }
        }

        if (best_action == -1)
        {
            int best_action_index = rl::common::get(legal_actions_.size());
            best_action = legal_actions_.at(best_action_index);
        }

        assert(legal_actions_[best_action_index] == best_action);

        return std::make_pair(best_action, best_action_index);
    }

} // namespace rl::players
