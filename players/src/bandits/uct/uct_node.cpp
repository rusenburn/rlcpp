#include <players/bandits/uct/uct_node.hpp>
#include <common/exceptions.hpp>
#include <chrono>
#include <algorithm>
#include <common/random.hpp>
namespace rl::players
{
    UctNode::UctNode(std::unique_ptr<const rl::common::IState> state_ptr, int n_game_actions, float cuct)
        : state_ptr_(std::move(state_ptr)),
          n_game_actions_(n_game_actions),
          cuct_(cuct),
          terminal_(),
          game_result_(),
          actions_legality_(),
          children_(),
          n_(0),
          qsa_(n_game_actions),
          nsa_(n_game_actions)
    {
        for (int i = 0; i < n_game_actions; i++)
        {
            children_.push_back(nullptr);
        }
    }

    void UctNode::search(int minimum_simulations, std::chrono::duration<int, std::milli> minimum_duration, float uct, float temperature, std::vector<float> &out_actions_probs)
    {
        if (terminal_.has_value() == false)
        {
            terminal_.emplace(state_ptr_->is_terminal());
        }
        if (terminal_.value())
        {
            throw rl::common::SteppingTerminalStateException("");
        }

        auto t_end = std::chrono::high_resolution_clock::now() + minimum_duration;
        for (int i = 0; i < minimum_simulations; i++)
        {
            simulateOne();
        }

        while (t_end > std::chrono::high_resolution_clock::now())
        {
            simulateOne();
        }

        getFinalProbabilities(temperature, out_actions_probs);
    }

    std::pair<float, int> UctNode::simulateOne()
    {
        if (terminal_.has_value() == false)
        {
            terminal_.emplace(state_ptr_->is_terminal());
        }

        if (terminal_.value())
        {
            if (game_result_.has_value() == false)
            {
                game_result_.emplace(state_ptr_->get_reward());
            }
            return std::make_pair(game_result_.value(), state_ptr_->player_turn());
        }

        if (actions_legality_.size() == 0) // if size is 0 then this is the first none terminal visit
        {
            actions_legality_ = state_ptr_->actions_mask();
            auto tp = rollout(state_ptr_->clone());
            return tp;
        }

        int best_action = findBestAction();

        if (children_[best_action] == nullptr)
        {
            std::unique_ptr<rl::common::IState> new_state_ptr = state_ptr_->step(best_action);
            std::unique_ptr<UctNode> new_node_ptr = std::make_unique<UctNode>(std::move(new_state_ptr), n_game_actions_, cuct_);
            children_[best_action] = std::move(new_node_ptr);
        }

        auto &new_node_ptr = children_[best_action];
        auto [z, p] = new_node_ptr->simulateOne();
        if (p != state_ptr_->player_turn())
        {
            z = -z;
        }
        n_++;
        nsa_[best_action]++;
        qsa_[best_action] += (z - qsa_[best_action]) / float(nsa_[best_action]);
        return std::make_pair(z, state_ptr_->player_turn());
    }

    int UctNode::findBestAction()
    {
        if (actions_legality_.size() == 0)
        {
            throw "Exception actions legality size should not be 0";
        }

        float max_u = -std::numeric_limits<float>::infinity();
        int best_action = -1;

        for (int action = 0; action < n_game_actions_; action++)
        {
            if (actions_legality_[action] == 0)
            {
                continue;
            }

            float qsa = qsa_[action];
            float nsa = float(nsa_[action]);
            float u;
            if (nsa == 0)
            {
                u = std::numeric_limits<float>::infinity();
            }
            else
            {
                u = qsa + cuct_ * sqrtf(logf(float(n_)) / nsa);
            }
            if (u > max_u)
            {
                max_u = u;
                best_action = action;
            }
        }

        if (best_action != -1)
        {
            return best_action;
        }

        // Should not reach this code unless something went wrong
        // Pick random action instead
        std::vector<int> best_actions{};
        for (int action = 0; action < n_game_actions_; action++)
        {
            if (actions_legality_[action] == 1)
            {
                best_actions.push_back(action);
            }
        }

        best_action = rl::common::get(best_actions.size());
        return best_actions[best_action];
    }

    void UctNode::getFinalProbabilities(float temperature, std::vector<float> &out_actions_probs)
    {
        if (temperature == 0.0f)
        {
            int max_visits_count = *std::max_element(nsa_.begin(), nsa_.end());

            std::vector<int> best_actions{};
            for (int action = 0; action < n_game_actions_; action++)
            {
                if (nsa_[action] == max_visits_count)
                {
                    best_actions.push_back(action);
                }
            }

            int best_action_idx = rl::common::get(best_actions.size());
            int best_action = best_actions[best_action_idx];

            for (int action = 0; action < n_game_actions_; action++)
            {
                out_actions_probs[action] = action == best_action ? 1.0f : 0.0f;
            }
            return;
        }

        float sum_probs = 0;
        for (int action = 0; action < n_game_actions_; action++)
        {
            float prob = float(nsa_[action]) / n_;
            float new_prob = powf(prob, 1.0f / temperature);
            sum_probs += new_prob;
            out_actions_probs[action] = new_prob;
        }

        for (int action = 0; action < n_game_actions_; action++)
        {
            out_actions_probs[action] /= sum_probs;
        }
    }

    std::pair<float, int> UctNode::rollout(std::unique_ptr<rl::common::IState> rollout_state_ptr)
    {
        int depth = 0;
        while (!rollout_state_ptr->is_terminal())
        {
            std::vector<bool> rollout_actions_legality = rollout_state_ptr->actions_mask();

            int n_legal_actions = 0;
            for (int i = 0; i < n_game_actions_; i++)
            {
                n_legal_actions += rollout_actions_legality[i];
            }

            int random_legal_action_idx = rl::common::get(n_legal_actions);
            int random_action = -1;
            while (random_legal_action_idx >= 0)
            {
                if (rollout_actions_legality[++random_action] == 1)
                    random_legal_action_idx--;
            }
            rollout_state_ptr = rollout_state_ptr->step(random_action);
            depth++;
        }
        float result = rollout_state_ptr->get_reward();
        return std::make_pair(result, rollout_state_ptr->player_turn());
    }

    UctNode::~UctNode()
    {
        children_.clear();
        actions_legality_.clear();
        nsa_.clear();
        qsa_.clear();

        children_.shrink_to_fit();
        actions_legality_.shrink_to_fit();
        nsa_.shrink_to_fit();
        qsa_.shrink_to_fit();
    }
} // namespace searchTrees
