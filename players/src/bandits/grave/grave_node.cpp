#include <players/bandits/grave/grave_node.hpp>
#include <common/exceptions.hpp>
#include <common/random.hpp>
namespace rl::players
{
    GraveNode::GraveNode(std::unique_ptr<IState> state_ptr, int n_game_actions)
        : state_ptr_(std::move(state_ptr)),
          n_game_actions_(n_game_actions),
          current_player_(state_ptr_->player_turn()),
          terminal_(),
          is_leaf_node_(true),
          children_(),
          n_(0),
          wsa_(std::vector<float>(n_game_actions)),
          nsa_(std::vector<int>(n_game_actions)),
          amaf_wsa_player_0(std::vector<float>(n_game_actions)),
          amaf_nsa_player_0(std::vector<int>(n_game_actions)),
          amaf_wsa_player_1(std::vector<float>(n_game_actions)),
          amaf_nsa_player_1(std::vector<int>(n_game_actions)),
          actions_legality_()
    {
        for (int i = 0; i < n_game_actions; i++)
        {
            children_.push_back(nullptr);
        }
        heuristic();
    }
    std::pair<float, int> GraveNode::simulateOne(GraveNode *amaf_node_ptr,
                                                 bool save_illegal_amaf_actions,
                                                 int depth,
                                                 std::vector<int> &out_our_actions,
                                                 std::vector<int> &out_opponent_actions,
                                                 const int &amaf_min_ref_count,
                                                 const float &b_square_ref)
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
            return std::make_pair(game_result_.value(), current_player_);
        }
        if (is_leaf_node_) // leaf node first non terminal visit
        {
            if (actions_legality_.size() == 0)
            {
                actions_legality_ = state_ptr_->actions_mask();
            }
            auto [z, p] = playout(out_our_actions, out_opponent_actions);
            if (p != current_player_)
            {
                z = -z;
            }
            is_leaf_node_ = false;

            update_amaf(z, out_our_actions, out_opponent_actions, depth, save_illegal_amaf_actions);
            return std::make_pair(z, current_player_);
        }
        if (n_ >= amaf_min_ref_count)
        {
            amaf_node_ptr = this;
        }
        int best_action = selectMove(amaf_node_ptr, depth, b_square_ref);

        if (children_[best_action] == nullptr)
        {
            std::unique_ptr<IState> new_state_ptr = state_ptr_->step(best_action);
            children_[best_action] = std::make_unique<GraveNode>(std::move(new_state_ptr), n_game_actions_);
        }
        std::unique_ptr<GraveNode> &new_node_ptr = children_[best_action];
        int new_player = new_node_ptr->current_player_;
        std::pair<float, int> pair;
        if (current_player_ != new_player)
        {
            pair = new_node_ptr->simulateOne(amaf_node_ptr, save_illegal_amaf_actions, depth + 1, out_opponent_actions, out_our_actions, amaf_min_ref_count, b_square_ref);
        }
        else
        {
            pair = new_node_ptr->simulateOne(amaf_node_ptr, save_illegal_amaf_actions, depth + 1, out_our_actions, out_opponent_actions, amaf_min_ref_count, b_square_ref);
        }
        auto [z, p] = pair;
        if (p != current_player_)
        {
            z = -z;
        }
        update_amaf(z, out_our_actions, out_opponent_actions, depth, save_illegal_amaf_actions);

        out_our_actions.push_back(best_action);
        n_++;
        nsa_[best_action]++;
        wsa_[best_action] += z;
        return std::make_pair(z, current_player_);
    }

    int GraveNode::selectMove(GraveNode *amaf_node_ptr, int depth, const float &b_square_ref)
    {
        int best_action = -1;
        float best_value = -std::numeric_limits<float>::infinity();
        for (int action = 0; action < n_game_actions_; action++)
        {
            if (actions_legality_[action] == 0)
            {
                continue;
            }
            float w = wsa_[action];
            float p = float(nsa_[action]);
            float wa = 0.0f;
            float pa = 1e-8f;
            if (amaf_node_ptr != nullptr)
            {
                wa = current_player_ == 0 ? amaf_wsa_player_0[action] : amaf_wsa_player_1[action];
                pa = static_cast<float>(current_player_ == 0 ? amaf_nsa_player_0[action] : amaf_nsa_player_1[action]);
            }
            float beta_action = pa / (pa + p + b_square_ref * pa * p);
            float amaf = wa / (pa + 1e-8f);
            float mean = w / (p + 1e-8f);
            float value = (1.0f - beta_action) * mean + beta_action * amaf;
            if (value > best_value)
            {
                best_value = value;
                best_action = action;
            }
        }
        return best_action;
    }

    std::pair<float, int> GraveNode::playout(std::vector<int> &out_our_actions, std::vector<int> &out_opponent_actions)
    {
        if (terminal_.has_value() == false || terminal_.value())
        {
            throw std::runtime_error("terminal status was not calculated or done");
        }

        // pick legal action
        int iter = 0;
        std::unique_ptr<IState> rollout_state_ptr = state_ptr_->clone();
        while (!rollout_state_ptr->is_terminal())
        {
            std::vector<bool> rollout_actions_legality = rollout_state_ptr->actions_mask();

            int n_legal_actions = 0;
            for (int action = 0; action < n_game_actions_; action++)
            {
                n_legal_actions += rollout_actions_legality[action];
            }

            int random_legal_action_idx = common::get(n_legal_actions);
            int random_action = -1;
            while (random_legal_action_idx >= 0)
            {
                if (rollout_actions_legality[++random_action] == 1)
                    random_legal_action_idx--;
            }
            if (rollout_state_ptr->player_turn() == current_player_)
            {
                out_our_actions.push_back(random_action);
            }
            else
            {
                out_opponent_actions.push_back(random_action);
            }
            rollout_state_ptr = rollout_state_ptr->step(random_action);
            iter++;
        }
        float final_result = rollout_state_ptr->get_reward();
        if (rollout_state_ptr->player_turn() != current_player_)
        {
            final_result = -final_result;
        }
        // relative to the caller 0
        return std::make_pair(final_result, current_player_);
    }
    void GraveNode::heuristic()
    {
        if (terminal_.has_value() == false)
        {
            terminal_.emplace(state_ptr_->is_terminal());
        }
        if (terminal_.value())
        {
            return;
        }
        if (actions_legality_.size() == 0)
        {
            actions_legality_ = state_ptr_->actions_mask();
        }

        for (int action = 0; action < n_game_actions_; action++)
        {
            // if (actions_legality_[action] == 0)
            //         continue;

            wsa_[action] = 0;
            nsa_[action] = 0;
            amaf_wsa_player_0[action] = 0;
            amaf_nsa_player_0[action] = 50;
            amaf_wsa_player_1[action] = 0;
            amaf_nsa_player_1[action] = 50;
        }
    }

    void GraveNode::update_amaf(float our_score, const std::vector<int> &our_actions_ref, const std::vector<int> &opponent_actions_ref, int depth, bool save_illegal_amaf_actions)
    {
        for (int action : our_actions_ref)
        {
            if (!save_illegal_amaf_actions && actions_legality_[action] == 0)
                continue;
            if (current_player_ == 0)
            {
                amaf_nsa_player_0[action]++;
                amaf_wsa_player_0[action] += our_score;
            }
            else
            {
                amaf_nsa_player_1[action]++;
                amaf_wsa_player_1[action] += our_score;
            }
        }

        for (int action : opponent_actions_ref)
        {
            if (!save_illegal_amaf_actions && actions_legality_[action] == 0)
                continue;
            if (current_player_ == 0)
            {
                amaf_nsa_player_1[action]++;
                amaf_wsa_player_1[action] -= our_score;
            }
            else
            {
                amaf_nsa_player_0[action]++;
                amaf_wsa_player_0[action] -= our_score;
            }
        }
    }
    GraveNode::~GraveNode() = default;

}