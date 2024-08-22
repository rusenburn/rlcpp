#include <players/bandits/grave/g_node.hpp>
#include <common/exceptions.hpp>
#include <common/random.hpp>
namespace rl::players
{
GNode::GNode(std::unique_ptr<IState> state_ptr, int amaf_min_ref_count, float bias, bool save_illegal_actions_amaf)
    : state_ptr_{ std::move(state_ptr) },
    amaf_min_ref_count_{ amaf_min_ref_count },
    bias_{ bias },
    save_illegal_actions_amaf_{ save_illegal_actions_amaf },
    current_player_{ state_ptr_->player_turn() },
    n_game_actions_{ state_ptr_->get_n_actions() },
    is_terminal_{},
    result_{},
    actions_mask_{},
    is_leaf_node_{ true },
    children_(),
    n_{ 0 },
    wsa_(std::vector<float>(n_game_actions_)),
    nsa_(std::vector<int>(n_game_actions_)),
    amaf_wsa_player_0_(std::vector<float>(n_game_actions_)),
    amaf_nsa_player_0_(std::vector<int>(n_game_actions_)),
    amaf_wsa_player_1_(std::vector<float>(n_game_actions_)),
    amaf_nsa_player_1_(std::vector<int>(n_game_actions_))
{

    heuristic();
}
GNode::~GNode() = default;

void GNode::heuristic()
{
    if (is_terminal_.has_value() == false)
    {
        is_terminal_.emplace(state_ptr_->is_terminal());
    }
    if (is_terminal_.value())
    {
        return;
    }
    if (actions_mask_.size() == 0)
    {
        actions_mask_ = state_ptr_->actions_mask();
    }
    if (children_.size() == 0)
    {
        for (int action = 0; action < n_game_actions_; action++)
        {
            children_.push_back(nullptr);

            wsa_[action] = 0;
            nsa_[action] = 0;
            amaf_wsa_player_0_[action] = 0;
            amaf_nsa_player_0_[action] = 50;
            amaf_wsa_player_1_[action] = 0;
            amaf_nsa_player_1_[action] = 50;
        }
    }
}

std::pair<float, int> GNode::playout(std::vector<int>& player_0_actions, std::vector<int>& player_1_actions)
{
    if (is_terminal_.has_value() == false || is_terminal_.value())
    {
        throw rl::common::UnreachableCodeException("state terminal is not calculate or it is done");
    }
    std::unique_ptr<IState> rollout_state_ptr{ state_ptr_->clone() };
    while (rollout_state_ptr->is_terminal() == false)
    {
        std::vector<bool> masks = rollout_state_ptr->actions_mask();

        int n_legal_actions = 0;
        for (int action = 0; action < n_game_actions_; action++)
        {
            n_legal_actions += masks[action];
        }

        int random_legal_action_idx = common::get(n_legal_actions);
        int random_action = -1;
        while (random_legal_action_idx >= 0)
        {
            if (masks[++random_action] == 1)
                random_legal_action_idx--;
        }

        if (rollout_state_ptr->player_turn() == 0)
        {
            player_0_actions.push_back(random_action);
        }
        else
        {
            player_1_actions.push_back(random_action);
        }
        rollout_state_ptr = rollout_state_ptr->step(random_action);
    }
    float reward = rollout_state_ptr->get_reward();
    return std::make_pair(reward, rollout_state_ptr->player_turn());
}

std::pair<float, int> GNode::simulate_one(const GNode* amaf_node_ptr, std::vector<int>& out_player_0_actions, std::vector<int>& out_player_1_actions)
{
    if (is_terminal_.has_value() == false)
    {
        is_terminal_.emplace(state_ptr_->is_terminal());
    }
    if (is_terminal_.value())
    {
        if (result_.has_value() == false)
        {
            result_.emplace(state_ptr_->get_reward());
        }
        return std::make_pair(result_.value(), current_player_);
    }

    if (is_leaf_node_)
    {
        auto playout_res = playout(out_player_0_actions, out_player_1_actions);
        update_amf(playout_res, out_player_0_actions, out_player_1_actions);
        is_leaf_node_ = false;
        return playout_res;
    }
    else
    {
        if (n_ > amaf_min_ref_count_)
        {
            amaf_node_ptr = this;
        }
        int best_action = choose_best_action(amaf_node_ptr);

        if (children_[best_action] == nullptr)
        {
            auto new_state_ptr = state_ptr_->step(best_action);
            children_[best_action] = std::make_unique<GNode>(std::move(new_state_ptr), amaf_min_ref_count_, bias_, save_illegal_actions_amaf_);
        }
        auto& new_node = children_[best_action];
        auto playout_res = new_node->simulate_one(amaf_node_ptr, out_player_0_actions, out_player_1_actions);
        update_amf(playout_res, out_player_0_actions, out_player_1_actions);
        if (current_player_ == 0)
        {
            out_player_0_actions.push_back(best_action);
        }
        else
        {
            out_player_1_actions.push_back(best_action);
        }
        auto& [z, new_player] = playout_res;
        n_++;
        nsa_[best_action]++;
        wsa_[best_action] += current_player_ == new_player ? z : -z;
        return playout_res;
    }
}

int GNode::choose_best_action(const GNode* amaf_ptr)
{
    int best_action = -1;
    float best_value = -std::numeric_limits<float>::infinity();
    for (int action = 0; action < n_game_actions_; action++)
    {
        if (actions_mask_[action] == false)
            continue;

        float w = wsa_[action];
        float n = static_cast<float>(nsa_[action]) + 1e-8f;
        float wa = current_player_ == 0 ? amaf_ptr->amaf_wsa_player_0_[action] : amaf_ptr->amaf_wsa_player_1_[action];
        float na = static_cast<float>(current_player_ == 0 ? amaf_ptr->amaf_nsa_player_0_[action] : amaf_ptr->amaf_nsa_player_1_[action]) + 1e-8f;
        float ba = na / (na + n + bias_ * na * n);
        float amaf = wa / (na + 1e-8f);
        float mean = w / (n + 1e-8f);
        float value = (1.0f - ba) * mean + ba * amaf;
        if (value > best_value)
        {
            best_value = value;
            best_action = action;
        }
    }
    if (best_action == -1)
    {
        throw rl::common::UnreachableCodeException("Best action of -1 in G-rave");
    }
    return best_action;
}

void GNode::update_amf(const std::pair<const float, const int>& playout_res, const std::vector<int>& player_0_actions_ref, const std::vector<int>& player_1_actions_ref)
{
    float z = std::get<0>(playout_res);
    int player = std::get<1>(playout_res);
    float player_0_score = player == 0 ? z : -z;

    for (int action : player_0_actions_ref)
    {
        if (save_illegal_actions_amaf_ == false && actions_mask_[action] == false)
            continue;
        amaf_nsa_player_0_[action]++;
        amaf_wsa_player_0_[action] += player_0_score;
    }
    float player_1_score = player == 1 ? z : -z;
    for (int action : player_1_actions_ref)
    {
        if (save_illegal_actions_amaf_ == false && actions_mask_[action] == false)
            continue;
        amaf_nsa_player_1_[action]++;
        amaf_wsa_player_1_[action] += player_1_score;
    }
}
} // namespace rl::players
