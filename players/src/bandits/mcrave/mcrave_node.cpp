#include <players/bandits/mcrave/mcrave_node.hpp>
#include <common/exceptions.hpp>
#include <chrono>
#include <common/random.hpp>
namespace rl::players
{
McraveNode::McraveNode(std::unique_ptr<IState> state_ptr, int n_game_actions)
    : state_ptr_(std::move(state_ptr)),
    n_game_actions_(n_game_actions),
    player_(state_ptr_->player_turn()),
    terminal_(),
    is_leaf_node(true),
    game_result_(),
    children_(),
    n_sa_(std::vector<int>(n_game_actions)),
    nsa_(std::vector<int>(n_game_actions)),
    q_sa_(std::vector<float>(n_game_actions)),
    qsa_(std::vector<float>(n_game_actions)),
    actions_legality_()
{
    heuristic();
    for (int i = 0; i < n_game_actions; i++)
    {
        children_.push_back(nullptr);
    }
}
void McraveNode::search(int minimum_simulations_count, std::chrono::duration<int, std::milli> minimum_duration, float b, std::vector<float>& out_actions_probs)
{

    if (terminal_.has_value() == false)
    {
        terminal_.emplace(state_ptr_->is_terminal());
    }
    if (terminal_.value())
    {
        throw rl::common::SteppingTerminalStateException("");
    }
    auto t_start = std::chrono::high_resolution_clock::now();
    auto t_end = t_start + minimum_duration;
    std::vector<int> out_our_actions{};
    std::vector<int> out_their_actions{};
    float z;
    int i = 0;
    for (; i < minimum_simulations_count; i++)
    {
        auto [z, p] = simulateOne(0, out_their_actions, out_our_actions);
        out_our_actions.clear();
        out_their_actions.clear();
    }

    while (t_end > std::chrono::high_resolution_clock::now())
    {
        auto [z, p] = simulateOne(0, out_their_actions, out_our_actions);
        out_our_actions.clear();
        out_their_actions.clear();
        i++;
    }

    int action = selectMove(b);
    out_actions_probs.resize(n_game_actions_);
    out_actions_probs[action] = 1;
}

std::pair<float, int> McraveNode::simulateOne(float b, std::vector<int>& out_our_actions, std::vector<int>& out_their_actions)
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

    if (is_leaf_node) // leaf node -> first visit
    {
        if (actions_legality_.size() == 0)
        {
            actions_legality_ = state_ptr_->actions_mask();
        }
        auto [z, p] = simDefault(out_our_actions, out_their_actions);
        is_leaf_node = false;
        // int current_player = state_ptr_->player_turn();
        if (p != player_)
        {
            z = -z;
        }
        for (int& action : out_our_actions)
        {
            if (actions_legality_[action] == 0)
                continue;
            n_sa_[action]++;
            q_sa_[action] += (z - q_sa_[action]) / n_sa_[action];
        }
        return std::make_pair(z, player_);
    }

    int best_action = selectMove(b);

    if (children_[best_action] == nullptr)
    {
        auto new_state_ptr = state_ptr_->step(best_action);
        children_[best_action] = std::make_unique<McraveNode>(std::move(new_state_ptr), n_game_actions_);
    }
    auto& new_node_ptr = children_[best_action];
    int new_player = new_node_ptr->player();
    std::pair<float, int> pair;
    if (new_player != player_)
    {
        pair = new_node_ptr->simulateOne(b, out_their_actions, out_our_actions);
    }
    else
    {
        pair = new_node_ptr->simulateOne(b, out_our_actions, out_their_actions);
    }
    auto [z, p] = pair;
    if (p != player_)
    {
        z = -z;
    }

    for (int& action : out_our_actions)
    {
        if (actions_legality_[action] == 0)
            continue;
        n_sa_[action]++;
        q_sa_[action] += (z - q_sa_[action]) / n_sa_[action];
    }
    out_our_actions.push_back(best_action);

    nsa_[best_action]++;
    qsa_[best_action] += (z - qsa_[best_action]) / nsa_[best_action];
    return std::make_pair(z, player_);
}

int McraveNode::selectMove(float b)
{
    float max_eval = -std::numeric_limits<float>::infinity();
    int best_action = -1;
    for (int action = 0; action < n_game_actions_; action++)
    {
        if (actions_legality_[action] == 0)
            continue;

        float beta = n_sa_[action] / static_cast<float>(nsa_[action] + n_sa_[action] + 4 * b * b * nsa_[action] * n_sa_[action] + 1e-8f);

        float eval = (1 - beta) * qsa_[action] + beta * q_sa_[action];
        if (eval > max_eval)
        {
            max_eval = eval;
            best_action = action;
        }
    }
    if (best_action == -1)
    {
        throw "best action is -1";
    }
    return best_action;
}

std::pair<float, int> McraveNode::simDefault(std::vector<int>& out_our_actions, std::vector<int>& out_their_actions)
{
    if (terminal_.has_value() == false || terminal_.value() == true)
    {
        throw std::runtime_error("terminal status was not calculated or done");
    }
    // if (terminal_ != common::TerminalStatus::NotYet)
    // {
    //     throw "terminal status was not calculated or done";
    // }

    // pick legal action
    int iter = 0;
    int our_player = state_ptr_->player_turn();
    std::unique_ptr<IState> rollout_state_ptr = state_ptr_->clone();
    while (!rollout_state_ptr->is_terminal())
    {
        std::vector<bool> rollout_actions_legality = rollout_state_ptr->actions_mask();

        int n_legal_actions = 0;
        for (int i = 0; i < n_game_actions_; i++)
        {
            n_legal_actions += rollout_actions_legality[i];
        }

        int random_legal_action_idx = rl::common::get(n_legal_actions);
        // int random_legal_action_idx = rand() % n_legal_actions;
        int random_action = -1;
        while (random_legal_action_idx >= 0)
        {
            if (rollout_actions_legality[++random_action] == 1)
                random_legal_action_idx--;
        }
        if (rollout_state_ptr->player_turn() == our_player)
        {
            out_our_actions.push_back(random_action);
        }
        else
        {
            out_their_actions.push_back(random_action);
        }
        rollout_state_ptr = rollout_state_ptr->step(random_action);
        iter++;
    }
    float last_result = rollout_state_ptr->get_reward();
    int last_player = rollout_state_ptr->player_turn();
    if (last_player != our_player)
    {
        last_result = -last_result;
    }
    // relative to the caller 0
    return std::make_pair(last_result, our_player);
}
void McraveNode::heuristic()
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
        if (actions_legality_[action] == 0)
            continue;

        // TODO: change these
        qsa_[action] = 0;
        n_sa_[action] = 0;
        q_sa_[action] = 0;
        n_sa_[action] = 50;
    }
}
int McraveNode::player()
{
    return player_;
}
McraveNode::~McraveNode() = default;

} // namespace rl::players
