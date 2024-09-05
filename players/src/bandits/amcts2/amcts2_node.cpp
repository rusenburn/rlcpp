#include <players/bandits/amcts2/amcts2_node.hpp>
#include <stdexcept>
#include <common/utils.hpp>
#include <common/random.hpp>
#include <iostream>
namespace rl::players
{

constexpr float EPS = 1e-8f;
Amcts2Node::Amcts2Node(std::unique_ptr<rl::common::IState> state_ptr, int n_game_actions, float cpuct)
    :state_ptr_{ std::move(state_ptr) }, n_game_actions_{ n_game_actions }, cpuct_{ cpuct }, children_{}
{
}



Amcts2Node::~Amcts2Node() = default;
void Amcts2Node::simulate_once(std::pair<rl::common::IState*, std::vector<Amcts2Info>> &rollout_info_ref, bool use_dirichlet_noise, float default_n, float default_w,Amcts2Node* const root_node_ptr)
{
    if (state_ptr_->is_terminal())
    {
        // state is terminal then get result
        float result = state_ptr_->get_reward();
        int player = state_ptr_->player_turn();
        std::vector<float> emtpy_probs{};
        root_node_ptr->backpropogate(std::get<1>(rollout_info_ref), 0, result, player, emtpy_probs, default_n, default_w);
        return;
    }

    if (children_.size() == 0) // first visit , then expand node
    {
        expand_node();

        rollout_info_ref.first = state_ptr_.get();
        return;
    }

    // continue down the tree

    int best_action = find_best_action(use_dirichlet_noise);

    if (children_.at(best_action) == nullptr)
    {
        children_.at(best_action) = std::make_unique<Amcts2Node>(state_ptr_->step(best_action), n_game_actions_, cpuct_);
    }

    auto& next_node_ptr = children_.at(best_action);

    if (next_node_ptr == nullptr)
    {
        throw std::runtime_error("new node was found to be null");
    }

    rollout_info_ref.second.push_back({ best_action,next_node_ptr->state_ptr_->player_turn() });


    n_visits_ += default_n;
    delta_wins += default_w;
    actions_visits_.at(best_action) += default_n;
    delta_actions_wins.at(best_action) += default_w;

    next_node_ptr->simulate_once(rollout_info_ref, false, default_n, default_w,root_node_ptr);
}

void players::Amcts2Node::backpropogate(std::vector<Amcts2Info>& visited_path, int depth, float final_result, int final_player, std::vector<float>& probs, float default_n, float default_w)
{
    if (visited_path.size() == depth)
    {
        if (probs.size() == 0)
            return;

        // probs_.clear();
        probs_ = {};
        probs_.reserve(n_game_actions_);
        float probs_sum{ 0 };
        for (int i = 0;i < n_game_actions_;i++)
        {
            float p = probs.at(i) * static_cast<float>(actions_mask_.at(i));
            probs_.emplace_back(p);
            probs_sum += p;
        }

        // normalize probs
        rl::common::utils::normalize_vector(probs_);
        return;
    }

    int current_player = state_ptr_->player_turn();
    int visited_action = visited_path.at(depth).action;

    float score = current_player == final_player ? final_result : -final_result;
    n_visits_ += 1 - default_n;
    actions_visits_.at(visited_action) += 1 - default_n;
    delta_wins += score - default_w;
    delta_actions_wins.at(visited_action) += score - default_w;
    auto& child = children_.at(visited_action);
    child->backpropogate(visited_path, depth + 1, final_result, final_player, probs, default_n, default_w);
}

void players::Amcts2Node::expand_node()
{
    if (state_ptr_->is_terminal())
    {
        throw std::runtime_error("Expanding a terminal state");
    }

    actions_mask_ = state_ptr_->actions_mask();
    n_visits_ = 0;
    actions_visits_.resize(n_game_actions_, 0.0f);
    delta_actions_wins.resize(n_game_actions_, 0.0f);
    probs_.clear();
    probs_.reserve(n_game_actions_);
    float n_legal_actions{ 0.0f };
    for (bool m : actions_mask_)
    {
        n_legal_actions += m;
    }

    for (bool m : actions_mask_)
    {
        probs_.emplace_back(static_cast<float>(m) / n_legal_actions);
    }

    children_.clear();
    for (int i = 0;i < n_game_actions_;i++)
    {
        children_.push_back(nullptr);
    }
}

std::vector<float> players::Amcts2Node::get_probs(float temperature)
{
    if (!state_ptr_)
    {
        std::runtime_error("Trying to get probabilities with nullptr state");
    }

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
            throw std::runtime_error("Amcts no legal actions were provided");
        }
        // normalize probs
        for (int action{ 0 }; action < n_game_actions_; action++)
        {
            probs.at(action) /= sum_probs;
        }
        return probs;
    }

    std::vector<float> probs{};
    probs.reserve(n_game_actions_);
    float sum_visits{ 0.0f };

    for (auto& a : actions_visits)
    {
        sum_visits += a;
    }
    for (auto& a : actions_visits)
    {
        float p = a / sum_visits;
        probs.emplace_back(p);
    }
    std::vector<float> probs_with_temp{};

    probs_with_temp.reserve(n_game_actions_);
    float sum_probs_with_temp{ 0.0f };

    for (auto p : probs)
    {
        float p_t = powf(p, 1 / temperature);
        probs_with_temp.emplace_back(p_t);
        sum_probs_with_temp += p_t;
    }
    for (auto& p_t : probs_with_temp)
    {
        p_t = p_t / sum_probs_with_temp;
    }

    float sum_probs{ 0.0 };
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

float players::Amcts2Node::get_evaluation()
{
    if (!state_ptr_)
    {
        std::runtime_error("Trying to get evaluation of nullptr state");
    }

    return delta_wins / n_visits_;
}

int players::Amcts2Node::find_best_action(bool use_dirichlet_noise)
{
    float max_u = -INFINITY;
    int best_action = -1;
    const auto& wsa_vec = delta_actions_wins;
    const auto& nsa_vec = actions_visits_;
    const auto& psa_vec = probs_;
    auto& dirichlet_noise = dirichlet_noise_;
    const auto& masks = actions_mask_;
    float current_state_visis = n_visits_;
    if (use_dirichlet_noise && dirichlet_noise.size() != psa_vec.size())
    {
        dirichlet_noise = rl::common::utils::get_dirichlet_noise(masks, -1.0, rl::common::mt);
    }

    for (int action{ 0 }; action < masks.size(); action++)
    {
        bool is_legal = masks.at(action);
        if (!is_legal)
        {
            continue;
        }
        float action_prob = psa_vec.at(action);
        if (use_dirichlet_noise)
        {
            constexpr float dirichlet_weighting_average = 0.25;
            action_prob = (1 - dirichlet_weighting_average) * action_prob + dirichlet_noise.at(action) * dirichlet_weighting_average;
        }
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



} // namespace rl::players




