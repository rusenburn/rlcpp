#include <players/bandits/mcrave/mcrave.hpp>
#include <players/bandits/mcrave/mcrave_node.hpp>

namespace rl::players
{
Mcrave::Mcrave(int n_game_actions, float b)
    : n_game_actions_(n_game_actions),
    b_(b) {}

std::vector<float> Mcrave::search(const rl::common::IState* state_ptr, int minimum_no_simulations, std::chrono::duration<int, std::milli> minimum_duration)
{
    McraveNode node(state_ptr->clone(), n_game_actions_);
    std::vector<float> actions_probs(n_game_actions_);
    node.search(minimum_no_simulations, minimum_duration, b_, actions_probs);
    return actions_probs;
}
Mcrave::~Mcrave() = default;
} // namespace rl::players
