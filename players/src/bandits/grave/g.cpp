#include <chrono>
#include <iostream>
#include <players/bandits/grave/g.hpp>
#include <players/bandits/grave/g_node.hpp>
#include <common/exceptions.hpp>

namespace rl::players
{
G::G(int n_game_actions, int min_ref_count, float bias, bool save_illegal_actions_amaf)
    : n_game_actions_{ n_game_actions },
    min_ref_count_{ min_ref_count },
    bias_{ bias },
    save_illegal_amaf_actions_{ save_illegal_actions_amaf }
{
}
G::~G() = default;
std::vector<float> G::search(const rl::common::IState* state_ptr, int minimum_no_simulations, std::chrono::duration<int, std::milli> minimum_duration)
{
    if (state_ptr->is_terminal())
    {
        throw rl::common::SteppingTerminalStateException("");
    }
    GNode root_node{ state_ptr->clone(), min_ref_count_, bias_, save_illegal_amaf_actions_ };

    auto t_end = std::chrono::high_resolution_clock::now() + minimum_duration;

    std::vector<int> player_0_actions{};
    std::vector<int> player_1_actions{};
    std::pair<float, int> pair;
    int i = 0;
    for (; i < minimum_no_simulations; i++)
    {
        pair = root_node.simulate_one(&root_node, player_0_actions, player_1_actions);
        player_0_actions.clear();
        player_1_actions.clear();
    }

    while (t_end > std::chrono::high_resolution_clock::now())
    {
        pair = root_node.simulate_one(&root_node, player_0_actions, player_1_actions);
        player_0_actions.clear();
        player_1_actions.clear();
        i++;
    }

    int action = root_node.choose_best_action(&root_node);
    std::vector<float> action_probs(n_game_actions_);
    action_probs[action] = 1.0f;
    // std::cout << "G " << i << std::endl;
    return action_probs;
}
} // namespace rl::players
