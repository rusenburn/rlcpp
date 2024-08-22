#include <players/bandits/grave/grave.hpp>
#include <common/exceptions.hpp>
#include <players/bandits/grave/grave_node.hpp>
#include <iostream>
namespace rl::players
{
Grave::Grave(int n_game_actions, int min_ref_count, float b_squared, bool save_illegal_amaf_actions)
    : n_game_actions_(n_game_actions),
    min_ref_count_(min_ref_count),
    b_squared_(b_squared),
    save_illegal_amaf_actions(save_illegal_amaf_actions) {};

std::vector<float> Grave::search(const rl::common::IState* state_ptr, int minimum_no_simulations, std::chrono::duration<int, std::milli> minimum_duration)
{
    if (state_ptr->is_terminal())
    {
        throw rl::common::SteppingTerminalStateException("");
    }
    GraveNode node(state_ptr->clone(), n_game_actions_);
    auto t_end = std::chrono::high_resolution_clock::now() + minimum_duration;

    std::vector<int> out_our_actions{};
    std::vector<int> out_their_actions{};
    std::pair<float, int> pair;
    int i = 0;
    for (; i < minimum_no_simulations; i++)
    {
        pair = node.simulateOne(&node, save_illegal_amaf_actions, 0, out_our_actions, out_their_actions, min_ref_count_, b_squared_);
        out_our_actions.clear();
        out_their_actions.clear();
    }

    while (t_end > std::chrono::high_resolution_clock::now())
    {
        pair = node.simulateOne(&node, save_illegal_amaf_actions, 0, out_their_actions, out_our_actions, min_ref_count_, b_squared_);
        out_our_actions.clear();
        out_their_actions.clear();
        i++;
    }

    int action = node.selectMove(&node, 0, b_squared_);
    std::vector<float> action_probs(n_game_actions_);
    action_probs[action] = 1.0f;
    // std::cout << "Grave " << i << std::endl;
    return action_probs;
}
Grave::~Grave() = default;
}