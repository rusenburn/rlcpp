#include <players/bandits/uct/uct.hpp>
#include <players/bandits/uct/uct_node.hpp>

namespace rl::players
{
    UctSearchTree::UctSearchTree(int n_game_actions, float cuct, float temperature)
        : n_game_actions_(n_game_actions),
          cuct_(cuct),
          temperature_(temperature)
    {}
    std::vector<float> UctSearchTree::search(const rl::common::IState *state_ptr, int simulation_count, std::chrono::duration<int, std::milli> duration)
    {
        UctNode node{state_ptr->clone(), n_game_actions_, cuct_};
        std::vector<float> actions_probs(n_game_actions_);
        node.search(simulation_count, duration, cuct_, temperature_, actions_probs);
        return actions_probs;
    }
} // namespace rl::players
