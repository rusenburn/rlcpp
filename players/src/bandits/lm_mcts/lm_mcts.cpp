#include <cassert>
#include <players/bandits/lm_mcts/lm_mcts.hpp>
#include <players/bandits/lm_mcts/lm_mcts_node.hpp>

namespace rl::players
{
    LMMcts::LMMcts(std::unique_ptr<IEvaluator> evaluator_ptr, int n_game_actions, float cpuct, float temperature)
        : evaluator_ptr_{std::move(evaluator_ptr)},
          n_game_actions_{n_game_actions},
          cpuct_{cpuct},
          temperature_{temperature}
    {
    }

    LMMcts::~LMMcts() = default;

    std::vector<float> LMMcts::search(const rl::common::IState *state_ptr, int minimum_no_simulations, std::chrono::duration<int, std::milli> minimum_duration)
    {
        assert(state_ptr->is_terminal() == false);
        auto root_node = std::make_unique<LMMctsNode>(state_ptr->get_n_actions(), cpuct_);
        return root_node->search_and_get_probs(state_ptr->clone(), evaluator_ptr_, minimum_no_simulations, minimum_duration, temperature_);
    }

} // namespace rl::players
