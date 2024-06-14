#include <common/random.hpp>
#include <players/mcts.hpp>
#include <players/mcts_player.hpp>

namespace rl::players
{
    MctsPlayer::MctsPlayer(int n_game_actions,
                           std::unique_ptr<IEvaluator> evaluator_ptr,
                           int minimum_simulations,
                           std::chrono::duration<int, std::milli> duration_in_millis,
                           float temperature,
                           float cpuct)
        : n_game_actions_{n_game_actions},
          evaluator_ptr_{std::move(evaluator_ptr)},
          minimum_simulations_{minimum_simulations},
          duration_in_millis_{duration_in_millis},
          temperature_{temperature},
          cpuct_{cpuct}
    {}

    MctsPlayer::~MctsPlayer() = default;

    int MctsPlayer::choose_action(const std::unique_ptr<rl::common::IState> &state_ptr)
    {
        auto mcts = MCTS(evaluator_ptr_->copy(), n_game_actions_, cpuct_, temperature_);

        std::vector<float> probs = mcts.search(state_ptr.get(), minimum_simulations_, duration_in_millis_);

        float p = rl::common::get();

        float remaining_prob = p;

        int action = 0;

        int last_action = n_game_actions_ - 1;

        // keep decreasing remaining probs until it is below zero or only 1 action remains
        while ((action < last_action) && ((remaining_prob -= probs.at(action)) >= 0))
        {
            action++;
        }

        return action;
    }
} // namespace rl::players
