#include <players/amcts_player.hpp>

#include <players/amcts_player.hpp>
#include <players/amcts.hpp>
#include <common/random.hpp>
namespace rl::players
{
AmctsPlayer::AmctsPlayer(int n_game_actions,
    std::unique_ptr<IEvaluator> evaluator_ptr,
    int minimum_simulations,
    std::chrono::duration<int, std::milli> duration_in_millis,
    float temperature,
    float cpuct,
    int max_async_simulations,
    float default_visits,
    float default_wins)
    : n_game_actions_{ n_game_actions },
    evaluator_ptr_{ std::move(evaluator_ptr) },
    minimum_simulations_{ minimum_simulations },
    duration_in_millis_{ duration_in_millis },
    temperature_{ temperature },
    cpuct_{ cpuct },
    max_async_simulations_{ max_async_simulations },
    default_visits_{ default_visits },
    default_wins_{ default_wins }
{
}

AmctsPlayer::~AmctsPlayer() = default;

int AmctsPlayer::choose_action(const std::unique_ptr<rl::common::IState>& state_ptr)
{
    auto mcts = Amcts(n_game_actions_, evaluator_ptr_->copy(), cpuct_, temperature_, max_async_simulations_, default_visits_, default_wins_);
    std::vector<float> probs = mcts.search(state_ptr.get(), minimum_simulations_, duration_in_millis_);

    // float p = rand() / static_cast<float>(RAND_MAX + 1);
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
