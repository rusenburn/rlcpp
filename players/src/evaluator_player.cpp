#include <common/random.hpp>
#include <players/evaluator_player.hpp>

namespace rl::players
{
    EvaluatorPlayer::EvaluatorPlayer(std::unique_ptr<IEvaluator> evaluator_ptr)
        : evaluator_ptr_{std::move(evaluator_ptr)}
    {}

    EvaluatorPlayer::~EvaluatorPlayer() = default;

    int EvaluatorPlayer::choose_action(const std::unique_ptr<rl::common::IState> &state_ptr)
    {
        int n_game_actions = state_ptr->get_n_actions();

        auto [probs, vs] = evaluator_ptr_->evaluate(state_ptr);

        float p = rl::common::get();

        float remaining_prob = p;

        int action = 0;

        int last_action = n_game_actions - 1;

        // keep decreasing remaining probs until it is below zero or only 1 action remains
        while ((action < last_action) && ((remaining_prob -= probs.at(action)) >= 0))
        {
            action++;
        }

        return action;
    }
}