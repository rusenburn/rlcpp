#ifndef RL_PLAYERS_EVALUATOR_PLAYER_HPP_
#define RL_PLAYERS_EVALUATOR_PLAYER_HPP_

#include <common/match.hpp>
#include <common/player.hpp>
#include "evaluator.hpp"
#include <memory>
namespace rl::players
{
class EvaluatorPlayer : public rl::common::IPlayer
{
private:
    std::unique_ptr<IEvaluator> evaluator_ptr_;

public:
    EvaluatorPlayer(std::unique_ptr<IEvaluator> evaluator_ptr);
    ~EvaluatorPlayer() override;
    int choose_action(const std::unique_ptr<rl::common::IState>& state_ptr);
};

} // namespace rl::players

#endif