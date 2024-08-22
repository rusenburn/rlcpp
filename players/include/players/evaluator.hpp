#ifndef RL_PLAYERS_EVALUATOR_HPP_
#define RL_PLAYERS_EVALUATOR_HPP_

#include <vector>
#include <tuple>
#include <memory>
#include <common/state.hpp>

namespace rl::players
{
class IEvaluator
{

public:
    virtual ~IEvaluator();

    /// @brief takes a vector of state pointers and returns 2 vectors 
    ///     first vector is a flat vector of all actions probabilities for all states in the vector
    ///     second vector is the evaluation for all states in input vector
    /// @param state_ptrs 
    /// @return 
    virtual std::tuple<std::vector<float>, std::vector<float>> evaluate(const std::vector<const rl::common::IState*>& state_ptrs) = 0;
    virtual std::tuple<std::vector<float>, std::vector<float>> evaluate(const rl::common::IState* state_ptrs) = 0;
    virtual std::tuple<std::vector<float>, std::vector<float>> evaluate(const std::unique_ptr<rl::common::IState>& state_ptrs) = 0;

    virtual std::unique_ptr<IEvaluator> clone() const = 0;
    virtual std::unique_ptr<IEvaluator> copy() const = 0;
};
} // namespace rl::players

#endif