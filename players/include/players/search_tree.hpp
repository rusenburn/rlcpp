#ifndef RL_PLAYERS_SEARCH_TREE_HPP_
#define RL_PLAYERS_SEARCH_TREE_HPP_

#include <vector>
#include <memory>
#include <chrono>
#include <common/state.hpp>

namespace rl::players
{
    class ISearchTree
    {
    public:
        virtual ~ISearchTree();
        virtual std::vector<float> search(const rl::common::IState *state_ptr, int minimum_no_simulations, std::chrono::duration<int, std::milli> minimum_duration) = 0;
        std::vector<float> search(const rl::common::IState *state_ptr, int minimum_no_simulations);
        std::vector<float> search(const rl::common::IState *state_ptr, std::chrono::duration<int, std::milli> minimum_duration);
    };

} // namespace rl::common

#endif