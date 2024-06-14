#ifndef RL_COMMON_PLAYER_HPP_
#define RL_COMMON_PLAYER_HPP_

#include "state.hpp"
namespace rl::common
{
    class IPlayer
    {
    public:
        virtual ~IPlayer();
        /// @brief takes a unique state pointer reference and returns an int action chosen by the player
        /// @param state_ptr 
        /// @return int action to be performed by the state
        virtual int choose_action(const std::unique_ptr<rl::common::IState> &state_ptr) = 0;
    };
} // namespace rl::common

#endif