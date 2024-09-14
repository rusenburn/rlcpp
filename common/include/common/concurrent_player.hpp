#ifndef RL_COMMON_CONCURRENT_PLAYER_HPP_
#define RL_COMMON_CONCURRENT_PLAYER_HPP_

#include "player.hpp"
#include <vector>
namespace rl::common
{
class IConcurrentPlayer : public IPlayer
{
public:
    virtual ~IConcurrentPlayer()override;
    virtual std::vector<int> choose_actions(const std::vector<const rl::common::IState*>& states_ptrs_ref) = 0;
    
};
} // namespace rl::common


#endif