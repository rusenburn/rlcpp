#ifndef RL_COMMON_MATCH_HPP_
#define RL_COMMON_MATCH_HPP_

#include <memory>
#include <tuple>
#include <vector>
#include "state.hpp"
#include "player.hpp"
#include "observer.hpp"
namespace rl::common
{
    class Match
    {
    private:
        std::unique_ptr<IState> initial_state_ptr_;
        std::vector<IPlayer*> players_ptrs_{};
        
        int n_sets_;
        bool render_;
        std::tuple<float, float> play_set(int starting_player);

    public:
        Match(std::unique_ptr<IState> initial_state_ptr,IPlayer* player_1_ptr,IPlayer* player_2_ptr, int n_sets, bool render);
        std::tuple<float, float> start();
        Subject<const IState *> state_changed_event;
        ~Match();
    };

} // namespace rl::common

#endif