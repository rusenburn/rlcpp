#ifndef RL_COMMON_MATCH_HPP_
#define RL_COMMON_MATCH_HPP_

#include <memory>
#include <tuple>
#include <vector>
#include "state.hpp"
#include "player.hpp"
namespace rl::common
{
    class Match
    {
    private:
        std::unique_ptr<IState> initial_state_ptr_;
        std::vector<std::unique_ptr<IPlayer>> players_ptrs_{};
        int n_sets_;
        bool render_;
        std::tuple<float, float> play_set(int starting_player);

    public:
        Match(std::unique_ptr<IState> initial_state_ptr, std::unique_ptr<IPlayer> player_1_ptr, std::unique_ptr<IPlayer> player_2_ptr, int n_sets, bool render);
        std::tuple<float, float> start();
        ~Match();
    };

} // namespace rl::common

#endif