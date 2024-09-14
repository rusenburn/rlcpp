#ifndef RL_COMMON_CONCURRENT_MATCH_HPP_
#define RL_COMMON_CONCURRENT_MATCH_HPP_

#include <memory>
#include <tuple>
#include <vector>
#include "state.hpp"
#include "concurrent_player.hpp"


namespace rl::common
{
    class ConcurrentMatch
    {
        private:
        std::unique_ptr<IState> initial_state_ptr_;
        std::vector<IConcurrentPlayer*> players_ptrs_{};

        int n_sets_;
        int batch_size_;
        float play_sets();
        public:
        ConcurrentMatch(const std::unique_ptr<IState>& initial_state_ptr, IConcurrentPlayer* player_1_ptr, IConcurrentPlayer* player_2_ptr, int n_sets,int batch_size);
        float start();
        ~ConcurrentMatch();
    };
} // namespace rl::common

#endif