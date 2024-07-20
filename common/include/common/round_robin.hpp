#ifndef RL_COMMON_ROUND_ROBIN_HPP_
#define RL_COMMON_ROUND_ROBIN_HPP_

#include <memory>
#include <array>
#include <vector>
#include <string>
#include "state.hpp"
#include "player.hpp"
#include "observer.hpp"

namespace rl::common
{

    struct PlayerInfo
    {
        IPlayer *player_ptr;
        std::string player_name;
    };

    struct MatchInfo
    {
        const IState *state_ptr;
        const int player_1_index;
        const int player_2_index;
    };
    struct PlayerScore
    {
        const int player_index;
        const int wins;
        const int draws;
        const int losses;
    };

    class RoundRobin
    {

    public:
        RoundRobin(std::unique_ptr<IState> initial_state_ptr, const std::vector<PlayerInfo> &players_info, int n_sets, bool render);
        ~RoundRobin();
        std::pair<std::vector<std::array<int, 3>>, std::vector<int>> start();

        // events
        Subject<MatchInfo> matchinfo_changed_event;
        Subject<std::vector<PlayerScore> &> players_scores_changed_event;
        void render_scores();

    private:
        std::unique_ptr<IState> initial_state_ptr_;
        std::vector<PlayerInfo> players_info_;
        std::shared_ptr<Observer<const IState*>> state_observer_ptr_{nullptr};

        int n_sets_;
        bool render_;
        std::vector<std::array<int, 3>> scores_;
        std::vector<std::vector<std::array<int, 3>>> scores_table_;
        std::vector<int> indices_;

        void execute_match_(int first_player_index, int second_player_index);
        void set_score_(int first_player_index, int second_player_index, const std::array<int, 3> &result);
        void get_rankings_out_(std::vector<int> &rankings_out);
        bool compare_predicate_(const std::tuple<int, int, int> &first, const std::tuple<int, int, int> &second) const;

        // event handler
        void on_state_changed_(const IState* state_ptr, int player_1_index, int player_2_index);
    };

} // namespace rl::common

#endif