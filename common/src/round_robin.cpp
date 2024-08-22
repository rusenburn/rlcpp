#include <common/round_robin.hpp>
#include <algorithm>
#include <common/random.hpp>
#include <common/match.hpp>
#include <tuple>
#include <functional>
#include <sstream>
#include <iostream>
namespace rl::common
{

RoundRobin::RoundRobin(std::unique_ptr<IState> initial_state_ptr, const std::vector<PlayerInfo>& players_info, int n_sets, bool render)
    : initial_state_ptr_{ std::move(initial_state_ptr) }, players_info_(players_info.begin(), players_info.end()), n_sets_{ n_sets }, render_{ render },
    scores_(players_info.size(), std::array<int, 3>({ 0, 0, 0 })),
    scores_table_(players_info.size(), std::vector<std::array<int, 3>>(players_info.size(), { 0, 0, 0 })),
    indices_()
{
    indices_.reserve(players_info.size());
    for (int i = 0; i < players_info_.size(); i++)
    {
        indices_.emplace_back(i);
    }
    std::shuffle(indices_.begin(), indices_.end(), rl::common::mt);
}

RoundRobin::~RoundRobin()
{
    players_info_.clear();
    players_info_.shrink_to_fit();
    scores_.clear();
    scores_.shrink_to_fit();
}

std::pair<std::vector<std::array<int, 3>>, std::vector<int>> RoundRobin::start()
{
    const int players_size = static_cast<int>(players_info_.size());
    bool is_even = players_size % 2 == 0 ? true : false;
    if (!is_even)
    {
        indices_.push_back(static_cast<int>(players_size));
        players_info_.push_back({ nullptr, "" });
    }
    const int even_players_size = static_cast<int>(players_info_.size());

    const int n_matches_per_round = even_players_size / 2;
    std::vector<std::pair<int, int>> matches;
    for (int i = 0; i < n_sets_; i++)
    {
        for (int j = 0; j < even_players_size - 1; j++)
        {
            for (int k = 0; k < n_matches_per_round; k++)
            {
                int first_player_seed;
                int second_player_seed;
                int first_player_index;
                int second_player_index;
                if (k == 0)
                {
                    first_player_seed = j;
                    first_player_index = indices_.at(first_player_seed);
                    // last_player
                    second_player_seed = (even_players_size - 1);
                    second_player_index = indices_.at(second_player_seed);
                    if (i + j % 2 == 0)
                    {
                        matches.push_back(std::make_pair(first_player_index, second_player_index));
                    }
                    else
                    {
                        matches.push_back(std::make_pair(second_player_index, first_player_index));
                    }
                }
                else
                {
                    first_player_seed = (j + k) % (even_players_size - 1);
                    first_player_index = indices_.at(first_player_seed);

                    second_player_seed = (j - k + even_players_size - 1) % (even_players_size - 1);
                    second_player_index = indices_.at(second_player_seed);
                    if (i + j % 2 == 0)
                    {
                        matches.push_back(std::make_pair(first_player_index, second_player_index));
                    }
                    else
                    {
                        matches.push_back(std::make_pair(second_player_index, first_player_index));
                    }
                }
            }
        }
    }

    for (auto [first_player_id, second_player_id] : matches)
    {
        execute_match_(first_player_id, second_player_id);
    }

    if (!is_even)
    {
        // if the original players info is not even , then we added null player , remove it
        players_info_.pop_back();
        indices_.pop_back();
    }

    std::vector<int> rankings{};
    get_rankings_out_(rankings);

    return std::make_pair(scores_, rankings);
}

void RoundRobin::render_scores()
{
    std::stringstream ss{};
    ss << "\n";
    ss << "****************\n";
    ss << "****************\n";
    for (int i = 0;i < scores_.size();i++)
    {
        auto& scores = scores_.at(i);
        int wins = std::get<0>(scores);
        int draws = std::get<1>(scores);
        int losses = std::get<2>(scores);
        int played = wins + draws + losses;
        int points = wins - losses;
        ss << players_info_.at(i).player_name << " " << points << " " << played;
        ss << "\n";
    }
    ss << "****************\n";
    ss << "****************\n";

    std::cout << ss.str() << std::endl;

}
void RoundRobin::execute_match_(int first_player_index, int second_player_index)
{
    PlayerInfo p1info = players_info_.at(first_player_index);
    PlayerInfo p2info = players_info_.at(second_player_index);
    if (p1info.player_ptr == nullptr || p2info.player_ptr == nullptr)
    {
        return;
    }
    rl::common::Match m{ initial_state_ptr_->clone(), p1info.player_ptr, p2info.player_ptr, 1, render_ };
    std::function<void(const IState*)> fn = std::bind(&RoundRobin::on_state_changed_, this, std::placeholders::_1, first_player_index, second_player_index);
    state_observer_ptr_ = m.state_changed_event.subscribe(fn);
    auto [p1, p2] = m.start();
    std::array<int, 3> p1_score{ 0, 0, 0 };
    if (p1 > p2)
    {
        p1_score = { 1, 0, 0 };
    }
    else if (p2 > p1)
    {
        p1_score = { 0, 0, 1 };
    }
    else
    {
        p1_score = { 0, 1, 0 };
    }
    set_score_(first_player_index, second_player_index, p1_score);
    render_scores();
}

void RoundRobin::set_score_(int first_player_index, int second_player_index, const std::array<int, 3>& result)
{
    std::array<int, 3>& first_player_score_ref = scores_.at(first_player_index);
    std::array<int, 3>& second_player_score_ref = scores_.at(second_player_index);
    std::array<int, 3>& first_player_vs_second_player_score_ref = scores_table_.at(first_player_index).at(second_player_index);
    std::array<int, 3>& second_player_vs_first_player_score_ref = scores_table_.at(second_player_index).at(first_player_index);
    for (int i = 0; i < result.size(); i++)
    {
        first_player_score_ref.at(i) += result.at(i);
        second_player_score_ref.at(result.size() - 1 - i) += result.at(i);

        first_player_vs_second_player_score_ref.at(i) += result.at(i);
        second_player_vs_first_player_score_ref.at(i) += result.at(result.size() - 1 - i);
    }
}

void RoundRobin::get_rankings_out_(std::vector<int>& rankings_out)
{
    const int players_count = players_info_.size();
    rankings_out.clear();
    rankings_out.resize(players_count, 0);
    std::vector<std::tuple<int, int, int>> points_and_tie_breakers(players_count, std::make_tuple(0, 0, 0));
    for (int i = 0; i < players_count; i++)
    {
        std::get<2>(points_and_tie_breakers.at(i)) = i;
    }

    // calculate points
    for (int i = 0; i < players_count; i++)
    {
        // score = wins *2 + draws
        int score = /*wins*/ scores_.at(i).at(0) * 2 + /*draws*/ scores_.at(i).at(1);
        std::get<0>(points_and_tie_breakers.at(i)) += score;
    }
    // calculate tie-breakers
    for (int i = 0; i < players_count; i++)
    {
        // sum of ( [wins_against_opponent * opponent_points*2 + draws_against_opponent * opponent_points ] for each opponent in opponents)
        int player_tie_breaker_points = 0;
        const auto& players_vs_scores_ref = scores_table_.at(i);
        for (int j = 0; j < players_count; j++)
        {
            int wins_vs_opponent = std::get<0>(players_vs_scores_ref.at(j));
            int draws_vs_opponent = std::get<1>(players_vs_scores_ref.at(j));
            int opponent_score = std::get<0>(points_and_tie_breakers.at(j));
            player_tie_breaker_points += opponent_score * wins_vs_opponent * 2 + opponent_score * draws_vs_opponent;
        }
        // player's tie-breaker points
        std::get<1>(points_and_tie_breakers.at(i)) = player_tie_breaker_points;
    }

    // sort non-descending-order(ascending order)'
    auto pred = [](const std::tuple<int, int, int>& first, const std::tuple<int, int, int>& second) -> bool
        {
            if (std::get<0>(first) < std::get<0>(second))
            {
                return true;
            }
            else if (std::get<0>(first) == std::get<0>(second))
            {
                return std::get<1>(first) < std::get<1>(second);
            }
            return false;
        };
    std::sort(points_and_tie_breakers.begin(), points_and_tie_breakers.end(), pred);
    // reverse to get the descending order
    std::reverse(points_and_tie_breakers.begin(), points_and_tie_breakers.end());
    for (int rank = 0; rank < players_count; rank++)
    {
        int player_index = std::get<2>(points_and_tie_breakers.at(rank));
        rankings_out.at(player_index) = rank;
    }
}

bool RoundRobin::compare_predicate_(const std::tuple<int, int, int>& first, const std::tuple<int, int, int>& second) const
{
    if (std::get<0>(first) < std::get<0>(second))
    {
        return true;
    }
    else if (std::get<0>(first) == std::get<0>(second))
    {
        return std::get<1>(first) < std::get<1>(second);
    }
    return false;
}

void RoundRobin::on_state_changed_(const IState* state_ptr, int player_1_index, int player_2_index)
{
    matchinfo_changed_event.notify(MatchInfo{ state_ptr,player_1_index,player_2_index });
}
} // namespace rl::common
