#include <common/concurrent_match.hpp>

#include <cassert>

namespace rl::common
{

ConcurrentMatch::ConcurrentMatch(const std::unique_ptr<IState>& initial_state_ptr, IConcurrentPlayer* player_1_ptr, IConcurrentPlayer* player_2_ptr, int n_sets, int batch_size)
    : initial_state_ptr_{ initial_state_ptr->clone() },
    n_sets_{ n_sets },
    batch_size_{ batch_size }
{
    players_ptrs_.push_back(player_1_ptr);
    players_ptrs_.push_back(player_2_ptr);
}

float common::ConcurrentMatch::start()
{
    return play_sets();
}

ConcurrentMatch::~ConcurrentMatch() = default;

float rl::common::ConcurrentMatch::play_sets()
{
    std::vector<bool> are_players_swapped{};
    float player_1_total_rewards = 0;
    are_players_swapped.reserve(n_sets_);
    for (int i = 0;i < n_sets_; i++)
    {
        are_players_swapped.push_back(i % 2 == 0);
    }

    int n_games_ended = 0;

    std::vector<std::unique_ptr<rl::common::IState>> current_states{};
    std::vector<int> current_are_players_swapped{};
    std::vector<int> current_players{};
    std::vector<const IState*> player_1_states{};
    std::vector<const IState*> player_2_states{};
    std::vector<int> player_1_indices{};
    std::vector<int> player_2_indices{};
    while (n_games_ended < n_sets_)
    {
        while ((current_states.size() < batch_size_) && (current_states.size() + n_games_ended < n_sets_))
        {
            int current_idx = n_games_ended + current_states.size();
            current_states.push_back(initial_state_ptr_->clone());
            current_are_players_swapped.push_back(are_players_swapped.at(current_idx));

        }

        current_players.clear();
        player_1_states.clear();
        player_2_states.clear();
        player_1_indices.clear();
        player_2_indices.clear();
        int n_states = current_states.size();

        for (int i = 0;i < n_states;i++)
        {
            const auto& state_ptr = current_states.at(i);
            int current_ingame_player = state_ptr->player_turn();
            bool is_swapped = current_are_players_swapped.at(i);
            int current_player = is_swapped ? 1 - current_ingame_player : current_ingame_player;

            if (current_player == 0)
            {
                player_1_states.push_back(state_ptr.get());
                player_1_indices.push_back(i);
            }
            else
            {
                player_2_states.push_back(state_ptr.get());
                player_2_indices.push_back(i);
            }
        }

        auto player_1_actions = players_ptrs_.at(0)->choose_actions(player_1_states);
        assert(player_1_actions.size() == player_1_indices.size());
        assert(player_1_actions.size() == player_1_states.size());
        auto player_2_actions = players_ptrs_.at(1)->choose_actions(player_2_states);
        assert(player_2_actions.size() == player_2_indices.size());
        assert(player_2_actions.size() == player_2_states.size());
        std::vector<int> current_actions(current_states.size(), -1);
        for (int i = 0;i < player_1_actions.size();i++)
        {
            int current_action = player_1_actions.at(i);
            int current_idx = player_1_indices.at(i);
            current_actions.at(current_idx) = current_action;
        }

        for (int i = 0;i < player_2_actions.size();i++)
        {
            int current_action = player_2_actions.at(i);
            int current_idx = player_2_indices.at(i);
            current_actions.at(current_idx) = current_action;
        }

        for (int i = 0;i < current_states.size();i++)
        {
            int current_action = current_actions.at(i);
            current_states.at(i) = current_states.at(i)->step(current_action);

        }

        for (int i = 0;i < current_states.size();i++)
        {
            auto& state_ptr = current_states.at(i);
            if (state_ptr->is_terminal())
            {
                float reward = state_ptr->get_reward();
                int ingame_player = state_ptr->player_turn();
                bool is_swapped = current_are_players_swapped.at(i);

                reward = ingame_player == 0 ? reward : -reward;
                reward = is_swapped ? -reward : reward;
                player_1_total_rewards += reward;
                n_games_ended++;

                // current_states.erase(std::next(current_states.begin() + i));
                // current_are_players_swapped.erase(std::next(current_are_players_swapped.begin() + i));
                // i--;
            }
        }

        auto states_it = current_states.begin();
        auto is_swapped_it = current_are_players_swapped.begin();
        while (states_it != current_states.end())
        {
            auto& a = *states_it;
            if (a->is_terminal())
            {
                current_states.erase(states_it);
                current_are_players_swapped.erase(is_swapped_it);
            }
            else
            {
                states_it++;
                is_swapped_it++;
            }
        }
    }

    return player_1_total_rewards / static_cast<float>(n_games_ended);

}

} // namespace rl::common

