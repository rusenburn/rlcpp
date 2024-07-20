#include <iostream>
#include <array>

#include <common/match.hpp>

namespace rl::common
{
    Match::Match(std::unique_ptr<IState> initial_state_ptr, IPlayer* player_1_ptr, IPlayer* player_2_ptr, int n_sets, bool render)
        : initial_state_ptr_{std::move(initial_state_ptr)},
          n_sets_{n_sets},
          render_{render}
    {
        players_ptrs_.push_back(player_1_ptr);
        players_ptrs_.push_back(player_2_ptr);
    }
    std::tuple<float, float> Match::start()
    {
        int starting_player = 0;
        std::tuple<float, float> scores{0.0f, 0.0f};
        for (int set_index{0}; set_index < n_sets_; set_index++)
        {
            auto set_scores = play_set(starting_player);
            std::get<0>(scores) += std::get<0>(set_scores);
            std::get<1>(scores) += std::get<1>(set_scores);
            starting_player = 1 - starting_player;
        }
        return scores;
    }

    std::tuple<float, float> Match::play_set(int starting_player)
    {
        std::array<float, 3> result{};
        auto state_ptr = initial_state_ptr_->reset();

        state_changed_event.notify(state_ptr.get());

        bool is_terminal = state_ptr->is_terminal();
        int game_player = state_ptr->player_turn();
        bool inverted = starting_player == game_player ? false : true;
        int current_player;
        while (!state_ptr->is_terminal())
        {
            if (render_)
            {
                state_ptr->render();
            }
            game_player = state_ptr->player_turn();
            current_player = inverted ? 1 - game_player : game_player;

            IPlayer* player_ptr = players_ptrs_.at(current_player);
            int action = player_ptr->choose_action(state_ptr);
            std::vector<bool> actions_mask = state_ptr->actions_mask();
            if (actions_mask.at(action) == false)
            {
                std::cout << "Player" << current_player + 1 << "chose the wrong action " << action << " \n";
                continue;
            }
            state_ptr = state_ptr->step(action);
            state_changed_event.notify(state_ptr.get());
            if (render_)
            {
                std::cout << "Player" << current_player + 1 << " played action " << action << " " << std::endl;
            }
        }

        if (render_)
        {
            state_ptr->render();
        }
        float game_result = state_ptr->get_reward();
        game_player = state_ptr->player_turn();
        current_player = inverted ? 1 - game_player : game_player;
        if (current_player == 0)
        {
            return std::make_tuple(game_result, -game_result);
        }
        else
        {
            return std::make_tuple(-game_result, game_result);
        }
    }
    Match::~Match() = default;
} // namespace rl::common
