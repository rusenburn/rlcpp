#include <players/g_player.hpp>
#include <players/bandits/grave/grave.hpp>
#include <common/random.hpp>
namespace rl::players
{
    GPlayer::GPlayer(int minimum_simulations,
                     std::chrono::duration<int, std::milli> duration_in_millis,
                     int min_ref_count,
                     float b_squared,
                     bool save_illegal_amaf_actions)
        : minimum_simulations_{minimum_simulations},
          duration_in_millis_{duration_in_millis},
          min_ref_count_{min_ref_count},
          bias_{bias_},
          save_illegal_amaf_actions_{save_illegal_amaf_actions}
    {
    }

    GPlayer::~GPlayer() = default;

    int GPlayer::choose_action(const std::unique_ptr<rl::common::IState> &state_ptr)
    {
        int n_game_actions = state_ptr->get_n_actions();
        auto tree = G(n_game_actions, min_ref_count_, bias_, save_illegal_amaf_actions_);
        auto probs = tree.search(state_ptr.get(), minimum_simulations_, duration_in_millis_);

        float p = rl::common::get();

        float remaining_prob = p;

        int action = 0;

        int last_action = n_game_actions - 1;

        // keep decreasing remaining probs until it is below zero or only 1 action remains
        while ((action < last_action) && ((remaining_prob -= probs.at(action)) >= 0))
        {
            action++;
        }
        return action;
    }
} // namespace rl::players
