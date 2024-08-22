#include <players/uct_player.hpp>
#include <common/random.hpp>
namespace rl::players
{
UctPlayer::UctPlayer(int minimum_simulations, std::chrono::duration<int, std::milli> duration_in_millis,
    float temperature, float cuct)
    : minimum_simulations_{ minimum_simulations },
    duration_in_millis_{ duration_in_millis },
    temperature_{ temperature },
    cuct_{ cuct }
{
}

UctPlayer::~UctPlayer() = default;
int UctPlayer::choose_action(const std::unique_ptr<rl::common::IState>& state_ptr)
{
    int n_game_actions = state_ptr->get_n_actions();
    auto uct = UctSearchTree(n_game_actions, cuct_, temperature_);
    auto probs = uct.search(state_ptr.get(), minimum_simulations_, duration_in_millis_);

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
