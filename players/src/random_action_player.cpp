#include <players/random_action_player.hpp>

#include <algorithm>
#include <common/random.hpp>

namespace rl::players
{
    RandomActionPlayer::RandomActionPlayer() = default;

    RandomActionPlayer::~RandomActionPlayer() = default;

    int RandomActionPlayer::choose_action(const std::unique_ptr<rl::common::IState> &state_ptr)
    {
        const std::vector<bool> mask = state_ptr->actions_mask();
        std::vector<int> legal_actions{};

        for (int i{0}; i < mask.size(); i++)
        {
            if (mask.at(i))
                legal_actions.push_back(i);
        }

        int action_idx = rl::common::get(static_cast<int>(legal_actions.size()));
        int action = legal_actions[action_idx];
        return action;
    }

} // namespace rl::players
