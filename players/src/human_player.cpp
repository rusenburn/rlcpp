#include <iostream>

#include <players/human_player.hpp>
namespace rl::players
{
    HumanPlayer::HumanPlayer() = default;
    
    HumanPlayer::~HumanPlayer() = default;
    

    int HumanPlayer::choose_action(const std::unique_ptr<rl::common::IState> &state_ptr)
    {
        state_ptr->render();
        int action;
        std::cin >> action;
        return action;
    }
} // namespace rl::players
