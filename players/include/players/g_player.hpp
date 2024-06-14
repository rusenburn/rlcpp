#ifndef RL_PLAYERS_G_PLAYER_HPP_
#define RL_PLAYERS_G_PLAYER_HPP_

#include <common/player.hpp>
#include <players/bandits/grave/g.hpp>
namespace rl::players
{
    class GPlayer : public common::IPlayer
    {
    private:
        int minimum_simulations_;
        std::chrono::duration<int, std::milli> duration_in_millis_;
        int min_ref_count_;
        float bias_;
        bool save_illegal_amaf_actions_;

    public:
        GPlayer(int minimum_simulations_,
                std::chrono::duration<int, std::milli> duration_in_millis_,
                int min_ref_count = 15,
                float bias = 0.04f,
                bool save_illegal_amaf_actions = true);
        ~GPlayer() override;
        int choose_action(const std::unique_ptr<rl::common::IState> &state_ptr) override;
    };

} // namespace rl::players

#endif