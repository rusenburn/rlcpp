#ifndef RL_PLAYERS_BANDITS_GRAVE_G_HPP_
#define RL_PLAYERS_BANDITS_GRAVE_G_HPP_

#include <memory>
#include <vector>
#include <utility>
#include <optional>
#include <players/search_tree.hpp>
#include <common/state.hpp>
namespace rl::players
{
    class G : public ISearchTree
    {
        using IState = rl::common::IState;

    private:
        int n_game_actions_;
        int min_ref_count_;
        float bias_;
        bool save_illegal_amaf_actions_;

    public:
        G(int n_game_actions, int min_ref_count, float bias, bool save_illegal_actions_amaf);
        ~G() override;
        std::vector<float> search(const rl::common::IState *state_ptr, int minimum_no_simulations, std::chrono::duration<int, std::milli> minimum_duration) override;
    };
} // namespace rl::players

#endif