#ifndef RL_PLAYERS_BANDITS_GRAVE_G_NODE_HPP_
#define RL_PLAYERS_BANDITS_GRAVE_G_NODE_HPP_

#include <memory>
#include <vector>
#include <utility>
#include <optional>
#include <common/state.hpp>
namespace rl::players
{
    class GNode
    {
        using IState = rl::common::IState;

    private:
        const std::unique_ptr<IState> state_ptr_;
        const int amaf_min_ref_count_;
        const float bias_;
        const bool save_illegal_actions_amaf_;
        const int current_player_;
        const int n_game_actions_;
        std::optional<bool> is_terminal_;
        std::optional<float> result_;
        std::vector<bool> actions_mask_{};
        bool is_leaf_node_{};
        std::vector<std::unique_ptr<GNode>> children_;

        int n_;
        std::vector<float> wsa_;
        std::vector<int> nsa_;
        std::vector<float> amaf_wsa_player_0_;
        std::vector<int> amaf_nsa_player_0_;
        std::vector<float> amaf_wsa_player_1_;
        std::vector<int> amaf_nsa_player_1_;
        void heuristic();
        std::pair<float, int> playout(std::vector<int> &player_0_actions, std::vector<int> &player_1_actions);
        void update_amf(const std::pair<const float, const int>& playout_res, const std::vector<int> &player_0_actions_ref, const std::vector<int> &player_1_actions_ref);

    public:
        GNode(std::unique_ptr<IState> state_ptr, int amaf_min_ref_count, float bias, bool save_illegal_actions_amaf);
        ~GNode();
        int choose_best_action(const GNode *amaf_ptr);
        std::pair<float, int> simulate_one(const GNode *amaf_ptr, std::vector<int> &player_0_actions, std::vector<int> &player_1_actions);
    };
} // namespace rl::players

#endif