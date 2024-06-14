#ifndef RL_PLAYERS_BANDITS_GRAVE_GRAVE_NODE_HPP_
#define RL_PLAYERS_BANDITS_GRAVE_GRAVE_NODE_HPP_

#include <common/state.hpp>
#include <memory>
#include <optional>
#include <utility>
namespace rl::players
{
    class GraveNode
    {
        using IState = rl::common::IState;

    private:
        std::unique_ptr<IState> state_ptr_;
        int n_game_actions_;
        int current_player_;
        std::optional<bool> terminal_;
        bool is_leaf_node_;
        std::optional<bool> game_result_;
        std::vector<std::unique_ptr<GraveNode>> children_;
        int n_;
        // total score for action "a"
        std::vector<float> wsa_;

        // total visits for action "a"
        std::vector<int> nsa_;

        // player 0 amaf total score for action "a"
        std::vector<float> amaf_wsa_player_0;

        // player 0 amaf total visits for action "a"
        std::vector<int> amaf_nsa_player_0;

        // player 1 amaf total score for action "a"
        std::vector<float> amaf_wsa_player_1;

        // player 1 amaf total score for action "a"
        std::vector<int> amaf_nsa_player_1;

        std::vector<bool> actions_legality_;
        void heuristic();
        void update_amaf(float our_score, const std::vector<int> &our_actions_ref, const std::vector<int> &opponents_actions_ref, int depth, bool save_illegal_amaf_actions);

    protected:
        // named playout in the paper
        virtual std::pair<float,int> playout(std::vector<int> &out_our_actions, std::vector<int> &out_opponent_actions);

    public:
        GraveNode(std::unique_ptr<IState> state_ptr, int n_game_actions);
        ~GraveNode();
        std::pair<float, int> simulateOne(GraveNode *amaf_node_ptr,
                                          bool save_illegal_amaf_actions,
                                          int depth,
                                          std::vector<int> &out_our_actions,
                                          std::vector<int> &out_opponent_actions,
                                          const int &amaf_min_ref_count,
                                          const float &b_square_ref);
        int selectMove(GraveNode *amaf_node_ptr, int depth, const float &b_square_ref);
    };
}

#endif