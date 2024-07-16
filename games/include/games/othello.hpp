#ifndef RL_GAMES_OTHELLO_HPP_
#define RL_GAMES_OTHELLO_HPP_

#include <common/state.hpp>
#include <memory>
#include <vector>
#include <array>

namespace rl::games
{
    class OthelloState : public rl::common::IState
    {
    private:
        static constexpr int ROWS = 8;
        static constexpr int COLS = 8;
        static constexpr int N_PLAYERS = 2;

        std::array<std::array<std::array<int8_t, COLS>, ROWS>, N_PLAYERS> observation_;
        int current_player_;
        int n_consecutive_skips_;
        mutable std::vector<bool> actions_legality_;
        mutable bool is_terminal_cached_;
        mutable bool is_terminal_;
        mutable bool is_result_cached_;
        mutable float cached_result_;

        std::vector<std::pair<int, int>> get_board_changes_on_action(int row, int col)const ;
        bool is_board_action(int row, int col)const;

    public:
        OthelloState(std::array<std::array<std::array<int8_t, COLS>, ROWS>, N_PLAYERS> observation, int player, int consecutive_skips);
        ~OthelloState() override;
        static std::unique_ptr<OthelloState> initialize_state();
        static std::unique_ptr<rl::common::IState> initialize();
        std::unique_ptr<rl::common::IState> reset() const override;
        std::unique_ptr<OthelloState> reset_state() const;
        std::unique_ptr<rl::common::IState> step(int action) const override;
        std::unique_ptr<OthelloState> step_state(int action) const ;
        void render() const override;
        bool is_terminal() const override;
        float get_reward() const override;
        std::vector<float> get_observation() const override;
        std::string to_short() const override;
        std::array<int, 3> get_observation_shape() const override;
        int get_n_actions() const override;
        int player_turn() const override;
        std::vector<bool> actions_mask() const override;
        std::unique_ptr<OthelloState> clone_state() const;
        std::unique_ptr<rl::common::IState> clone() const override;
        void get_symmetrical_obs_and_actions(std::vector<float> const &obs, std::vector<float> const &actions_distribution, std::vector<std::vector<float>> &out_syms, std::vector<std::vector<float>> &out_actions_distribution)const override;
    };

} // namespace rl::games

#endif