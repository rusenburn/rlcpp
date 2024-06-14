#ifndef RL_GAMES_TICTACTOE_HPP_
#define RL_GAMES_TICTACTOE_HPP_

#include <common/state.hpp>
#include <array>
namespace rl::games
{
    class TicTacToeState : public rl::common::IState
    {
    private:
        static constexpr int ROWS{3};
        static constexpr int COLS{3};
        static constexpr int CHANNELS{2};
        static constexpr std::array<int, 2> FLAGS{1, -1};
        std::array<std::array<int8_t, ROWS>, COLS> board_;
        int8_t player_;

        // used for caching , should not change the state
        mutable std::vector<bool> legal_actions_;
        mutable bool is_game_over_cached_;
        mutable bool is_game_over_;
        mutable bool is_game_result_cached_;
        mutable float game_result_cache_;

        bool is_legal_action(int action) const;
        bool is_winning(int player) const;
        bool is_full() const;
        bool is_horizontal_win(int player) const;
        bool is_vertical_win(int player) const;
        bool is_forward_diagonal_win(int player) const;
        bool is_backward_diagonal_win(int player) const;

    public:
        TicTacToeState(std::array<std::array<int8_t, COLS>, ROWS> board, int8_t player);
        static std::unique_ptr<rl::common::IState> initialize();
        static std::unique_ptr<TicTacToeState> initialize_state();
        std::unique_ptr<rl::common::IState> step(int action) const override;
        std::unique_ptr<TicTacToeState> step_state(int action)const;

        std::unique_ptr<rl::common::IState> reset()const override;
        std::unique_ptr<TicTacToeState> reset_state()const;

        void render() const override;

        bool is_terminal() const override;

        float get_reward() const override;

        std::vector<float> get_observation() const override;

        std::string to_short() const override;

        std::array<int,3> get_observation_shape() const override;

        int get_n_actions() const override;

        int player_turn() const override;
        std::vector<bool> actions_mask()const override;
        ~TicTacToeState() override;
        std::unique_ptr<rl::common::IState> clone()const override;
        std::unique_ptr<TicTacToeState> clone_state()const;
    };
} // namespace rl::games

#endif