#ifndef RL_GAMES_TICTACTOE_HPP_
#define RL_GAMES_TICTACTOE_HPP_

#include <common/state.hpp>
#include <array>
namespace rl::games
{
class TicTacToeState : public rl::common::IState
{
public:
    static constexpr int ROWS{ 3 };
    static constexpr int COLS{ 3 };
    static constexpr int CHANNELS{ 2 };
    static constexpr int OBSERVATION_SIZE{ CHANNELS * ROWS * COLS };
    static constexpr int N_ACTIONS{ ROWS * COLS };
    static constexpr std::array<int, 2> FLAGS{ 1, -1 };
    TicTacToeState(std::array<std::array<int8_t, COLS>, ROWS> board, int8_t player);
    static std::unique_ptr<rl::common::IState> initialize();
    static std::unique_ptr<TicTacToeState> initialize_state();
    std::unique_ptr<rl::common::IState> step(int action) const override;
    std::unique_ptr<TicTacToeState> step_state(int action) const;

    std::unique_ptr<rl::common::IState> reset() const override;
    std::unique_ptr<TicTacToeState> reset_state() const;

    void render() const override;

    bool is_terminal() const override;

    float get_reward() const override;

    std::vector<float> get_observation() const override;

    std::string to_short() const override;

    std::array<int, 3> get_observation_shape() const override;

    int get_n_actions() const override;

    int player_turn() const override;
    std::vector<bool> actions_mask() const override;
    ~TicTacToeState() override;
    std::unique_ptr<rl::common::IState> clone() const override;
    std::unique_ptr<TicTacToeState> clone_state() const;
    void get_symmetrical_obs_and_actions(std::vector<float> const& obs, std::vector<float> const& actions_distribution, std::vector<std::vector<float>>& out_syms, std::vector<std::vector<float>>& out_actions_distribution) const override;

private:
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
};
} // namespace rl::games

namespace rl::games::tictactoe_syms
{

constexpr std::array<int, rl::games::TicTacToeState::OBSERVATION_SIZE> FIRST_SYM =
{ {6, 3, 0, 7, 4, 1, 8, 5, 2, 15, 12, 9, 16, 13, 10, 17, 14, 11} };
constexpr std::array<int, rl::games::TicTacToeState::N_ACTIONS> FIRST_ACTIONS =
{ {6, 3, 0, 7, 4, 1, 8, 5, 2} };
constexpr std::array<int, rl::games::TicTacToeState::OBSERVATION_SIZE> SECOND_SYM =
{ {8, 7, 6, 5, 4, 3, 2, 1, 0, 17, 16, 15, 14, 13, 12, 11, 10, 9} };
constexpr std::array<int, rl::games::TicTacToeState::N_ACTIONS> SECOND_ACTIONS =
{ {8, 7, 6, 5, 4, 3, 2, 1, 0} };
constexpr std::array<int, rl::games::TicTacToeState::OBSERVATION_SIZE> THIRD_SYM =
{ {2, 5, 8, 1, 4, 7, 0, 3, 6, 11, 14, 17, 10, 13, 16, 9, 12, 15} };
constexpr std::array<int, rl::games::TicTacToeState::N_ACTIONS> THIRD_ACTIONS =
{ {2, 5, 8, 1, 4, 7, 0, 3, 6} };
} // namespace rl::games::tictactoe_sym

#endif