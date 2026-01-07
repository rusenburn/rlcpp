#ifndef RL_GAMES_GOBBLET_GOBLERS_HPP_
#define RL_GAMES_GOBBLET_GOBLERS_HPP_

#include <common/state.hpp>
#include <tuple>
#include <array>
#include <optional>

namespace rl::games
{
class GobbletGoblersState : public rl::common::IState
{
public:

    static constexpr int ROWS{ 3 };
    static constexpr int COLS{ 3 };
    static constexpr int CHANNELS{ 12 };
    static constexpr int OBSERVATION_SIZE{ CHANNELS * ROWS * COLS };
    static constexpr int TURN_CHANNEL = 11;
    static constexpr int SELECTION_PHASE_CHANNEL = 10;
    static constexpr int SELECTED_PIECE_ONBOARD_CHANNEL = 6;
    static constexpr int SELECTED_PIECE_SMALL_CHANNEL = 7;
    static constexpr int SELECTED_PIECE_MEDIUM_CHANNEL = 8;
    static constexpr int SELECTED_PIECE_LARGE_CHANNEL = 9;
    static constexpr int MAX_TURNS = 50;

    static constexpr int N_ACTIONS{ ROWS * COLS + 3 };
    GobbletGoblersState(std::array<std::array<std::array<int8_t, COLS>, ROWS>, CHANNELS> board, int8_t player, int turn);
    static std::unique_ptr<rl::common::IState> initialize();
    static std::unique_ptr<GobbletGoblersState> initialize_state();
    std::unique_ptr<rl::common::IState> step(int action) const override;
    std::unique_ptr<GobbletGoblersState> step_state(int action) const;
    std::unique_ptr<rl::common::IState> reset() const override;
    std::unique_ptr<GobbletGoblersState> reset_state() const;
    void render() const override;
    bool is_terminal() const override;
    float get_reward() const override;
    std::vector<float> get_observation() const override;

    std::string to_short() const override;

    std::array<int, 3> get_observation_shape() const override;

    int get_n_actions() const override;

    int player_turn() const override;
    std::vector<bool> actions_mask() const override;
    ~GobbletGoblersState() override;
    std::unique_ptr<rl::common::IState> clone() const override;
    std::unique_ptr<GobbletGoblersState> clone_state() const;
    void get_symmetrical_obs_and_actions(std::vector<float> const& obs, std::vector<float> const& actions_distribution, std::vector<std::vector<float>>& out_syms, std::vector<std::vector<float>>& out_actions_distribution) const override;
    std::array<std::array<int, COLS>, ROWS> get_board()const;


private:
    std::array<std::array<std::array<int8_t, COLS>, ROWS>, CHANNELS> board_;
    int8_t player_;
    int turn_;

    // cached
    mutable std::vector<bool> legal_actions_;
    mutable std::optional<bool> cached_is_terminal_;
    mutable std::optional<float> cached_result_;
    mutable std::vector<float> cached_observation_;

    static void get_top_board(std::array<std::array<std::array<int8_t, COLS>, ROWS>, CHANNELS> const& board, std::array<std::array<int, COLS>, ROWS>& top_board);

    static bool is_winning(std::array<std::array<int, COLS>, ROWS>const& top_board, int player);
    bool is_draw()const;

    static bool is_horizontal_win(std::array<std::array<int, COLS>, ROWS>const& top_board, int player);
    static bool is_vertical_win(std::array<std::array<int, COLS>, ROWS>const& top_board, int player);
    static bool is_forward_diagonal_win(std::array<std::array<int, COLS>, ROWS>const& top_board, int player);
    static bool is_backward_diagonal_win(std::array<std::array<int, COLS>, ROWS>const& top_board, int player);
    static void fill_channel(std::array<std::array<std::array<int8_t, COLS>, ROWS>, CHANNELS>& array, int channel, int fill_value);

    const std::string RESET = "\033[0m";
    const std::string RED = "\033[31m";


};
}


namespace rl::games::gobblet_syms
{

constexpr int SIZE = rl::games::GobbletGoblersState::CHANNELS * rl::games::GobbletGoblersState::ROWS * rl::games::GobbletGoblersState::COLS;
constexpr std::array<int, SIZE> FIRST_OBS_SYM =
{ {
    6,  3,  0,  7,  4,  1,  8,  5,  2,
     15, 12,  9, 16, 13, 10, 17, 14, 11,
      24, 21, 18, 25, 22, 19, 26, 23, 20,
       33, 30, 27, 34, 31, 28, 35, 32, 29,
        42, 39, 36, 43, 40, 37, 44, 41, 38,
        51, 48, 45, 52, 49, 46, 53, 50, 47,
         60, 57, 54, 61, 58, 55, 62, 59, 56,
         63, 64, 65, 66, 67, 68, 69, 70, 71,
         72, 73, 74, 75, 76, 77, 78, 79, 80,
         81, 82, 83, 84, 85, 86, 87, 88, 89,
         90, 91, 92, 93, 94, 95, 96, 97, 98,
         99,100,101,102,103,104,105,106,107
} };

constexpr std::array<int, rl::games::GobbletGoblersState::N_ACTIONS> FIRST_ACTIONS = { {
    6,  3,  0,  7,  4,  1,  8,  5,  2,
     9, 10, 11,
} };

constexpr std::array<int, SIZE> SECOND_OBS_SYM =
{ {
    8,  7,  6,  5,  4,  3,  2,  1,  0,
     17, 16, 15, 14, 13, 12, 11, 10,  9,
      26, 25, 24, 23, 22, 21, 20, 19, 18,
       35, 34, 33, 32, 31, 30, 29, 28, 27,
        44, 43, 42, 41, 40, 39, 38, 37, 36,
        53, 52, 51, 50, 49, 48, 47, 46, 45,
         62, 61, 60, 59, 58, 57, 56, 55, 54,
          63, 64, 65, 66, 67, 68, 69, 70, 71,
           72, 73, 74, 75, 76, 77, 78, 79, 80,
            81, 82, 83, 84, 85, 86, 87, 88, 89,
             90, 91, 92, 93, 94, 95, 96, 97, 98,
              99,100,101,102,103,104,105,106,107,
} };

constexpr std::array<int, rl::games::GobbletGoblersState::N_ACTIONS> SECOND_ACTIONS = { {
    8,  7,  6,  5,  4,  3,  2,  1,  0,
    9, 10, 11
} };
constexpr std::array<int, SIZE> THIRD_OBS_SYM = { {
    2,  5,  8,  1,  4,  7,  0,  3,  6, 11, 14, 17, 10, 13, 16,  9, 12, 15, 20, 23, 26, 19, 22, 25, 18,
21, 24, 29, 32, 35, 28, 31, 34, 27, 30, 33, 38, 41, 44, 37, 40, 43, 36, 39, 42, 47, 50, 53, 46, 49, 52, 45, 48, 51, 56, 59, 62, 55, 58, 61, 54, 57, 60, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99,100,101,102,103,104,105,106,107,
} };

constexpr std::array<int, rl::games::GobbletGoblersState::N_ACTIONS> THIRD_ACTIONS = { {
    2,  5,  8,  1,  4,  7,  0,  3,  6,
    9, 10, 11,
} };

} // namespace rl::games::gobblet_syms

#endif


