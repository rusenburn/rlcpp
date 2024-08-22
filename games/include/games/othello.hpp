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

    std::vector<std::pair<int, int>> get_board_changes_on_action(int row, int col) const;
    bool is_board_action(int row, int col) const;

public:
    OthelloState(std::array<std::array<std::array<int8_t, COLS>, ROWS>, N_PLAYERS> observation, int player, int consecutive_skips);
    ~OthelloState() override;
    static std::unique_ptr<OthelloState> initialize_state();
    static std::unique_ptr<rl::common::IState> initialize();
    std::unique_ptr<rl::common::IState> reset() const override;
    std::unique_ptr<OthelloState> reset_state() const;
    std::unique_ptr<rl::common::IState> step(int action) const override;
    std::unique_ptr<OthelloState> step_state(int action) const;
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
    void get_symmetrical_obs_and_actions(std::vector<float> const& obs, std::vector<float> const& actions_distribution, std::vector<std::vector<float>>& out_syms, std::vector<std::vector<float>>& out_actions_distribution) const override;
};

} // namespace rl::games

namespace rl::games::othello_syms
{
/*
If you want to convert probabilities of an actions distribution then probs_b[i] = probs_a[b[i]]
where prob_a is the probabilties of the original distribution
prob_b probabilties of the sym observation that we need to find
b is one of the actions array represented below
*/

constexpr std::array<int, 128> FIRST_OBS_SYM =
{ {56, 48, 40, 32, 24, 16, 8, 0,
  57, 49, 41, 33, 25, 17, 9, 1,
  58, 50, 42, 34, 26, 18, 10, 2,
  59, 51, 43, 35, 27, 19, 11, 3,
  60, 52, 44, 36, 28, 20, 12, 4,
  61, 53, 45, 37, 29, 21, 13, 5,
  62, 54, 46, 38, 30, 22, 14, 6,
  63, 55, 47, 39, 31, 23, 15, 7,
  120, 112, 104, 96, 88, 80, 72, 64,
  121, 113, 105, 97, 89, 81, 73, 65,
  122, 114, 106, 98, 90, 82, 74, 66,
  123, 115, 107, 99, 91, 83, 75, 67,
  124, 116, 108, 100, 92, 84, 76, 68,
  125, 117, 109, 101, 93, 85, 77, 69,
  126, 118, 110, 102, 94, 86, 78, 70,
  127, 119, 111, 103, 95, 87, 79, 71} };

constexpr std::array<int, 65> FIRST_ACTIONS_SYM =
{ {56, 48, 40, 32, 24, 16, 8, 0,
  57, 49, 41, 33, 25, 17, 9, 1,
  58, 50, 42, 34, 26, 18, 10, 2,
  59, 51, 43, 35, 27, 19, 11, 3,
  60, 52, 44, 36, 28, 20, 12, 4,
  61, 53, 45, 37, 29, 21, 13, 5,
  62, 54, 46, 38, 30, 22, 14, 6,
  63, 55, 47, 39, 31, 23, 15, 7,
  64} };

constexpr std::array<int, 128> SECOND_OBS_SYM =
{ {63, 62, 61, 60, 59, 58, 57, 56,
  55, 54, 53, 52, 51, 50, 49, 48,
  47, 46, 45, 44, 43, 42, 41, 40,
  39, 38, 37, 36, 35, 34, 33, 32,
  31, 30, 29, 28, 27, 26, 25, 24,
  23, 22, 21, 20, 19, 18, 17, 16,
  15, 14, 13, 12, 11, 10, 9, 8,
  7, 6, 5, 4, 3, 2, 1, 0,
  127, 126, 125, 124, 123, 122, 121, 120,
  119, 118, 117, 116, 115, 114, 113, 112,
  111, 110, 109, 108, 107, 106, 105, 104,
  103, 102, 101, 100, 99, 98, 97, 96,
  95, 94, 93, 92, 91, 90, 89, 88,
  87, 86, 85, 84, 83, 82, 81, 80,
  79, 78, 77, 76, 75, 74, 73, 72,
  71, 70, 69, 68, 67, 66, 65, 64} };
constexpr std::array<int, 65> SECOND_ACTIONS_SYM =
{ {63, 62, 61, 60, 59, 58, 57, 56,
  55, 54, 53, 52, 51, 50, 49, 48,
  47, 46, 45, 44, 43, 42, 41, 40,
  39, 38, 37, 36, 35, 34, 33, 32,
  31, 30, 29, 28, 27, 26, 25, 24,
  23, 22, 21, 20, 19, 18, 17, 16,
  15, 14, 13, 12, 11, 10, 9, 8,
  7, 6, 5, 4, 3, 2, 1, 0,
  64} };

constexpr std::array<int, 128> THIRD_OBS_SYM =
{ {7, 15, 23, 31, 39, 47, 55, 63,
  6, 14, 22, 30, 38, 46, 54, 62,
  5, 13, 21, 29, 37, 45, 53, 61,
  4, 12, 20, 28, 36, 44, 52, 60,
  3, 11, 19, 27, 35, 43, 51, 59,
  2, 10, 18, 26, 34, 42, 50, 58,
  1, 9, 17, 25, 33, 41, 49, 57,
  0, 8, 16, 24, 32, 40, 48, 56,
  71, 79, 87, 95, 103, 111, 119, 127,
  70, 78, 86, 94, 102, 110, 118, 126,
  69, 77, 85, 93, 101, 109, 117, 125,
  68, 76, 84, 92, 100, 108, 116, 124,
  67, 75, 83, 91, 99, 107, 115, 123,
  66, 74, 82, 90, 98, 106, 114, 122,
  65, 73, 81, 89, 97, 105, 113, 121,
  64, 72, 80, 88, 96, 104, 112, 120} };
constexpr std::array<int, 65> THIRD_ACTIONS_SYM =
{ {7, 15, 23, 31, 39, 47, 55, 63,
  6, 14, 22, 30, 38, 46, 54, 62,
  5, 13, 21, 29, 37, 45, 53, 61,
  4, 12, 20, 28, 36, 44, 52, 60,
  3, 11, 19, 27, 35, 43, 51, 59,
  2, 10, 18, 26, 34, 42, 50, 58,
  1, 9, 17, 25, 33, 41, 49, 57,
  0, 8, 16, 24, 32, 40, 48, 56,
  64} };

} // namespace rl::games::othello_syms

#endif