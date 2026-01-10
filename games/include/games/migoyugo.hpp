#ifndef RL_GAMES_MIGOYUGO_HPP_
#define RL_GAMES_MIGOYUGO_HPP_


#include <common/state.hpp>
#include <memory>
#include <vector>
#include <array>
#include <optional>

namespace rl::games
{
class MigoyugoState : public rl::common::IState
{
private:
  static constexpr int ROWS = 8;
  static constexpr int COLS = 8;
  static constexpr int CHANNELS = 4;
  static constexpr int N_PLAYERS = 2;
  static constexpr int N_ACTIONS = ROWS * COLS;


  static constexpr int OUR_MIGO_CHANNEL = 0;
  static constexpr int OUR_YUGO_CHANNEL = 1;
  static constexpr int OPPONENT_MIGO_CHANNEL = 2;
  static constexpr int OPPONENT_YUGO_CHANNEL = 3;

  std::array<std::array<int8_t, COLS>, ROWS> board_;
  int current_player_;
  mutable std::vector<bool> cached_actions_masks_{};
  mutable std::optional<bool> cached_is_terminal_;
  mutable std::optional<float> cached_result_;
  mutable std::vector<float> cached_observation_{};

  std::vector<std::pair<int, int>> get_board_changes_on_action(int row, int col) const;
  std::vector<std::pair<int, int>> get_direction_streak(int row, int col, int row_dir, int col_dir, int player) const;
  bool is_in_board(int row, int col) const;

  bool is_opponent_won()const;
  bool check_row_winning(std::array<std::array<int8_t, COLS>, ROWS> const& opponent_yugos_board, int row, int col)const;
  bool check_col_winning(std::array<std::array<int8_t, COLS>, ROWS> const& opponent_yugos_board, int row, int col)const;
  bool check_forward_diagonal_winning(std::array<std::array<int8_t, COLS>, ROWS> const& opponent_yugos_board, int row, int col)const;
  bool check_backward_diagonal_winning(std::array<std::array<int8_t, COLS>, ROWS> const& opponent_yugos_board, int row, int col)const;
  bool has_legal_action()const;



public:
  MigoyugoState(std::array<std::array<int8_t, COLS>, ROWS> board, int player);
  ~MigoyugoState() override;
  static std::unique_ptr<MigoyugoState> initialize_state();
  static std::unique_ptr<rl::common::IState> initialize();
  std::unique_ptr<rl::common::IState> reset() const override;
  std::unique_ptr<MigoyugoState> reset_state() const;
  std::unique_ptr<rl::common::IState> step(int action) const override;
  std::unique_ptr<MigoyugoState> step_state(int action) const;
  void render() const override;
  bool is_terminal() const override;
  float get_reward() const override;
  std::vector<float> get_observation() const override;
  std::string to_short() const override;
  std::array<int, 3> get_observation_shape() const override;
  int get_n_actions() const override;
  int player_turn() const override;
  std::vector<bool> actions_mask() const override;

  std::unique_ptr<MigoyugoState> clone_state() const;
  std::unique_ptr<rl::common::IState> clone() const override;
  void get_symmetrical_obs_and_actions(std::vector<float> const& obs, std::vector<float> const& actions_distribution, std::vector<std::vector<float>>& out_syms, std::vector<std::vector<float>>& out_actions_distribution) const override;
  int encode_action(int row, int col)const;


};
}

namespace rl::games::miguyugo_syms
{
constexpr std::array<int, 256> FIRST_OBS_SYM =
{ {
  56, 48, 40, 32, 24, 16, 8, 0,
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
  127, 119, 111, 103, 95, 87, 79, 71,

  184, 176, 168, 160, 152, 144, 136, 128,
  185, 177, 169, 161, 153, 145, 137, 129,
  186, 178, 170, 162, 154, 146, 138, 130,
  187, 179, 171, 163, 155, 147, 139, 131,
  188, 180, 172, 164, 156, 148, 140, 132,
  189, 181, 173, 165, 157, 149, 141, 133,
  190, 182, 174, 166, 158, 150, 142, 134,
  191, 183, 175, 167, 159, 151, 143, 135,

  248, 240, 232, 224, 216, 208, 200, 192,
  249, 241, 233, 225, 217, 209, 201, 193,
  250, 242, 234, 226, 218, 210, 202, 194,
  251, 243, 235, 227, 219, 211, 203, 195,
  252, 244, 236, 228, 220, 212, 204, 196,
  253, 245, 237, 229, 221, 213, 205, 197,
  254, 246, 238, 230, 222, 214, 206, 198,
  255, 247, 239, 231, 223, 215, 207, 199
} };

constexpr std::array<int, 64> FIRST_ACTIONS_SYM =
{ {56, 48, 40, 32, 24, 16, 8, 0,
  57, 49, 41, 33, 25, 17, 9, 1,
  58, 50, 42, 34, 26, 18, 10, 2,
  59, 51, 43, 35, 27, 19, 11, 3,
  60, 52, 44, 36, 28, 20, 12, 4,
  61, 53, 45, 37, 29, 21, 13, 5,
  62, 54, 46, 38, 30, 22, 14, 6,
  63, 55, 47, 39, 31, 23, 15, 7,
  } };


constexpr std::array<int, 256> SECOND_OBS_SYM =
{ {
  63, 62, 61, 60, 59, 58, 57, 56,
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
  71, 70, 69, 68, 67, 66, 65, 64,


191, 190, 189, 188, 187, 186, 185, 184,
183, 182, 181, 180, 179, 178, 177, 176,
175, 174, 173, 172, 171, 170, 169, 168,
167, 166, 165, 164, 163, 162, 161, 160,
159, 158, 157, 156, 155, 154, 153, 152,
151, 150, 149, 148, 147, 146, 145, 144,
143, 142, 141, 140, 139, 138, 137, 136,
135, 134, 133, 132, 131, 130, 129, 128,

255, 254, 253, 252, 251, 250, 249, 248,
 247, 246, 245, 244, 243, 242, 241, 240,
 239, 238, 237, 236, 235, 234, 233, 232,
 231, 230, 229, 228, 227, 226, 225, 224,
 223, 222, 221, 220, 219, 218, 217, 216,
 215, 214, 213, 212, 211, 210, 209, 208,
 207, 206, 205, 204, 203, 202, 201, 200,
 199, 198, 197, 196, 195, 194, 193, 192
} };
constexpr std::array<int, 64> SECOND_ACTIONS_SYM =
{ {63, 62, 61, 60, 59, 58, 57, 56,
  55, 54, 53, 52, 51, 50, 49, 48,
  47, 46, 45, 44, 43, 42, 41, 40,
  39, 38, 37, 36, 35, 34, 33, 32,
  31, 30, 29, 28, 27, 26, 25, 24,
  23, 22, 21, 20, 19, 18, 17, 16,
  15, 14, 13, 12, 11, 10, 9, 8,
  7, 6, 5, 4, 3, 2, 1, 0
  } };

constexpr std::array<int, 256> THIRD_OBS_SYM =
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
  64, 72, 80, 88, 96, 104, 112, 120,

    135, 143, 151, 159, 167, 175, 183, 191,
    134, 142, 150, 158, 166, 174, 182, 190,
    133, 141, 149, 157, 165, 173, 181, 189,
    132, 140, 148, 156, 164, 172, 180, 188,
    131, 139, 147, 155, 163, 171, 179, 187,
    130, 138, 146, 154, 162, 170, 178, 186,
    129, 137, 145, 153, 161, 169, 177, 185,
    128, 136, 144, 152, 160, 168, 176, 184,

    199, 207, 215, 223, 231, 239, 247, 255,
     198, 206, 214, 222, 230, 238, 246, 254,
     197, 205, 213, 221, 229, 237, 245, 253,
      196, 204, 212, 220, 228, 236, 244, 252,
       195, 203, 211, 219, 227, 235, 243, 251,
       194, 202, 210, 218, 226, 234, 242, 250,
       193, 201, 209, 217, 225, 233, 241, 249,
       192, 200, 208, 216, 224, 232, 240, 248
} };
constexpr std::array<int, 64> THIRD_ACTIONS_SYM =
{ {7, 15, 23, 31, 39, 47, 55, 63,
  6, 14, 22, 30, 38, 46, 54, 62,
  5, 13, 21, 29, 37, 45, 53, 61,
  4, 12, 20, 28, 36, 44, 52, 60,
  3, 11, 19, 27, 35, 43, 51, 59,
  2, 10, 18, 26, 34, 42, 50, 58,
  1, 9, 17, 25, 33, 41, 49, 57,
  0, 8, 16, 24, 32, 40, 48, 56
  } };
}

#endif