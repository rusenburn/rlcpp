#ifndef RL_GAMES_MIGOYUGO_LIGHT_HPP_
#define RL_GAMES_MIGOYUGO_LIGHT_HPP_


#include <common/state.hpp>
#include <memory>
#include <vector>
#include <array>
#include <optional>

namespace rl::games
{


struct NNUEUpdate {
  std::vector<int> white_added;
  std::vector<int> white_removed;
  std::vector<int> black_added;
  std::vector<int> black_removed;
};
class MigoyugoLightState : public rl::common::IState
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
  int step_;
  int last_action_;
  mutable std::vector<bool> cached_actions_masks_{};
  mutable std::optional<bool> cached_is_terminal_;
  mutable std::optional<float> cached_result_;
  mutable std::vector<float> cached_observation_{};
  mutable std::optional<std::string>cached_short_{};

  int get_streak_count(int row, int col, int row_dir, int col_dir, int player) const;
  bool is_in_board(int row, int col) const;

  bool is_opponent_won()const;
  bool check_row_winning(std::array<std::array<int8_t, COLS>, ROWS> const& opponent_yugos_board, int row, int col)const;
  bool check_col_winning(std::array<std::array<int8_t, COLS>, ROWS> const& opponent_yugos_board, int row, int col)const;
  bool check_forward_diagonal_winning(std::array<std::array<int8_t, COLS>, ROWS> const& opponent_yugos_board, int row, int col)const;
  bool check_backward_diagonal_winning(std::array<std::array<int8_t, COLS>, ROWS> const& opponent_yugos_board, int row, int col)const;
  bool has_legal_action()const;



public:
  MigoyugoLightState(std::array<std::array<int8_t, COLS>, ROWS> board, int player, int step, int last_action);
  ~MigoyugoLightState() override;
  static std::unique_ptr<MigoyugoLightState> initialize_state();
  static std::unique_ptr<rl::common::IState> initialize();
  std::unique_ptr<rl::common::IState> reset() const override;
  std::unique_ptr<MigoyugoLightState> reset_state() const;
  std::unique_ptr<rl::common::IState> step(int action) const override;
  std::unique_ptr<MigoyugoLightState> step_state(int action) const;
  std::unique_ptr<MigoyugoLightState> step_state_light(int action,NNUEUpdate& update)const;
  void render() const override;
  bool is_terminal() const override;
  float get_reward() const override;
  std::vector<float> get_observation() const override;
  std::string to_short() const override;
  std::array<int, 3> get_observation_shape() const override;
  int get_n_actions() const override;
  int player_turn() const override;
  std::vector<bool> actions_mask() const override;

  std::unique_ptr<MigoyugoLightState> clone_state() const;
  std::unique_ptr<rl::common::IState> clone() const override;
  void get_symmetrical_obs_and_actions(std::vector<float> const& obs, std::vector<float> const& actions_distribution, std::vector<std::vector<float>>& out_syms, std::vector<std::vector<float>>& out_actions_distribution) const override;
  static int encode_action(int row, int col);
  int get_last_action()const;


};
}

#endif
