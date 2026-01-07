#ifndef RL_UI_GOBBLET_GOBLERS_HPP_
#define RL_UI_GOBBLET_GOBLERS_HPP_


#include <games/gobblet_goblers.hpp>
#include <raylib.h>
#include <common/player.hpp>
#include <players/players.hpp>
#include <chrono>
#include "../IGameui.hpp"
#include "../players_utils.hpp"
#include "gobblet_goblers_windows.hpp"
#include <deeplearning/network_evaluator.hpp>
#include <players/random_rollout_evaluator.hpp>
#include <deeplearning/alphazero/networks/shared_res_nn.hpp>


namespace rl::ui {

class GobbletGoblersUI : public IGameui
{
    using GobbletGoblersState = rl::games::GobbletGoblersState;
    using GobbletGoblersStatePtr = std::unique_ptr<rl::games::GobbletGoblersState>;
    using IPlayer = rl::common::IPlayer;
    using IPlayerPtr = std::unique_ptr<rl::common::IPlayer>;

public:
    GobbletGoblersUI(int width, int height);
    ~GobbletGoblersUI()override;
    void draw_game()override;
    void handle_events()override;

private:
    int width_, height_, cell_size_, padding_, inner_cell_size_;
    int selected_row_, selected_col_, selected_size_;
    GobbletGoblersWindow current_window_;
    GobbletGoblersStatePtr state_ptr_;
    bool paused_{ false };
    std::vector<std::pair<Rectangle, Color>> buttons_{};
    std::vector<std::unique_ptr<PlayerInfoFull>> players_{};
    double pause_until_{};
    std::vector<float> obs_;
    std::array<std::array<int, rl::games::GobbletGoblersState::COLS>, rl::games::GobbletGoblersState::ROWS> board_;
    std::vector<bool> actions_legality_;

    void set_state(GobbletGoblersStatePtr new_state_ptr);
    void reset_state();
    void initialize_buttons();
    void draw_board();
    void draw_menu();
    void handle_board_events();
    void handle_menu_events();
    void perform_action(int action);
    void perform_player_action(int row, int col);

    void draw_piece(int left_center, int top_center, int player);
    void draw_ground(int left_center, int top_center);
    void draw_legal_actions();
};
}

#endif