#ifndef RL_UI_ULTIMATE_TICTACTOE_UI_HPP_
#define RL_UI_ULTIMATE_TICTACTOE_UI_HPP_

#include <memory>
#include <vector>
#include <chrono>
#include <games/ultimate_tictactoe.hpp>
#include <common/player.hpp>
#include <players/players.hpp>
#include <raylib.h>
#include "../IGameui.hpp"
#include "../players_utils.hpp"

namespace rl::ui
{
enum class UltimateTicTacToeWindow
{
    menu,
    game
};

using UltimateTicTacToeStatePtr = std::unique_ptr<rl::games::UltimateTicTacToeState>;
using IPlayerPtr = std::unique_ptr<rl::common::IPlayer>;

class UltimateTicTacToeUI : public IGameui
{
private:
    int width_;
    int height_;
    int padding_;
    int cell_size_;
    int inner_cell_size_;
    UltimateTicTacToeStatePtr state_ptr_;
    UltimateTicTacToeWindow current_window_;
    std::vector<std::unique_ptr<PlayerInfoFull>> players_;
    std::vector<float> obs_;
    std::vector<bool> actions_legality_;
    std::vector<std::pair<Rectangle, Color>> buttons_;
    std::vector<int> history_{};
    bool paused_;
    double pause_until_;

    // Player selection UI variables
    std::string selected_player_type_;
    std::string duration_input_;
    std::string loadname_input_;
    bool duration_input_focused_;
    bool loadname_input_focused_;
    int player_type_index_;

    void initialize_buttons();
    void draw_board();
    void draw_menu();
    void handle_board_events();
    void handle_menu_events();
    void perform_action(int action);
    void perform_player_action(int action);
    void draw_piece(int left, int top, int player);
    void draw_legal_actions();
    void draw_ultimate_board_owner(int left, int top, int owner);

public:
    UltimateTicTacToeUI(int width, int height);
    ~UltimateTicTacToeUI() override;
    void draw_game() override;
    void handle_events() override;
    void set_state(UltimateTicTacToeStatePtr new_state_ptr);
    void reset_state();
};
} // namespace rl::ui

#endif