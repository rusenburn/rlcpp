#if !defined(RL_UI_OTHELLO_UI_HPP_)
#define RL_UI_OTHELLO_UI_HPP_

#include <games/othello.hpp>
#include <memory>
#include <vector>
#include "othello_ui_windows.hpp"
#include <common/player.hpp>
#include <raylib.h>
#include <players/players.hpp>
#include <chrono>
#include "../IGameui.hpp"

namespace rl::ui
{
class OthelloUI : public IGameui
{
    using OthelloState = rl::games::OthelloState;
    using IPlayer = rl::common::IPlayer;
    using IPlayerPtr = std::unique_ptr<rl::common::IPlayer>;

private:
    int width_, height_, cell_size_, padding_, inner_cell_size_;
    bool paused_{ false };
    double pause_until_{ 0.0 };
    OthelloWindow current_window;
    std::vector<std::pair<Rectangle, Color>> buttons_{};
    std::unique_ptr<rl::games::OthelloState> state_ptr_;
    std::vector<std::unique_ptr<IPlayer>> players_{};
    void initialize_buttons();
    void draw_board();
    void draw_menu();
    void draw_score();
    void handle_board_events();
    void handle_menu_events();
    void perform_action(int action);
    std::unique_ptr<rl::players::GPlayer> get_default_g_player(int n_sims, std::chrono::duration<int, std::milli> minimum_duration);
    std::unique_ptr<rl::players::MctsPlayer> get_random_rollout_player_ptr(int n_sims, std::chrono::duration<int, std::milli> minimum_duration);
    std::pair<int, int> get_scores();

public:
    OthelloUI(int width, int height);
    ~OthelloUI() override;
    void draw_game() override;
    void handle_events() override;
};

} // namespace rl::ui

#endif // RL_UI_OTHELLO_UI_HPP_
