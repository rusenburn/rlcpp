#ifndef RL_UI_DAMMA_UI_HPP_
#define RL_UI_DAMMA_UI_HPP_

#include <games/damma.hpp>
#include <common/player.hpp>
#include <players/players.hpp>
#include <optional>
#include <utility>
#include <memory>
#include <vector>
#include "damma_ui_windows.hpp"
#include <raylib.h>
#include "../IGameui.hpp"

namespace rl::ui
{
class DammaUI : public IGameui
{
    using DammaState = rl::games::DammaState;
    using IPlayer = rl::common::IPlayer;
    using IPlayerPtr = std::unique_ptr<rl::common::IPlayer>;

private:
    int width_, height_, cell_size_, padding_, inner_cell_size_;
    std::optional<std::pair<int, int>> selected_cell_;
    bool paused_{ false };
    double pause_until_{ 0.0 };
    DammaWindow current_window;
    std::vector<std::pair<Rectangle, Color>> buttons_{};
    std::unique_ptr<DammaState> state_ptr_;
    std::vector<float> obs_;
    std::vector<bool> actions_legality_;
    std::vector<std::vector<bool>> selectable_squares_;
    std::vector<std::vector<std::vector<std::vector<bool>>>> squares_actions_legality_;
    std::vector<IPlayerPtr> players_{};
    void initialize_buttons();
    void draw_board();
    void draw_menu();
    void draw_pawn(int left, int top, int player, bool is_fade);
    void draw_king(int left, int top, int player, bool is_fade);
    void handle_board_events();
    void handle_menu_events();
    void set_state(std::unique_ptr<DammaState> state_ptr);
    void perform_action(int action);
    std::unique_ptr<rl::players::GPlayer> get_default_g_player(int n_sims, std::chrono::duration<int, std::milli> minimum_duration);
    std::unique_ptr<rl::players::MctsPlayer> get_random_rollout_player_ptr(int n_sims, std::chrono::duration<int, std::milli> minimum_duration);
    std::unique_ptr<rl::players::HumanPlayer> get_human_player();

public:
    DammaUI(int width, int height);
    ~DammaUI();
    void draw_game() override;
    void handle_events() override;
};
} // namespace rl::ui

#endif