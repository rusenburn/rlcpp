#ifndef RL_UI_MIGOYUGO_UI_HPP_
#define RL_UI_MIGOYUGO_UI_HPP_

#include <memory>
#include <vector>
#include <chrono>
#include <games/migoyugo.hpp>
#include <common/player.hpp>
#include <players/players.hpp>
#include <raylib.h>
#include "../IGameui.hpp"
#include "../players_utils.hpp"

namespace rl::ui
{
enum class MigoyugoWindow
{
    menu,
    game
};

using MigoyugoStatePtr = std::unique_ptr<rl::games::MigoyugoState>;
using IPlayerPtr = std::unique_ptr<rl::common::IPlayer>;

class MigoyugoUI : public IGameui
{
private:
    int width_;
    int height_;
    int padding_;
    int cell_size_;
    int inner_cell_size_;
    MigoyugoStatePtr state_ptr_;
    MigoyugoWindow current_window_;
    std::vector<std::unique_ptr<PlayerInfoFull>> players_;
    std::vector<float> obs_;
    std::vector<bool> actions_legality_;
    std::vector<std::pair<Rectangle, Color>> buttons_;
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
    void perform_player_action(int row, int col);
    void draw_piece(int left, int top, int player, bool is_fade);
    void draw_legal_actions();

    std::unique_ptr<rl::players::AmctsPlayer> get_network_amcts_player(int n_sims, std::chrono::duration<int, std::milli> minimum_duration, std::string load_name);

public:
    MigoyugoUI(int width, int height);
    ~MigoyugoUI() override;
    void draw_game() override;
    void handle_events() override;
    void set_state(MigoyugoStatePtr new_state_ptr);
    void reset_state();
};
} // namespace rl::ui

#endif
