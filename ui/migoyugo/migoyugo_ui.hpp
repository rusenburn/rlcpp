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

enum class ToneWave
{
    Sine,
    Square
};

class MigoyugoUI : public IGameui
{
private:
    int width_;
    int height_;
    int padding_;
    int cell_size_;
    int inner_cell_size_;
    int left_margin_;
    int header_height_;
    MigoyugoStatePtr state_ptr_;
    MigoyugoWindow current_window_;
    std::vector<std::unique_ptr<PlayerInfoFull>> players_;
    std::vector<float> obs_;
    std::vector<bool> actions_legality_;
    std::vector<std::pair<Rectangle, Color>> buttons_;
    std::vector<int> history_{};
    bool game_over_;

    // Game-over bookkeeping
    int winner_; // -2 = no result yet, -1 = draw, 0/1 = winning player index
    Rectangle exit_button_rect_;
    Rectangle rematch_button_rect_;

    // Score tracking, indices line up with players_
    std::vector<int> wins_;
    int draws_;

    // Audio
    Sound move_sound_;
    Sound win_sound_;
    Sound draw_sound_;

    // Player selection UI variables
    std::string selected_player_type_;
    std::string duration_input_;
    std::string loadname_input_;
    bool duration_input_focused_;
    bool loadname_input_focused_;
    int player_type_index_;

    void initialize_buttons();
    void draw_board();
    void draw_header();
    void draw_game_over_overlay();
    void draw_menu();
    void handle_board_events();
    void handle_menu_events();
    void perform_action(int action);
    void perform_player_action(int row, int col);
    void draw_piece(int left, int top, int player, bool is_fade);
    void draw_legal_actions();
    int compute_winner() const;
    void count_yugos(int& p0_yugos, int& p1_yugos) const;

    Wave synthesize_tone(float freq_hz, float duration_sec, ToneWave shape, float amplitude) const;
    Wave synthesize_sequence(const std::vector<std::pair<float, float>>& notes, ToneWave shape, float amplitude) const;
    Sound load_tone_sound(Wave wave) const;

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
