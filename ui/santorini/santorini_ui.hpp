#ifndef RL_UI_SANTORINI_HPP_
#define RL_UI_SANTORINI_HPP_

#include <games/santorini.hpp>
#include <memory>
#include <vector>
#include "santorini_ui_windows.hpp"
#include <common/player.hpp>
#include <raylib.h>
#include <players/players.hpp>
#include <chrono>
#include "../IGameui.hpp"
namespace rl::ui
{
    class SantoriniUI : public IGameui
    {
        using SantoriniState = rl::games::SantoriniState;
        using SantoriniPhase = rl::games::SantoriniPhase;
        using SantoriniStatePtr = std::unique_ptr<SantoriniState>;
        using IPlayer = rl::common::IPlayer;
        using IPlayerPtr = std::unique_ptr<rl::common::IPlayer>;

    public:
        SantoriniUI(int width, int height);
        ~SantoriniUI() override;
        void draw_game() override;
        void handle_events() override;

    private:
        int width_, height_, cell_size_, padding_, inner_cell_size_;
        SantoriniWindow current_window_;
        SantoriniStatePtr state_ptr_;
        SantoriniPhase phase_;
        int selected_row_,selected_col_;
        bool paused_{false};
        std::vector<std::pair<Rectangle, Color>> buttons_{};
        std::vector<std::unique_ptr<IPlayer>> players_{};
        double pause_until_{};
        std::vector<float> obs_;
        std::vector<bool> actions_legality_;
        void set_state(SantoriniStatePtr new_state_ptr);
        void reset_state();
        void initialize_buttons();
        void draw_board();
        void draw_menu();
        void handle_board_events();
        void handle_menu_events();
        void perform_action(int action);
        void perform_player_action(int row, int col);

        void draw_ground(int left, int top);
        void draw_floor1(int left, int top);
        void draw_floor2(int left, int top);
        void draw_floor3(int left, int top);
        void draw_dome(int left, int top);
        void draw_piece(int left,int top,int player,bool is_fade);
        void draw_legal_actions();
        std::unique_ptr<rl::players::GPlayer> get_default_g_player(int n_sims, std::chrono::duration<int, std::milli> minimum_duration);
        std::unique_ptr<rl::players::MctsPlayer> get_random_rollout_player_ptr(int n_sims, std::chrono::duration<int, std::milli> minimum_duration);
    };
} // namespace rl::ui

#endif
