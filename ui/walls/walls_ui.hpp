#ifndef RL_UI_WALLS_UI_HPP_
#define RL_UI_WALLS_UI_HPP_

#include <games/walls.hpp>
#include <memory>
#include <vector>
#include "walls_ui_windows.hpp"
#include <common/player.hpp>
#include <raylib.h>
#include <players/players.hpp>
#include <chrono>
#include "../IGameui.hpp"
namespace rl::ui
{

    class WallsUi : public IGameui
    {
        using WallsState = rl::games::WallsState;
        using WallsStatePtr = std::unique_ptr<rl::games::WallsState>;
        using IPlayer = rl::common::IPlayer;
        using IPlayerPtr = std::unique_ptr<rl::common::IPlayer>;

    public:
        WallsUi(int width,int height);
        ~WallsUi()override;
        void draw_game()override;
        void handle_events()override;
    private:
        int width_, height_, cell_size_, padding_, inner_cell_size_;
        WallsWindow current_window;
        std::unique_ptr<rl::games::WallsState> state_ptr_;
        bool paused_{false};
        std::vector<std::pair<Rectangle, Color>> buttons_{};
        std::vector<std::unique_ptr<IPlayer>> players_{};
        double pause_until_{};
        std::vector<float> obs_;
        std::vector<bool> actions_legality_;
        std::vector<bool> jumping_legality_;
        std::vector<std::vector<bool>> building_legality_;

        bool is_jumping_phase_;
        bool is_building_phase_;
        std::vector<std::vector<bool>> valid_jumps_;
        std::vector<std::vector<std::vector<std::pair<int,int>>>> valid_builds_;
        void set_state(WallsStatePtr new_state_ptr);
        void reset_state();
        void initialize_buttons();
        void draw_board();
        void draw_menu();
        void handle_board_events();
        void handle_menu_events();
        void perform_action(int action);
        void perform_player_action(int row,int col);
        std::unique_ptr<rl::players::GPlayer> get_default_g_player(int n_sims, std::chrono::duration<int, std::milli> minimum_duration);
        std::unique_ptr<rl::players::MctsPlayer> get_random_rollout_player_ptr(int n_sims, std::chrono::duration<int, std::milli> minimum_duration);
    };

} // namespace rl::ui

#endif