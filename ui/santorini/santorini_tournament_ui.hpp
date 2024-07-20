#ifndef RL_COMMON_UI_SANTORINI_TOURNAMENT_UI_HPP_
#define RL_COMMON_UI_SANTORINI_TOURNAMENT_UI_HPP_

#include <memory>
#include <string>
#include <thread>
#include "../IGameui.hpp"
#include "../players_utils.hpp"
#include <common/state.hpp>
#include <games/santorini.hpp>
#include <common/observer.hpp>
#include <common/round_robin.hpp>

namespace rl::ui
{

    class SantoriniTournamentUI : public IGameui
    {
        using IPlayer = rl::common::IPlayer;
        using SantoriniState = rl::games::SantoriniState;
        using SantoriniPhase = rl::games::SantoriniPhase;
        using StateObserver = rl::common::Observer<rl::common::MatchInfo>;

    public:
        SantoriniTournamentUI(int width, int height);
        ~SantoriniTournamentUI();
        void draw_game() override;
        void handle_events() override;
        void init();
        void dispose();

    private:
        int board_width_, board_height_, cell_size_, padding_, inner_cell_size_;
        int secondary_tap_width_,secondary_tap_height_ ;
        int footing_width_, footing_height_;
        int player1_ = -1;
        int player2_ = -1;
        std::unique_ptr<std::thread> t_ptr_{nullptr};
        bool paused_{false};
        double pause_until_{};
        std::unique_ptr<rl::common::RoundRobin> round_robin_ptr_;
        std::unique_ptr<SantoriniState> state_ptr_;
        std::vector<std::unique_ptr<rl::ui::PlayerInfoFull>> players_;
        std::vector<float> obs_;
        std::vector<bool> actions_legality_;
        SantoriniPhase phase_;
        int current_player_;
        int selected_row_;
        int selected_col_;

        std::shared_ptr<StateObserver> state_observer_ptr_;
        std::unique_ptr<rl::common::RoundRobin> get_new_round_robin();
        void set_state_ptr(std::unique_ptr<SantoriniState> &new_state_ptr);
        void on_state_changed(rl::common::MatchInfo match_info);

        void draw_board();
        void draw_players_scores();
        void draw_players_name();
        void handle_board_events();
        void draw_ground(int left, int top);
        void draw_floor1(int left, int top);
        void draw_floor2(int left, int top);
        void draw_floor3(int left, int top);
        void draw_dome(int left, int top);
        void draw_piece(int left,int top,int player,bool is_fade);
        void draw_legal_actions();
    };
} // namespace rl::ui

#endif
