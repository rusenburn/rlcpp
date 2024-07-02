#include "damma_ui.hpp"

#include <raylib.h>
#include <array>
#include <common/exceptions.hpp>
#include <players/random_rollout_evaluator.hpp>
#include <sstream>
#include <iostream>

namespace rl::ui
{

    void DammaUI::initialize_buttons()
    {
        float button_width = 100;
        float button_height = 20;

        float top, left;
        left = (width_ - button_width) / 2;
        top = 20;
        buttons_.push_back(std::make_pair<Rectangle, Color>(Rectangle{left, top, button_width, button_height}, GRAY));
    }

    void DammaUI::draw_board()
    {
        constexpr int ROWS = DammaState::ROWS;
        constexpr int COLS = DammaState::COLS;
        constexpr int CURRENT_PLAYER_P_CHANNEL = 0;
        constexpr int CURRENT_PLAYER_K_CHANNEL = 1;
        constexpr int OPPONENT_PLAYER_P_CHANNEL = 2;
        constexpr int OPPONENT_PLAYER_K_CHANNEL = 3;
        constexpr int CHANNEL_SIZE = ROWS * COLS;

        int left, top, width, height, left_center, top_center;

        int current_player = state_ptr_->player_turn();
        auto actions_legality = state_ptr_->actions_mask();

        int player_0_p_channel = current_player == 0 ? 0 : 2;
        int player_0_k_channel = player_0_p_channel + 1;
        int player_1_p_channel = current_player == 0 ? 2 : 0;
        int player_1_k_channel = player_1_p_channel + 1;
        int selected_row = -1;
        int selected_col = -1;

        if (selected_cell_.has_value())
        {
            selected_row = std::get<0>(selected_cell_.value());
            selected_col = std::get<1>(selected_cell_.value());
        }

        for (int row = 0; row < ROWS; row++)
        {
            int relative_row = current_player == 0 ? row : DammaState::ROWS - 1 - row;
            for (int col = 0; col < COLS; col++)
            {

                left = col * cell_size_ + padding_;
                top = relative_row * cell_size_ + padding_;
                left_center = left + inner_cell_size_ / 2;
                top_center = top + inner_cell_size_ / 2;

                DrawRectangle(left, top, inner_cell_size_, inner_cell_size_, DARKGREEN);
                Rectangle ol = {left, top, inner_cell_size_, inner_cell_size_};
                if (relative_row == selected_row && col == selected_col)
                {
                    DrawRectangleLinesEx(ol, 2.0f, RED);
                }
                else if (selectable_squares_.at(relative_row).at(col))
                {
                    DrawRectangleLinesEx(ol, 2.0f, GREEN);
                }

                int player_0_p_ind = player_0_p_channel * CHANNEL_SIZE + row * COLS + col;
                int player_0_k_ind = player_0_k_channel * CHANNEL_SIZE + row * COLS + col;
                int player_1_p_ind = player_1_p_channel * CHANNEL_SIZE + row * COLS + col;
                int player_1_k_ind = player_1_k_channel * CHANNEL_SIZE + row * COLS + col;

                if (obs_.at(player_0_p_ind) == 1)
                {
                    // DrawCircle(left_center, top_center, inner_cell_size_ / 3, BLACK);
                    draw_pawn(left,top,0,false);
                }
                else if (obs_.at(player_0_k_ind) == 1)
                {
                    // DrawCircle(left_center, top_center, inner_cell_size_ / 3, BLACK);
                    draw_king(left,top,0,false);
                }
                else if (obs_.at(player_1_p_ind) == 1)
                {
                    // DrawCircle(left_center, top_center, inner_cell_size_ / 3, WHITE);
                    draw_pawn(left,top,1,false);
                }
                else if (obs_.at(player_1_k_ind) == 1)
                {
                    // DrawCircle(left_center, top_center, inner_cell_size_ / 3, WHITE);
                    draw_king(left,top,1,false);
                }

                if (selected_cell_.has_value())
                {
                    auto [base_row, base_col] = selected_cell_.value();
                    int relative_base_row = current_player == 0 ? base_row : DammaState::ROWS - 1 - base_row;
                    if (squares_actions_legality_.at(base_row).at(base_col).at(relative_row).at(col))
                    {
                        if (current_player == 0)
                        {
                            // DrawCircle(left_center, top_center, inner_cell_size_ / 3, FADE_BLACK);
                            draw_pawn(left,top,0,true);
                        }
                        else
                        {
                            // DrawCircle(left_center, top_center, inner_cell_size_ / 3, FADE_WHITE);
                            draw_pawn(left,top,0,true);
                        }
                    }
                }
            }
        }
    }
    void DammaUI::draw_menu()
    {
        DrawRectangleRec(std::get<0>(buttons_.at(0)), std::get<1>(buttons_.at(0)));
    }

    void DammaUI::draw_pawn(int left, int top, int player,bool is_fade)
    {
        auto FADE_BLACK = BLACK;
        auto FADE_WHITE = WHITE;
        FADE_BLACK.a = 50;
        FADE_WHITE.a = 50;
        int left_center = left + inner_cell_size_ / 2;
        int top_center = top + inner_cell_size_ / 2;
        if (player == 0)
        {
            if (is_fade)
            {
                DrawCircle(left_center, top_center, inner_cell_size_ / 3, FADE_BLACK);
            }
            else
            {
                DrawCircle(left_center, top_center, inner_cell_size_ / 3, BLACK);
            }
        }
        else
        {
            if (is_fade)
            {
                DrawCircle(left_center, top_center, inner_cell_size_ / 3, FADE_WHITE);
            }
            else
            {
                DrawCircle(left_center, top_center, inner_cell_size_ / 3, WHITE);
            }
        }
    }

    void DammaUI::draw_king(int left, int top, int player, bool is_fade)
    {
        auto FADE_BLACK = BLACK;
        auto FADE_WHITE = WHITE;
        FADE_BLACK.a = 50;
        FADE_WHITE.a = 50;
        int left_center = left + inner_cell_size_ / 2;
        int top_center = top + inner_cell_size_ / 2;
        int rectangle_long = inner_cell_size_/4;
        int rectangle_short = 6;
        if (player == 0)
        {
            if (is_fade)
            {
                DrawCircle(left_center, top_center, inner_cell_size_ / 3, FADE_BLACK);
                // DRAW cross
                DrawRectangle(left + inner_cell_size_/2 - rectangle_long/2,top + inner_cell_size_/2 - rectangle_short/2,rectangle_long,rectangle_short,FADE_WHITE);
                DrawRectangle(left + inner_cell_size_/2 - rectangle_short/2,top + inner_cell_size_/2 - rectangle_long/2,rectangle_short,rectangle_long,FADE_WHITE);
            }
            else
            {
                DrawCircle(left_center, top_center, inner_cell_size_ / 3, BLACK);
                DrawRectangle(left + inner_cell_size_/2 - rectangle_long/2,top + inner_cell_size_/2 - rectangle_short/2,rectangle_long,rectangle_short,WHITE);
                DrawRectangle(left + inner_cell_size_/2 - rectangle_short/2,top + inner_cell_size_/2 - rectangle_long/2,rectangle_short,rectangle_long,WHITE);
            }
        }
        else
        {
            if (is_fade)
            {
                DrawCircle(left_center, top_center, inner_cell_size_ / 3, FADE_WHITE);
                DrawRectangle(left + inner_cell_size_/2 - rectangle_long/2,top + inner_cell_size_/2 - rectangle_short/2,rectangle_long,rectangle_short,FADE_BLACK);
                DrawRectangle(left + inner_cell_size_/2 - rectangle_short/2,top + inner_cell_size_/2 - rectangle_long/2,rectangle_short,rectangle_long,FADE_BLACK);
            }
            else
            {
                DrawCircle(left_center, top_center, inner_cell_size_ / 3, WHITE);
                DrawRectangle(left + inner_cell_size_/2 - rectangle_long/2,top + inner_cell_size_/2 - rectangle_short/2,rectangle_long,rectangle_short,BLACK);
                DrawRectangle(left + inner_cell_size_/2 - rectangle_short/2,top + inner_cell_size_/2 - rectangle_long/2,rectangle_short,rectangle_long,BLACK);
            }
        }
    }

    void DammaUI::handle_board_events()
    {
        if (!paused_)
        {
            int current_player_ind = state_ptr_->player_turn();
            IPlayerPtr &current_player_ptr_ref = players_.at(current_player_ind);
            auto player_p = dynamic_cast<const rl::players::HumanPlayer *>(current_player_ptr_ref.get());
            if (player_p != nullptr)
            {
                if (IsMouseButtonDown(MOUSE_BUTTON_LEFT))
                {
                    if (selected_cell_.has_value())
                    {
                        Vector2 mouse_position = GetMousePosition();
                        int target_row = mouse_position.y / cell_size_;
                        int target_col = mouse_position.x / cell_size_;
                        auto [row, col] = selected_cell_.value();
                        if (target_row == row && target_col == col)
                        {
                            selected_cell_.reset();
                        }
                        else if (selectable_squares_.at(target_row).at(target_col))
                        {
                            selected_cell_ = std::make_pair(target_row, target_col);
                        }
                        else
                        {
                            if (state_ptr_->player_turn() != 0)
                            {
                                row = DammaState::ROWS - 1 - row;
                                target_row = DammaState::ROWS - 1 - target_row;
                            }
                            if (row != target_row || col != target_col)
                            {
                                try
                                {
                                    int action = DammaState::encode_action(row, col, target_row, target_col);
                                    perform_action(action);
                                }
                                catch (const rl::common::UnreachableCodeException &e)
                                {
                                    std::cerr << e.what() << '\n';
                                }
                            }
                        }
                    }
                    else
                    {
                        Vector2 mouse_position = GetMousePosition();
                        int row = mouse_position.y / cell_size_;
                        int col = mouse_position.x / cell_size_;
                        selected_cell_.emplace(std::pair(row, col));
                    }
                }
            }
            else
            {
                int action = current_player_ptr_ref->choose_action(state_ptr_->clone());
                perform_action(action);
            }
            int action_padding = 20;

            if (state_ptr_->is_terminal())
            {
                paused_ = true;
                pause_until_ = GetTime() + 5;
            }
        }
        else
        {
            double current_time = GetTime();
            if (current_time > pause_until_)
            {
                current_window = DammaWindow::menu;
                paused_ = false;
            }
        }
    }
    void DammaUI::handle_menu_events()
    {
        if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON))
        {
            auto mouse_pos = GetMousePosition();
            auto [rec, col] = buttons_.at(0);
            auto is_button_pressed = CheckCollisionPointRec(mouse_pos, rec);
            if (is_button_pressed)
            {
                set_state(state_ptr_->reset_state());
                current_window = DammaWindow::game;
                players_.clear();
                auto player_g_duration = std::chrono::duration<int, std::milli>(1000);
                // players_.push_back(get_human_player());
                players_.push_back(get_human_player());
                players_.push_back(get_random_rollout_player_ptr(3, player_g_duration));
                // players_.push_back(get_random_rollout_player_ptr(3, player_g_duration));
                // players_.push_back(get_default_g_player(3, player_g_duration));
            }
        };
    }
    void DammaUI::set_state(std::unique_ptr<DammaState> state_ptr)
    {
        state_ptr_ = std::move(state_ptr);
        obs_ = state_ptr_->get_observation();
        actions_legality_ = state_ptr_->actions_mask();
        int current_player = state_ptr_->player_turn();
        selectable_squares_ = std::vector(DammaState::ROWS, std::vector<bool>(DammaState::COLS, 0));
        squares_actions_legality_ = std::vector(
            DammaState::ROWS, std::vector(
                                  DammaState::COLS, std::vector(
                                                        DammaState::ROWS, std::vector<bool>(
                                                                              DammaState::COLS, false))));
        for (int action = 0; action < actions_legality_.size(); action++)
        {
            if (actions_legality_.at(action))
            {
                auto [row, col, target_row, target_col] = DammaState::decode_action(action);
                int relative_row = current_player == 0 ? row : DammaState::ROWS - 1 - row;
                selectable_squares_.at(relative_row).at(col) = true;

                int relative_target_row = current_player == 0 ? target_row : DammaState::ROWS - 1 - target_row;
                squares_actions_legality_.at(relative_row).at(col).at(relative_target_row).at(target_col) = true;
            }
        }
    }
    void DammaUI::perform_action(int action)
    {
        auto actions_legality = state_ptr_->actions_mask();
        if (action < actions_legality.size() && actions_legality.at(action) && state_ptr_->is_terminal() == false)
        {
            set_state(state_ptr_->step_state(action));
        }
    }
    std::unique_ptr<rl::players::GPlayer> DammaUI::get_default_g_player(int n_sims, std::chrono::duration<int, std::milli> minimum_duration)
    {
        return std::make_unique<rl::players::GPlayer>(n_sims, minimum_duration, 15, 0.04f);
    }

    std::unique_ptr<rl::players::MctsPlayer> DammaUI::get_random_rollout_player_ptr(int n_sims, std::chrono::duration<int, std::milli> minimum_duration)
    {
        auto ev_ptr = std::make_unique<rl::players::RandomRolloutEvaluator>(state_ptr_->get_n_actions());
        return std::make_unique<rl::players::MctsPlayer>(state_ptr_->get_n_actions(), ev_ptr->copy(), n_sims, minimum_duration, 1.0f, 2.0f);
    }

    std::unique_ptr<rl::players::HumanPlayer> DammaUI::get_human_player()
    {
        return std::make_unique<rl::players::HumanPlayer>();
    }

    DammaUI::DammaUI(int width, int height)
        : width_{width * 4 / 5}, height_{height * 4 / 5}, padding_{2},
          state_ptr_{rl::games::DammaState::initialize_state()},
          current_window{DammaWindow::menu},
          players_{},
          selected_cell_{},
          selectable_squares_(std::vector(DammaState::ROWS, std::vector<bool>(DammaState::COLS, 0))),
          squares_actions_legality_(std::vector(
              DammaState::ROWS, std::vector(
                                    DammaState::COLS, std::vector(
                                                          DammaState::ROWS, std::vector<bool>(
                                                                                DammaState::COLS, false)))))

    {
        set_state(rl::games::DammaState::initialize_state());
        cell_size_ = width_ / DammaState::ROWS;
        inner_cell_size_ = cell_size_ - 2 * padding_;
        initialize_buttons();
    }
    DammaUI::~DammaUI() = default;

    void DammaUI::draw_game()
    {
        if (current_window == DammaWindow::game)
        {
            draw_board();
        }
        else if (current_window == DammaWindow::menu)
        {
            draw_menu();
        }
    }
    void DammaUI::handle_events()
    {
        if (current_window == DammaWindow::game)
        {
            handle_board_events();
        }
        else if (current_window == DammaWindow::menu)
        {
            handle_menu_events();
        }
    }
}
