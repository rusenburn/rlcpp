#include "othello_ui.hpp"
#include <raylib.h>
#include <array>
#include <players/random_rollout_evaluator.hpp>
#include <sstream>
#include <iostream>
namespace rl::ui
{
void OthelloUI::initialize_buttons()
{
    float button_width = 100;
    float button_height = 20;

    float top, left;
    left = (width_ - button_width) / 2;
    top = 20;
    buttons_.push_back(std::make_pair<Rectangle, Color>(Rectangle{ left, top, button_width, button_height }, GRAY));
}

void OthelloUI::draw_board()
{
    const int n_rows = 8;
    const int n_cols = 8;
    auto obs = state_ptr_->get_observation();
    int left, top, width, height, left_center, top_center;
    int current_player = state_ptr_->player_turn();
    auto actions_legality = state_ptr_->actions_mask();
    for (int row = 0; row < 8; row++)
    {
        for (int col = 0; col < 8; col++)
        {
            left = col * cell_size_ + padding_;
            top = row * cell_size_ + padding_;
            left_center = left + inner_cell_size_ / 2;
            top_center = top + inner_cell_size_ / 2;
            DrawRectangle(left, top, inner_cell_size_, inner_cell_size_, DARKGREEN);

            int player_0_ind = current_player == 0 ? 0 * 8 * 8 + row * 8 + col : 1 * 8 * 8 + row * 8 + col;
            int player_1_ind = current_player == 0 ? 1 * 8 * 8 + row * 8 + col : 0 * 8 * 8 + row * 8 + col;
            int action = row * 8 + col;
            if (obs.at(player_0_ind) == 1)
            {
                DrawCircle(left_center, top_center, inner_cell_size_ / 2, BLACK);
            }
            else if (obs.at(player_1_ind) == 1)
            {
                DrawCircle(left_center, top_center, inner_cell_size_ / 2, WHITE);
            }

            else if (actions_legality.at(action))
            {
                if (current_player == 0)
                {
                    DrawCircle(left_center, top_center, inner_cell_size_ / 2, { 0, 0, 0, 50 });
                }
                else
                {
                    DrawCircle(left_center, top_center, inner_cell_size_ / 2, { 255, 255, 255, 50 });
                }
            }
        }
    }
}

void OthelloUI::draw_menu()
{
    DrawRectangleRec(std::get<0>(buttons_.at(0)), std::get<1>(buttons_.at(0)));
}

void OthelloUI::draw_score()
{
    char text[50];
    std::stringstream ss;

    int left, top;
    left = width_ + 5;
    top = 5;
    auto [player_1_score, player_2_score] = get_scores();
    ss << player_1_score << " : " << player_2_score;
    std::string a = ss.str();
    strcpy(text, a.c_str());
    DrawText(text, left, top, 20, WHITE);
}

void OthelloUI::handle_board_events()
{
    if (!paused_)
    {
        int current_player_ind = state_ptr_->player_turn();
        IPlayerPtr& current_player_ptr_ref = players_.at(current_player_ind);
        auto player_p = dynamic_cast<const rl::players::HumanPlayer*>(current_player_ptr_ref.get());
        if (player_p != nullptr)
        {
            auto actions_legality = state_ptr_->actions_mask();
            if (actions_legality.at(64))
            {
                perform_action(64);
            }
            if (IsMouseButtonDown(MOUSE_BUTTON_LEFT))
            {
                Vector2 mouse_position = GetMousePosition();
                int row = mouse_position.y / cell_size_;
                int col = mouse_position.x / cell_size_;
                int action = row * 8 + col;
                perform_action(action);
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
            current_window = OthelloWindow::menu;
            paused_ = false;
        }
    }
}

void OthelloUI::handle_menu_events()
{
    if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON))
    {
        auto mouse_pos = GetMousePosition();
        auto [rec, col] = buttons_.at(0);
        auto is_button_pressed = CheckCollisionPointRec(mouse_pos, rec);
        if (is_button_pressed)
        {
            state_ptr_ = state_ptr_->reset_state();
            current_window = OthelloWindow::game;
            players_.clear();
            auto player_g_duration = std::chrono::duration<int, std::milli>(1000);
            players_.push_back(get_random_rollout_player_ptr(3, player_g_duration));
            players_.push_back(get_default_g_player(3, player_g_duration));
        }
    };
}

void OthelloUI::perform_action(int action)
{
    auto actions_legality = state_ptr_->actions_mask();
    if (action < actions_legality.size() && actions_legality.at(action) && state_ptr_->is_terminal() == false)
    {
        state_ptr_ = state_ptr_->step_state(action);
    }
}

std::unique_ptr<rl::players::GPlayer> OthelloUI::get_default_g_player(int n_sims, std::chrono::duration<int, std::milli> minimum_duration)
{
    return std::make_unique<rl::players::GPlayer>(n_sims, minimum_duration, 15, 0.04f);
}

std::unique_ptr<rl::players::MctsPlayer> OthelloUI::get_random_rollout_player_ptr(int n_sims, std::chrono::duration<int, std::milli> minimum_duration)
{
    auto ev_ptr = std::make_unique<rl::players::RandomRolloutEvaluator>(state_ptr_->get_n_actions());
    return std::make_unique<rl::players::MctsPlayer>(state_ptr_->get_n_actions(), ev_ptr->copy(), n_sims, minimum_duration, 1.0f, 2.0f);
}

std::pair<int, int> OthelloUI::get_scores()
{
    int player_0_score = 0;
    int player_1_score = 0;
    auto obs = state_ptr_->get_observation();
    int current_player = state_ptr_->player_turn();

    for (int row = 0; row < 8; row++)
    {
        for (int col = 0; col < 8; col++)
        {
            int player_0_ind = row * 8 + col;
            int player_1_ind = 1 * 8 * 8 + row * 8 + col;
            if (obs.at(player_0_ind) == 1)
            {
                if (current_player == 0)
                {
                    player_0_score++;
                }
                else
                {
                    player_1_score++;
                }
            }
            else if (obs.at(player_1_ind) == 1)
            {
                if (current_player == 0)
                {
                    player_1_score++;
                }
                else
                {
                    player_0_score++;
                }
            }
        }
    }

    return std::make_pair(player_0_score, player_1_score);
}

OthelloUI::OthelloUI(int width, int height)
    : width_{ width * 4 / 5 }, height_{ height * 4 / 5 }, padding_{ 2 },
    state_ptr_{ rl::games::OthelloState::initialize_state() },
    current_window{ OthelloWindow::menu },
    players_{}
{
    cell_size_ = width_ / 8;
    inner_cell_size_ = cell_size_ - 2 * padding_;
    initialize_buttons();
}
OthelloUI::~OthelloUI() = default;

void OthelloUI::draw_game()
{
    if (current_window == OthelloWindow::game)
    {
        draw_board();
        draw_score();
    }
    else if (current_window == OthelloWindow::menu)
    {
        draw_menu();
    }
}
void OthelloUI::handle_events()
{
    if (current_window == OthelloWindow::game)
    {
        handle_board_events();
    }
    else if (current_window == OthelloWindow::menu)
    {
        handle_menu_events();
    }
}
} // namespace rl::ui
