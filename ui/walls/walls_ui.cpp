#include "walls_ui.hpp"
#include <players/random_rollout_evaluator.hpp>
#include <players/players.hpp>
#include <deeplearning/alphazero/networks/shared_res_nn.hpp>
#include <filesystem>
#include "../players_utils.hpp"
namespace rl::ui
{

WallsUi::WallsUi(int width, int height)
    : width_{ width },
    height_{ height },
    padding_{ 2 },
    state_ptr_{ rl::games::WallsState::initialize_state() },
    current_window{ WallsWindow::menu },
    players_{}
{
    cell_size_ = width_ / 7;
    inner_cell_size_ = cell_size_ - 2 * padding_;
    initialize_buttons();
    reset_state();
}

WallsUi::~WallsUi() = default;

void WallsUi::draw_game()
{
    if (current_window == WallsWindow::game)
    {
        draw_board();
    }
    else if (current_window == WallsWindow::menu)
    {
        draw_menu();
    }
}

void WallsUi::handle_events()
{
    if (current_window == WallsWindow::game)
    {
        handle_board_events();
    }
    else if (current_window == WallsWindow::menu)
    {
        handle_menu_events();
    }
}

void WallsUi::set_state(WallsStatePtr new_state_ptr)
{
    is_jumping_phase_ = true;
    is_building_phase_ = false;
    state_ptr_ = std::move(new_state_ptr);
    obs_ = state_ptr_->get_observation();
    actions_legality_ = state_ptr_->actions_mask();

    std::vector<std::vector<bool>> valid_jumps(7, std::vector<bool>(7, false));
    std::vector<std::vector<std::vector<std::pair<int, int>>>> valid_builds(7,
        std::vector<std::vector<std::pair<int, int>>>(7,
            std::vector<std::pair<int, int>>()));
    state_ptr_->get_valid_jumps(valid_jumps, valid_builds);

    valid_jumps_ = valid_jumps;
    valid_builds_ = valid_builds;
}

void WallsUi::reset_state()
{
    set_state(state_ptr_->reset_state());
}

void WallsUi::initialize_buttons()
{
    float button_width = 100;
    float button_height = 20;

    float top, left;
    left = (width_ - button_width) / 2;
    top = 20;
    buttons_.push_back(std::make_pair<Rectangle, Color>(Rectangle{ left, top, button_width, button_height }, GRAY));
}

void WallsUi::draw_board()
{
    const int n_rows = 7;
    const int n_cols = 7;
    int left, top, width, height, left_center, top_center;
    int current_player = state_ptr_->player_turn();
    Color FADE_BLUE = BLUE;
    FADE_BLUE.a = 50;
    Color FADE_RED = RED;
    FADE_RED.a = 50;
    Color FADE_GRAY = GRAY;
    FADE_GRAY.a = 50;

    int player_row, player_col;
    for (int ind = 0; ind < 7 * 7; ind++)
    {
        if (obs_.at(ind) == 1)
        {
            player_row = ind / 7;
            player_col = ind % 7;
            break;
        }
    }

    for (int row = 0; row < n_rows; row++)
    {
        for (int col = 0; col < n_cols; col++)
        {
            left = col * cell_size_ + padding_;
            top = row * cell_size_ + padding_;
            left_center = left + inner_cell_size_ / 2;
            top_center = top + inner_cell_size_ / 2;
            DrawRectangle(left, top, inner_cell_size_, inner_cell_size_, DARKGREEN);
            int player_0_ind = current_player == 0 ? 0 * 7 * 7 + row * 7 + col : 1 * 7 * 7 + row * 7 + col;
            int player_1_ind = current_player == 0 ? 1 * 7 * 7 + row * 7 + col : 0 * 7 * 7 + row * 7 + col;
            int walls_ind = 2 * 7 * 7 + row * 7 + col;
            if (obs_.at(player_0_ind) == 1)
            {
                DrawCircle(left_center, top_center, inner_cell_size_ / 4, BLUE);
            }
            else if (obs_.at(player_1_ind) == 1)
            {
                DrawCircle(left_center, top_center, inner_cell_size_ / 4, RED);
            }
            else if (obs_.at(walls_ind) == 1)
            {
                DrawCircle(left_center, top_center, inner_cell_size_ / 3, GRAY);
            }
            else if (is_jumping_phase_ && valid_jumps_.at(row).at(col))
            {
                if (current_player == 0)
                {
                    DrawCircle(left_center, top_center, inner_cell_size_ / 4, FADE_BLUE);
                }
                else
                {
                    DrawCircle(left_center, top_center, inner_cell_size_ / 4, FADE_RED);
                }
            }
            else if (is_building_phase_)
            {
                for (auto [build_row, build_col] : valid_builds_.at(player_row).at(player_col))
                {
                    if (build_row == row && build_col == col)
                    {
                        DrawCircle(left_center, top_center, inner_cell_size_ / 3, FADE_GRAY);
                        break;
                    }
                }
            }
        }
    }
}

void WallsUi::draw_menu()
{
    DrawRectangleRec(std::get<0>(buttons_.at(0)), std::get<1>(buttons_.at(0)));
}

void WallsUi::handle_board_events()
{
    if (!paused_)
    {
        int current_player_ind = state_ptr_->player_turn();
        IPlayerPtr& current_player_ptr_ref = players_.at(current_player_ind)->player_ptr_;
        auto player_p = dynamic_cast<const rl::players::HumanPlayer*>(current_player_ptr_ref.get());
        if (player_p != nullptr)
        {
            if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT))
            {
                Vector2 mouse_position = GetMousePosition();
                int row, col;
                row = mouse_position.y / cell_size_;
                col = mouse_position.x / cell_size_;
                perform_player_action(row, col);
            }
        }
        else
        {
            int action = current_player_ptr_ref->choose_action(state_ptr_->clone_state());
            perform_action(action);
        }
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
            current_window = WallsWindow::menu;
            paused_ = false;
        }
    }
}

void WallsUi::handle_menu_events()
{
    if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON))
    {
        auto mouse_pos = GetMousePosition();
        auto [rec, col] = buttons_.at(0);
        auto is_button_pressed = CheckCollisionPointRec(mouse_pos, rec);
        if (is_button_pressed)
        {
            reset_state();
            current_window = WallsWindow::game;
            players_.clear();
            auto player_g_duration = std::chrono::duration<int, std::milli>(1000);
            players_.push_back(get_human_player(state_ptr_.get()));
            players_.push_back(get_network_amcts2_player(state_ptr_.get(), 3, player_g_duration, "walls_new_240_strongest.pt"));
        }
    }
}

void WallsUi::perform_action(int action)
{
    auto actions_legality = state_ptr_->actions_mask();
    if (action < actions_legality.size() && actions_legality.at(action) && state_ptr_->is_terminal() == false)
    {
        set_state(state_ptr_->step_state(action));
    }
}

void WallsUi::perform_player_action(int row, int col)
{
    if (is_jumping_phase_)
    {
        bool is_continue_searching = true;
        if (valid_jumps_.at(row).at(col) == true)
        {
            for (int i = 0; i < 7 && is_continue_searching; i++)
            {
                for (int o = 0; o < 7 && is_continue_searching; o++)
                {
                    int player_ind = i * 7 + o;
                    if (obs_.at(player_ind) == 1)
                    {
                        // to break the double loops
                        is_continue_searching = false;

                        // remove player
                        obs_.at(player_ind) = 0;

                        int new_index = row * 7 + col;
                        obs_.at(new_index) = 1;

                        is_jumping_phase_ = false;
                        is_building_phase_ = true;
                    }
                }
            }
        }
    }
    else if (is_building_phase_)
    {
        bool is_continue_searching = true;
        for (int i = 0; i < 7 && is_continue_searching; i++)
        {
            for (int o = 0; o < 7 && is_continue_searching; o++)
            {
                int player_ind = i * 7 + o;
                if (obs_.at(player_ind) == 1)
                {
                    // to break the double loops
                    is_continue_searching = false;

                    auto v_b = valid_builds_.at(i).at(o);
                    for (auto [valid_row, valid_col] : v_b)
                    {
                        if (valid_row == row && valid_col == col)
                        {
                            int action = state_ptr_->encode_action(i, o, row, col);
                            perform_action(action);
                        }
                    }
                }
            }
        }
    }
}

std::unique_ptr<rl::players::GPlayer> WallsUi::get_default_g_player(int n_sims, std::chrono::duration<int, std::milli> minimum_duration)
{

    return std::make_unique<rl::players::GPlayer>(n_sims, minimum_duration, 15, 0.04f);
}

std::unique_ptr<rl::players::MctsPlayer> WallsUi::get_random_rollout_player_ptr(int n_sims, std::chrono::duration<int, std::milli> minimum_duration)
{
    auto ev_ptr = std::make_unique<rl::players::RandomRolloutEvaluator>(state_ptr_->get_n_actions());
    return std::make_unique<rl::players::MctsPlayer>(state_ptr_->get_n_actions(), ev_ptr->copy(), n_sims, minimum_duration, 1.0f, 2.0f);
}
} // namespace rl::ui
