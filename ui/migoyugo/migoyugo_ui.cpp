#include "migoyugo_ui.hpp"
#include <iostream>

namespace rl::ui
{
MigoyugoUI::MigoyugoUI(int width, int height)
    : width_{ width }, height_{ height }, padding_{ 2 }, state_ptr_{ rl::games::MigoyugoState::initialize_state() },
    current_window_{ MigoyugoWindow::menu },
    players_{},
    paused_{ false },
    pause_until_{ 0.0 }

{
    // Reserve space for coordinate labels: left margin for numbers, right/bottom for letters
    // Allocate space for row numbers on the left
    int left_margin = 30;  // Fixed space for row numbers
    int board_width = width_ - left_margin;
    cell_size_ = board_width / 8;
    inner_cell_size_ = cell_size_ - 2 * padding_;
    initialize_buttons();
    reset_state();
}

MigoyugoUI::~MigoyugoUI() = default;

void MigoyugoUI::draw_game()
{
    if (current_window_ == MigoyugoWindow::game)
    {
        draw_board();
    }
    else if (current_window_ == MigoyugoWindow::menu)
    {
        draw_menu();
    }
}

void MigoyugoUI::handle_events()
{

    if (current_window_ == MigoyugoWindow::game)
    {
        handle_board_events();
    }
    else if (current_window_ == MigoyugoWindow::menu)
    {
        handle_menu_events();
    }
}

void MigoyugoUI::set_state(MigoyugoStatePtr new_state_ptr)
{
    state_ptr_ = std::move(new_state_ptr);
    obs_ = state_ptr_->get_observation();
    actions_legality_ = state_ptr_->actions_mask();
}

void MigoyugoUI::reset_state()
{
    set_state(state_ptr_->reset_state());
}

void MigoyugoUI::initialize_buttons()
{
    float button_width = 100;
    float button_height = 20;

    float top, left;
    left = (width_ - button_width) / 2;
    top = 20;
    buttons_.push_back(std::make_pair<Rectangle, Color>(Rectangle{ left, top, button_width, button_height }, GRAY));
}

void MigoyugoUI::draw_board()
{
    constexpr int ROWS = 8;
    constexpr int COLS = 8;
    constexpr int OUR_MIGO_CHANNEL = 0;
    constexpr int OUR_YUGO_CHANNEL = 1;
    constexpr int OPP_MIGO_CHANNEL = 2;
    constexpr int OPP_YUGO_CHANNEL = 3;
    int left, top, width, height;

    int current_player = state_ptr_->player_turn();
    int last_action = state_ptr_->get_last_action();

    for (int row = 0; row < ROWS; row++)
    {
        for (int col = 0; col < COLS; col++)
        {
            left = col * cell_size_ + padding_;
            top = row * cell_size_ + padding_;

            // Draw cell background
            DrawRectangle(left, top, inner_cell_size_, inner_cell_size_, DARKGREEN);

            int channel_size = ROWS * COLS;
            int our_migo_ind = OUR_MIGO_CHANNEL * channel_size + row * COLS + col;
            int our_yugo_ind = OUR_YUGO_CHANNEL * channel_size + row * COLS + col;
            int opp_migo_ind = OPP_MIGO_CHANNEL * channel_size + row * COLS + col;
            int opp_yugo_ind = OPP_YUGO_CHANNEL * channel_size + row * COLS + col;

            // Map observation channels to consistent player colors
            // Observation is always from current player's perspective
            // We want consistent coloring: player 0 always black/darkblue, player 1 always white/darkgreen
            int actual_player_for_our, actual_player_for_opp;
            if (current_player == 0) {
                actual_player_for_our = 0; // "our" channels contain player 0 pieces
                actual_player_for_opp = 1; // "opp" channels contain player 1 pieces
            }
            else {
                actual_player_for_our = 1; // "our" channels contain player 1 pieces
                actual_player_for_opp = 0; // "opp" channels contain player 0 pieces
            }

            if (obs_.at(our_migo_ind) == 1.0f)
            {
                draw_piece(left, top, actual_player_for_our, false);
            }
            else if (obs_.at(our_yugo_ind) == 1.0f)
            {
                draw_piece(left, top, actual_player_for_our, true); // yugo
            }
            else if (obs_.at(opp_migo_ind) == 1.0f)
            {
                draw_piece(left, top, actual_player_for_opp, false);
            }
            else if (obs_.at(opp_yugo_ind) == 1.0f)
            {
                draw_piece(left, top, actual_player_for_opp, true); // yugo
            }

            // Highlight last action with red square
            int current_action = row * COLS + col;
            if (current_action == last_action && last_action >= 0) {
                Rectangle last_action_rect = { left, top, inner_cell_size_, inner_cell_size_ };
                DrawRectangleLinesEx(last_action_rect, 3.0f, RED);
            }
        }
    }

    // Draw coordinate labels
    // Column letters a-h at bottom
    for (int col = 0; col < COLS; col++) {
        left = col * cell_size_ + padding_ + inner_cell_size_ / 2 - 5;
        top = ROWS * cell_size_ + padding_ + 5;
        char letter = 'a' + col;
        DrawText(&letter, left, top, 16, BLACK);
    }

    // Row numbers 1-8 (1 at bottom, 8 at top)
    for (int row = 0; row < ROWS; row++) {
        left = 5;  // Fixed position within the left margin
        top = row * cell_size_ + padding_ + inner_cell_size_ / 2 - 8;  // Center vertically in the cell
        char number = '8' - row;  // 8 at top (row 0), 1 at bottom (row 7)
        DrawText(&number, left, top, 16, BLACK);
    }

    draw_legal_actions();
}

void MigoyugoUI::draw_menu()
{
    DrawRectangleRec(std::get<0>(buttons_.at(0)), std::get<1>(buttons_.at(0)));
    DrawText("Start Game", std::get<0>(buttons_.at(0)).x + 10, std::get<0>(buttons_.at(0)).y + 5, 16, BLACK);
}

void MigoyugoUI::handle_board_events()
{
    if (!paused_)
    {
        int current_player_ind = state_ptr_->player_turn();
        auto& current_player_info = players_.at(current_player_ind);
        auto player_p = dynamic_cast<const rl::players::HumanPlayer*>(current_player_info->player_ptr_.get());
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
            int action = current_player_info->player_ptr_->choose_action(state_ptr_->clone_state());
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
            current_window_ = MigoyugoWindow::menu;
            paused_ = false;
        }
    }
}

void MigoyugoUI::handle_menu_events()
{
    if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON))
    {
        auto mouse_pos = GetMousePosition();
        auto [rec, col] = buttons_.at(0);
        auto is_button_pressed = CheckCollisionPointRec(mouse_pos, rec);
        if (is_button_pressed)
        {
            reset_state();
            current_window_ = MigoyugoWindow::game;
            players_.clear();
            auto players_duration = std::chrono::milliseconds(5000);

            // players_.push_back(get_default_g_player(state_ptr_.get(), 2, players_duration));
            // players_.push_back(get_human_player(state_ptr_.get()));
            
            players_.push_back(get_network_amcts2_player(state_ptr_.get(),3,players_duration,"migoyugo_strongest_400.pt"));
            players_.push_back(get_network_amcts2_player(state_ptr_.get(),3,players_duration,"migoyugo_strongest_480.pt"));
        }
    }
}

void MigoyugoUI::perform_action(int action)
{
    auto actions_legality = state_ptr_->actions_mask();
    if (action < actions_legality.size() && actions_legality.at(action) && state_ptr_->is_terminal() == false)
    {
        set_state(state_ptr_->step_state(action));
    }
}

void MigoyugoUI::perform_player_action(int row, int col)
{
    int action = rl::games::MigoyugoState::encode_action(row, col);
    if (action < state_ptr_->get_n_actions() && actions_legality_.at(action))
    {
        perform_action(action);
    }
}

void MigoyugoUI::draw_piece(int left, int top, int player, bool is_yugo)
{
    int left_center = left + inner_cell_size_ / 2;
    int top_center = top + inner_cell_size_ / 2;
    int radius = inner_cell_size_ / 4;

    Color piece_color;
    if (player == 0)
    {
        piece_color = is_yugo ? WHITE : WHITE;
    }
    else
    {
        piece_color = is_yugo ? BLACK : BLACK;
    }

    DrawCircle(left_center, top_center, radius, piece_color);

    // Add small red circle inside yugo pieces
    if (is_yugo)
    {
        DrawCircle(left_center, top_center, radius / 3, RED);
        DrawCircleLines(left_center, top_center, radius, BLACK);
    }
}

void MigoyugoUI::draw_legal_actions()
{
    int top, left;
    for (int action = 0; action < actions_legality_.size(); action++)
    {
        if (actions_legality_.at(action))
        {
            int row = action / 8;
            int col = action % 8;
            left = col * cell_size_ + padding_;
            top = row * cell_size_ + padding_;
            Rectangle ol = { left, top, inner_cell_size_, inner_cell_size_ };
            DrawRectangleLinesEx(ol, 2.0f, GREEN);
        }
    }
}

} // namespace rl::ui
