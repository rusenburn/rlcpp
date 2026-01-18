#include "migoyugo_ui.hpp"
#include <iostream>

namespace rl::ui
{
const std::vector<std::string> PLAYER_TYPES = {"default_g_player", "human", "network"};

MigoyugoUI::MigoyugoUI(int width, int height)
    : width_{ width }, height_{ height }, padding_{ 2 }, state_ptr_{ rl::games::MigoyugoState::initialize_state() },
    current_window_{ MigoyugoWindow::menu },
    players_{},
    paused_{ false },
    pause_until_{ 0.0 },
    selected_player_type_{ "default_g_player" },
    duration_input_{ "5000" },
    loadname_input_{ "" },
    duration_input_focused_{ false },
    loadname_input_focused_{ false },
    player_type_index_{ 0 }

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
    float button_width = 120;
    float button_height = 25;

    float top = 20;
    float left = 20;

    // Player type selector button
    buttons_.push_back(std::make_pair<Rectangle, Color>(Rectangle{ left, top, button_width, button_height }, GRAY));

    // Add player button
    top += button_height + 10;
    buttons_.push_back(std::make_pair<Rectangle, Color>(Rectangle{ left, top, button_width, button_height }, GREEN));

    // Clear players button
    top += button_height + 10;
    buttons_.push_back(std::make_pair<Rectangle, Color>(Rectangle{ left, top, button_width, button_height }, RED));

    // Start game button (position will be calculated dynamically)
    buttons_.push_back(std::make_pair<Rectangle, Color>(Rectangle{ left + button_width + 20, top, button_width, button_height }, BLUE));
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
    float left = 20;
    float top = 20;
    float button_width = 120;
    float button_height = 25;
    float input_width = 120;
    float input_height = 25;

    // Player Type label
    DrawText("Player Type:", left, top - 5, 16, BLACK);
    top += 20;

    // Player type selector button
    auto& player_type_button = buttons_[0];
    DrawRectangleRec(std::get<0>(player_type_button), std::get<1>(player_type_button));
    DrawText(selected_player_type_.c_str(), std::get<0>(player_type_button).x + 10, std::get<0>(player_type_button).y + 5, 14, BLACK);
    top += button_height + 10;

    // Duration label and input
    DrawText("Duration (ms):", left, top - 5, 16, BLACK);
    top += 20;
    Rectangle duration_rect = {left, top, input_width, input_height};
    DrawRectangleRec(duration_rect, LIGHTGRAY);
    if (duration_input_focused_) DrawRectangleLinesEx(duration_rect, 2, BLUE);
    DrawText(duration_input_.c_str(), left + 5, top + 5, 14, BLACK);
    top += input_height + 10;

    // Load name label and input (only for network)
    if (selected_player_type_ == "network") {
        DrawText("Load Name:", left, top - 5, 16, BLACK);
        top += 20;
        Rectangle loadname_rect = {left, top, input_width, input_height};
        DrawRectangleRec(loadname_rect, LIGHTGRAY);
        if (loadname_input_focused_) DrawRectangleLinesEx(loadname_rect, 2, BLUE);
        DrawText(loadname_input_.c_str(), left + 5, top + 5, 14, BLACK);
        top += input_height + 10;
    }

    // Update button positions dynamically
    buttons_[1].first.y = top; // Add Player button
    top += button_height + 10;
    buttons_[2].first.y = top; // Clear Players button
    top += button_height + 10;
    buttons_[3].first.y = top; // Start Game button

    // Add Player button
    auto& add_button = buttons_[1];
    DrawRectangleRec(std::get<0>(add_button), std::get<1>(add_button));
    DrawText("Add Player", std::get<0>(add_button).x + 10, std::get<0>(add_button).y + 5, 14, BLACK);

    // Clear Players button
    auto& clear_button = buttons_[2];
    DrawRectangleRec(std::get<0>(clear_button), std::get<1>(clear_button));
    DrawText("Clear Players", std::get<0>(clear_button).x + 10, std::get<0>(clear_button).y + 5, 14, BLACK);

    // Start Game button (only if players >= 2)
    if (players_.size() >= 2) {
        auto& start_button = buttons_[3];
        DrawRectangleRec(std::get<0>(start_button), std::get<1>(start_button));
        DrawText("Start Game", std::get<0>(start_button).x + 10, std::get<0>(start_button).y + 5, 14, BLACK);
    }

    // Draw players list
    top += button_height + 20;
    DrawText("Players:", left, top - 5, 16, BLACK);
    top += 20;
    for (size_t i = 0; i < players_.size(); ++i) {
        std::string player_text = "Player " + std::to_string(i + 1) + ": " + players_[i]->name_;
        DrawText(player_text.c_str(), left, top, 14, BLACK);
        top += 20;
    }
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
    Vector2 mouse_pos = GetMousePosition();
    bool mouse_clicked = IsMouseButtonPressed(MOUSE_LEFT_BUTTON);

    // Handle text input focus
    float left = 20;
    float top = 20 + 20 + 25 + 10 + 20; // Position of duration input
    Rectangle duration_rect = {left, top, 120, 25};
    if (mouse_clicked && CheckCollisionPointRec(mouse_pos, duration_rect)) {
        duration_input_focused_ = true;
        loadname_input_focused_ = false;
    } else if (selected_player_type_ == "network") {
        float loadname_top = top + 25 + 10 + 20;
        Rectangle loadname_rect = {left, loadname_top, 120, 25};
        if (mouse_clicked && CheckCollisionPointRec(mouse_pos, loadname_rect)) {
            duration_input_focused_ = false;
            loadname_input_focused_ = true;
        } else if (mouse_clicked) {
            duration_input_focused_ = false;
            loadname_input_focused_ = false;
        }
    } else if (mouse_clicked) {
        duration_input_focused_ = false;
        loadname_input_focused_ = false;
    }

    // Handle text input
    if (duration_input_focused_ || loadname_input_focused_) {
        int key = GetCharPressed();
        while (key > 0) {
            if ((key >= 32) && (key <= 125)) {
                if (duration_input_focused_) {
                    duration_input_ += (char)key;
                } else if (loadname_input_focused_) {
                    loadname_input_ += (char)key;
                }
            }
            key = GetCharPressed();
        }
        if (IsKeyPressed(KEY_BACKSPACE)) {
            if (duration_input_focused_ && !duration_input_.empty()) {
                duration_input_.pop_back();
            } else if (loadname_input_focused_ && !loadname_input_.empty()) {
                loadname_input_.pop_back();
            }
        }
    }

    // Handle button clicks
    if (mouse_clicked) {
        // Player type selector
        if (CheckCollisionPointRec(mouse_pos, std::get<0>(buttons_[0]))) {
            player_type_index_ = (player_type_index_ + 1) % PLAYER_TYPES.size();
            selected_player_type_ = PLAYER_TYPES[player_type_index_];
        }
        // Add player button
        else if (CheckCollisionPointRec(mouse_pos, std::get<0>(buttons_[1]))) {
            try {
                int duration_ms = std::stoi(duration_input_);
                auto duration = std::chrono::duration_cast<std::chrono::duration<int, std::milli>>(std::chrono::milliseconds(duration_ms));

                if (selected_player_type_ == "default_g_player") {
                    players_.push_back(get_default_g_player(state_ptr_.get(), 2, duration));
                } else if (selected_player_type_ == "human") {
                    players_.push_back(get_human_player(state_ptr_.get()));
                } else if (selected_player_type_ == "network") {
                    if (loadname_input_.empty()) {
                        // Default load name if empty
                        loadname_input_ = "migoyugo_strongest_480.pt";
                    }
                    players_.push_back(get_network_amcts2_player(state_ptr_.get(), 2, duration, loadname_input_));
                }
            } catch (const std::invalid_argument&) {
                // Invalid duration, ignore
            }
        }
        // Clear players button
        else if (CheckCollisionPointRec(mouse_pos, std::get<0>(buttons_[2]))) {
            players_.clear();
        }
        // Start game button (only if >=2 players)
        else if (players_.size() >= 2 && CheckCollisionPointRec(mouse_pos, std::get<0>(buttons_[3]))) {
            reset_state();
            current_window_ = MigoyugoWindow::game;
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
