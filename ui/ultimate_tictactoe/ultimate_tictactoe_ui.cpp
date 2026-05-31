#include "ultimate_tictactoe_ui.hpp"
#include <iostream>
#include <sstream>

namespace rl::ui
{
const std::vector<std::string> PLAYER_TYPES = { "default_g_player", "human", "network" };

UltimateTicTacToeUI::UltimateTicTacToeUI(int width, int height)
    : width_{ width }, height_{ height }, padding_{ 2 }, state_ptr_{ rl::games::UltimateTicTacToeState::initialize_state() },
    current_window_{ UltimateTicTacToeWindow::menu },
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
    // Reserve space for row numbers on the left
    int left_margin = 30;
    int board_width = width_ - left_margin;
    constexpr int GRID_SIZE = 9; // 9x9 overall grid
    cell_size_ = board_width / GRID_SIZE;
    inner_cell_size_ = cell_size_ - 2 * padding_;
    initialize_buttons();
    reset_state();
}

UltimateTicTacToeUI::~UltimateTicTacToeUI() = default;

void UltimateTicTacToeUI::draw_game()
{
    if (current_window_ == UltimateTicTacToeWindow::game)
    {
        draw_board();
    }
    else if (current_window_ == UltimateTicTacToeWindow::menu)
    {
        draw_menu();
    }
}

void UltimateTicTacToeUI::handle_events()
{
    if (current_window_ == UltimateTicTacToeWindow::game)
    {
        handle_board_events();
    }
    else if (current_window_ == UltimateTicTacToeWindow::menu)
    {
        handle_menu_events();
    }
}

void UltimateTicTacToeUI::set_state(UltimateTicTacToeStatePtr new_state_ptr)
{
    state_ptr_ = std::move(new_state_ptr);
    obs_ = state_ptr_->get_observation();
    actions_legality_ = state_ptr_->actions_mask();
}

void UltimateTicTacToeUI::reset_state()
{
    set_state(state_ptr_->reset_state());
}

void UltimateTicTacToeUI::initialize_buttons()
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

void UltimateTicTacToeUI::draw_board()
{
    constexpr int ROWS = 3;
    constexpr int COLS = 3;
    constexpr int BOARDS = 9;

    int current_player = state_ptr_->player_turn();
    int last_action = state_ptr_->get_last_action();

    // Draw the 9x9 grid
    for (int board_no = 0; board_no < BOARDS; board_no++)
    {
        int meta_row = board_no / COLS;
        int meta_col = board_no % COLS;

        // Check if this board is won/drawn
        int ultimate_owner = 0;
        int ultimate_owner_flag = 0;
        for (int r = 0; r < ROWS && ultimate_owner_flag == 0; r++) {
            for (int c = 0; c < COLS && ultimate_owner_flag == 0; c++) {
                // Use observation channels 9 (current player) and 19 (opponent)
                // obs_[9 * 9 + r * 3 + c] == 1 if current player owns board at (r,c)
                // obs_[19 * 9 + r * 3 + c] == 1 if opponent owns board at (r,c)
                if (r == meta_row && c == meta_col) {
                    if (obs_.at(9 * 9 + r * 3 + c) == 1.0f) {
                        ultimate_owner = 1; // current player's flag
                        ultimate_owner_flag = 1;
                    }
                    else if (obs_.at(19 * 9 + r * 3 + c) == 1.0f) {
                        ultimate_owner = -1; // opponent's flag
                        ultimate_owner_flag = 1;
                    }
                }
            }
        }

        // If we can't find it in obs, check the state directly
        if (ultimate_owner_flag == 0) {
            // Check from the state's ultimate_board_ via obs channel logic
            // Actually let's derive from player turn:
            // Channel 9 = current player's ultimate, channel 19 = opponent's ultimate
            // So we already checked above. If not found, board is not won by anyone.
            ultimate_owner = 0;
        }

        bool is_terminal_board = false;
        if (ultimate_owner != 0) {
            is_terminal_board = true;
        }
        else {
            // Check if board is drawn (full but no winner)
            bool board_full = true;
            for (int r = 0; r < ROWS; r++) {
                for (int c = 0; c < COLS; c++) {
                    int obs_idx_current = board_no * 9 + r * 3 + c;
                    int obs_idx_opponent = (10 + board_no) * 9 + r * 3 + c;
                    if (obs_.at(obs_idx_current) == 0.0f && obs_.at(obs_idx_opponent) == 0.0f) {
                        board_full = false;
                    }
                }
            }
            if (board_full) {
                is_terminal_board = true;
            }
        }

        // Draw the cells in this board
        for (int row = 0; row < ROWS; row++)
        {
            for (int col = 0; col < COLS; col++)
            {
                int overall_row = meta_row * ROWS + row;
                int overall_col = meta_col * COLS + col;

                int left = overall_col * cell_size_ + padding_;
                int top = overall_row * cell_size_ + padding_;

                // Determine cell background based on board status
                Color cell_color;
                if (ultimate_owner != 0)
                {
                    cell_color = DARKGRAY; // Won board - dark background
                }
                else if (is_terminal_board)
                {
                    cell_color = GRAY; // Drawn board
                }
                else
                {
                    cell_color = DARKGREEN; // Active board
                }
                DrawRectangle(left, top, inner_cell_size_, inner_cell_size_, cell_color);

                // Check for pieces in this cell using observations
                // Current player pieces: channels 0-8
                int obs_idx_current = board_no * 9 + row * 3 + col;
                // Opponent pieces: channels 10-18
                int obs_idx_opponent = (10 + board_no) * 9 + row * 3 + col;

                // Map which player is X and which is O
                // If current_player == 0: current player=player 0 (flag 1) = X, opponent=player 1 (flag -1) = O
                // If current_player == 1: current player=player 1 (flag -1) = O, opponent=player 0 (flag 1) = X
                int piece_player = -1; // -1 means no piece
                if (obs_.at(obs_idx_current) == 1.0f)
                {
                    piece_player = 0; // Current player's piece
                }
                else if (obs_.at(obs_idx_opponent) == 1.0f)
                {
                    piece_player = 1; // Opponent's piece
                }

                if (piece_player >= 0)
                {
                    draw_piece(left, top, piece_player);
                }

                // Highlight last action with red square
                int current_action = board_no * 9 + row * 3 + col;
                if (current_action == last_action && last_action >= 0)
                {
                    Rectangle last_action_rect = { left, top, inner_cell_size_, inner_cell_size_ };
                    DrawRectangleLinesEx(last_action_rect, 3.0f, RED);
                }
            }
        }

        // If board is won, draw a big X or O over the whole board
        if (ultimate_owner != 0)
        {
            int board_left = meta_col * COLS * cell_size_ + padding_;
            int board_top = meta_row * ROWS * cell_size_ + padding_;
            int board_pixel_size = COLS * cell_size_ - 2 * padding_;

            // Determine the player who won this board from perspective of the game
            // ultimate_owner is relative to current_player's perspective
            // Channel 9 = current player, so if ultimate_owner == 1, current player won
            // We want to draw X for player 0, O for player 1
            int absolute_winner;
            if (current_player == 0) {
                absolute_winner = ultimate_owner == 1 ? 0 : 1;
            }
            else {
                absolute_winner = ultimate_owner == 1 ? 1 : 0;
            }
            draw_ultimate_board_owner(board_left, board_top, absolute_winner);
        }
        else if (is_terminal_board)
        {
            // Draw a small "D" for drawn board
            int board_left = meta_col * COLS * cell_size_ + padding_;
            int board_top = meta_row * ROWS * cell_size_ + padding_;
            int board_pixel_size = COLS * cell_size_ - 2 * padding_;
            int center_x = board_left + board_pixel_size / 2 - 10;
            int center_y = board_top + board_pixel_size / 2 - 10;
            DrawText("D", center_x, center_y, 30, DARKGRAY);
        }
    }

    // Draw thin cell borders
    for (int row = 0; row < 9; row++) {
        for (int col = 0; col < 9; col++) {
            int left = col * cell_size_ + padding_;
            int top = row * cell_size_ + padding_;
            Rectangle cell_rect = { left, top, inner_cell_size_, inner_cell_size_ };
            DrawRectangleLinesEx(cell_rect, 1.0f, Color{ 100, 100, 100, 100 }); // subtle gray
        }
    }

    // Draw thick lines between meta-boards
    float meta_line_thickness = 8.0f;
    for (int i = 1; i < 3; i++) {
        float x = static_cast<float>(i * 3 * cell_size_);
        DrawLineEx({ x, 0.0f }, { x, 9.0f * cell_size_ }, meta_line_thickness, BLACK);
        float y = static_cast<float>(i * 3 * cell_size_);
        DrawLineEx({ 0.0f, y }, { 9.0f * cell_size_, y }, meta_line_thickness, BLACK);
    }

    draw_legal_actions();
}

void UltimateTicTacToeUI::draw_menu()
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
    Rectangle duration_rect = { left, top, input_width, input_height };
    DrawRectangleRec(duration_rect, LIGHTGRAY);
    if (duration_input_focused_) DrawRectangleLinesEx(duration_rect, 2, BLUE);
    DrawText(duration_input_.c_str(), left + 5, top + 5, 14, BLACK);
    top += input_height + 10;

    // Load name label and input (only for network)
    if (selected_player_type_ == "network") {
        DrawText("Load Name:", left, top - 5, 16, BLACK);
        top += 20;
        Rectangle loadname_rect = { left, top, input_width, input_height };
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

void UltimateTicTacToeUI::handle_board_events()
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
                // Convert mouse position to overall 9x9 grid coordinates
                int overall_col = mouse_position.x / cell_size_;
                int overall_row = mouse_position.y / cell_size_;
                // Clip to valid range
                if (overall_row >= 0 && overall_row < 9 && overall_col >= 0 && overall_col < 9)
                {
                    int board_no = (overall_row / 3) * 3 + (overall_col / 3);
                    int row = overall_row % 3;
                    int col = overall_col % 3;
                    int action = board_no * 9 + row * 3 + col;
                    perform_player_action(action);
                }
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

            std::cout << "Actions History: ";
            for (int action : history_) {
                std::cout << action << ' ';
            }
            std::cout << "-1\n";
        }
    }
    else
    {
        double current_time = GetTime();
        if (current_time > pause_until_)
        {
            current_window_ = UltimateTicTacToeWindow::menu;
            paused_ = false;
        }
    }
}

void UltimateTicTacToeUI::handle_menu_events()
{
    Vector2 mouse_pos = GetMousePosition();
    bool mouse_clicked = IsMouseButtonPressed(MOUSE_LEFT_BUTTON);

    // Handle text input focus
    float left = 20;
    float top = 20 + 20 + 25 + 10 + 20; // Position of duration input
    Rectangle duration_rect = { left, top, 120, 25 };
    if (mouse_clicked && CheckCollisionPointRec(mouse_pos, duration_rect)) {
        duration_input_focused_ = true;
        loadname_input_focused_ = false;
    }
    else if (selected_player_type_ == "network") {
        float loadname_top = top + 25 + 10 + 20;
        Rectangle loadname_rect = { left, loadname_top, 120, 25 };
        if (mouse_clicked && CheckCollisionPointRec(mouse_pos, loadname_rect)) {
            duration_input_focused_ = false;
            loadname_input_focused_ = true;
        }
        else if (mouse_clicked) {
            duration_input_focused_ = false;
            loadname_input_focused_ = false;
        }
    }
    else if (mouse_clicked) {
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
                }
                else if (loadname_input_focused_) {
                    loadname_input_ += (char)key;
                }
            }
            key = GetCharPressed();
        }
        if (IsKeyPressed(KEY_BACKSPACE)) {
            if (duration_input_focused_ && !duration_input_.empty()) {
                duration_input_.pop_back();
            }
            else if (loadname_input_focused_ && !loadname_input_.empty()) {
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
                }
                else if (selected_player_type_ == "human") {
                    players_.push_back(get_human_player(state_ptr_.get()));
                }
                else if (selected_player_type_ == "network") {
                    if (loadname_input_.empty()) {
                        loadname_input_ = "uttt.pt";
                    }
                    players_.push_back(get_network_amcts2_player(state_ptr_.get(), 2, duration, loadname_input_));
                }
            }
            catch (const std::invalid_argument&) {
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
            history_.clear();
            current_window_ = UltimateTicTacToeWindow::game;
        }
    }
}

void UltimateTicTacToeUI::perform_action(int action)
{
    auto actions_legality = state_ptr_->actions_mask();
    if (action < static_cast<int>(actions_legality.size()) && actions_legality.at(action) && state_ptr_->is_terminal() == false)
    {
        history_.push_back(action);
        set_state(state_ptr_->step_state(action));
    }
}

void UltimateTicTacToeUI::perform_player_action(int action)
{
    if (action < state_ptr_->get_n_actions() && actions_legality_.at(action))
    {
        perform_action(action);
    }
}

void UltimateTicTacToeUI::draw_piece(int left, int top, int player)
{
    int current_player = state_ptr_->player_turn();

    // player=0 means it's a "current player" piece in observation
    // player=1 means it's an "opponent" piece in observation
    // We draw:
    // If current_player == 0 (flag=1=X): current pieces = X, opponent pieces = O
    // If current_player == 1 (flag=-1=O): current pieces = O, opponent pieces = X
    bool is_x;
    if (current_player == 0) {
        is_x = (player == 0); // player 0 = X
    }
    else {
        is_x = (player == 1); // player 1's pieces → when current is 1, opponent = player 0 = X
    }

    int center_x = left + inner_cell_size_ / 2;
    int center_y = top + inner_cell_size_ / 2;
    int radius = inner_cell_size_ / 4;

    if (is_x)
    {
        // Draw X with thick lines
        int offset = radius / 2;
        float thickness = 4.0f;
        Color x_color = BLUE;
        DrawLineEx({static_cast<float>(center_x - offset), static_cast<float>(center_y - offset)},
                   {static_cast<float>(center_x + offset), static_cast<float>(center_y + offset)},
                   thickness, x_color);
        DrawLineEx({static_cast<float>(center_x - offset), static_cast<float>(center_y + offset)},
                   {static_cast<float>(center_x + offset), static_cast<float>(center_y - offset)},
                   thickness, x_color);
    }
    else
    {
        // Draw O
        Color o_color = WHITE;
        DrawCircle(center_x, center_y, radius, o_color);
        DrawCircle(center_x, center_y, radius - 3, DARKGREEN); // Erase center for ring effect
        DrawCircleLines(center_x, center_y, radius, o_color);
    }
}

void UltimateTicTacToeUI::draw_legal_actions()
{
    for (int action = 0; action < static_cast<int>(actions_legality_.size()); action++)
    {
        if (actions_legality_.at(action))
        {
            int board_no = action / 9;
            int row = (action % 9) / 3;
            int col = (action % 9) % 3;
            int overall_row = (board_no / 3) * 3 + row;
            int overall_col = (board_no % 3) * 3 + col;
            int left = overall_col * cell_size_ + padding_;
            int top = overall_row * cell_size_ + padding_;
            Rectangle ol = { left, top, inner_cell_size_, inner_cell_size_ };
            DrawRectangleLinesEx(ol, 2.0f, GREEN);
        }
    }
}

void UltimateTicTacToeUI::draw_ultimate_board_owner(int left, int top, int owner)
{
    int board_pixel_size = 3 * cell_size_ - 2 * padding_;
    int center_x = left + board_pixel_size / 2;
    int center_y = top + board_pixel_size / 2;

    if (owner == 0)
    {
        // Player 0 won - draw big X
        int offset = board_pixel_size / 3;
        DrawLine(center_x - offset, center_y - offset, center_x + offset, center_y + offset, BLUE);
        DrawLine(center_x - offset, center_y + offset, center_x + offset, center_y - offset, BLUE);
    }
    else
    {
        // Player 1 won - draw big O
        int radius = board_pixel_size / 3;
        DrawCircleLines(center_x, center_y, radius, WHITE);
    }
}

} // namespace rl::ui