#include "migoyugo_ui.hpp"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace rl::ui
{
const std::vector<std::string> PLAYER_TYPES = { "default_g_player", "human", "network","nnue" ,"nnue_mcts", "nnue_layerstacks", "nnue_layerstacks_v2", "migoyugo_grave" };

// Player types that take a weights file name. migoyugo_grave is here because
// the field is useful to it, but it is the one type for which leaving the
// field empty is a valid choice rather than an omission.
static bool uses_load_name(const std::string& player_type)
{
    return player_type == "network"
        || player_type == "nnue"
        || player_type == "nnue_layerstacks"
        || player_type == "nnue_layerstacks_v2"
        || player_type == "migoyugo_grave";
}

MigoyugoUI::MigoyugoUI(int width, int height)
    : width_{ width }, height_{ height }, padding_{ 2 },
    cell_size_{ 0 }, inner_cell_size_{ 0 },
    left_margin_{ 24 }, header_height_{ 60 },
    state_ptr_{ rl::games::MigoyugoState::initialize_state() },
    current_window_{ MigoyugoWindow::menu },
    players_{},
    obs_{}, actions_legality_{}, buttons_{}, history_{},
    game_over_{ false },
    winner_{ -2 },
    exit_button_rect_{}, rematch_button_rect_{},
    wins_{}, draws_{ 0 },
    move_sound_{}, win_sound_{}, draw_sound_{},
    selected_player_type_{ "default_g_player" },
    duration_input_{ "5000" },
    loadname_input_{ "" },
    duration_input_focused_{ false },
    loadname_input_focused_{ false },
    player_type_index_{ 0 }

{
    // Reserve space for coordinate labels: left margin for row numbers, header
    // for names/score/turn indicator, bottom margin for column letters.
    int bottom_margin = 24;
    int available_w = width_ - left_margin_;
    int available_h = height_ - header_height_ - bottom_margin;
    cell_size_ = std::min(available_w, available_h) / 8;
    inner_cell_size_ = cell_size_ - 2 * padding_;
    initialize_buttons();
    reset_state();

    move_sound_ = load_tone_sound(synthesize_tone(760.0f, 0.09f, ToneWave::Sine, 0.5f));
    win_sound_ = load_tone_sound(synthesize_sequence({ {523.25f, 0.12f}, {659.25f, 0.12f}, {784.00f, 0.18f} }, ToneWave::Sine, 0.5f));
    draw_sound_ = load_tone_sound(synthesize_sequence({ {392.00f, 0.12f}, {349.23f, 0.22f} }, ToneWave::Square, 0.4f));
}

MigoyugoUI::~MigoyugoUI()
{
    UnloadSound(move_sound_);
    UnloadSound(win_sound_);
    UnloadSound(draw_sound_);
}

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

    // Game-over overlay buttons, centered over the board
    Rectangle panel = { (width_ - 380) / 2.0f, header_height_ + (height_ - header_height_ - 24 - 170) / 2.0f, 380, 170 };
    float overlay_button_width = 160;
    float overlay_button_height = 34;
    float overlay_button_y = panel.y + panel.height - overlay_button_height - 16;
    exit_button_rect_ = { panel.x + 20, overlay_button_y, overlay_button_width, overlay_button_height };
    rematch_button_rect_ = { panel.x + panel.width - overlay_button_width - 20, overlay_button_y, overlay_button_width, overlay_button_height };
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

    draw_header();

    for (int row = 0; row < ROWS; row++)
    {
        for (int col = 0; col < COLS; col++)
        {
            left = left_margin_ + col * cell_size_ + padding_;
            top = header_height_ + row * cell_size_ + padding_;

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
        left = left_margin_ + col * cell_size_ + padding_ + inner_cell_size_ / 2 - 5;
        top = header_height_ + ROWS * cell_size_ + padding_ + 5;
        char letter = 'a' + col;
        DrawText(&letter, left, top, 16, BLACK);
    }

    // Row numbers 1-8 (1 at bottom, 8 at top)
    for (int row = 0; row < ROWS; row++) {
        left = 5;  // Fixed position within the left margin
        top = header_height_ + row * cell_size_ + padding_ + inner_cell_size_ / 2 - 8;  // Center vertically in the cell
        char number = '8' - row;  // 8 at top (row 0), 1 at bottom (row 7)
        DrawText(&number, left, top, 16, BLACK);
    }

    draw_legal_actions();

    if (game_over_)
    {
        draw_game_over_overlay();
    }
}

void MigoyugoUI::count_yugos(int& p0_yugos, int& p1_yugos) const
{
    constexpr int ROWS = 8;
    constexpr int COLS = 8;
    constexpr int OUR_YUGO_CHANNEL = 1;
    constexpr int OPP_YUGO_CHANNEL = 3;
    constexpr int channel_size = ROWS * COLS;

    p0_yugos = 0;
    p1_yugos = 0;

    // obs_ is always from the current player's perspective; map "our"/"opp"
    // channels back to the fixed player-0/player-1 identities used everywhere
    // else in the UI (see draw_board()'s identical mapping).
    int current_player = state_ptr_->player_turn();
    int actual_player_for_our = current_player == 0 ? 0 : 1;
    int actual_player_for_opp = 1 - actual_player_for_our;

    for (int i = 0; i < channel_size; i++)
    {
        if (obs_.at(OUR_YUGO_CHANNEL * channel_size + i) == 1.0f)
        {
            (actual_player_for_our == 0 ? p0_yugos : p1_yugos)++;
        }
        if (obs_.at(OPP_YUGO_CHANNEL * channel_size + i) == 1.0f)
        {
            (actual_player_for_opp == 0 ? p0_yugos : p1_yugos)++;
        }
    }
}

void MigoyugoUI::draw_header()
{
    int current_player = state_ptr_->player_turn();

    if (!game_over_)
    {
        DrawRectangle(0, current_player == 0 ? 0 : 20, width_, 20, Fade(YELLOW, 0.35f));
    }

    int p0_yugos = 0, p1_yugos = 0;
    count_yugos(p0_yugos, p1_yugos);

    std::string p1_name = players_.size() > 0 ? players_[0]->name_ : "?";
    std::string p2_name = players_.size() > 1 ? players_[1]->name_ : "?";
    int p1_wins = wins_.size() > 0 ? wins_[0] : 0;
    int p2_wins = wins_.size() > 1 ? wins_[1] : 0;

    std::string p1_text = "P1 White: " + p1_name + "  (Wins: " + std::to_string(p1_wins) + ", Yugos: " + std::to_string(p0_yugos) + ")";
    std::string p2_text = "P2 Black: " + p2_name + "  (Wins: " + std::to_string(p2_wins) + ", Yugos: " + std::to_string(p1_yugos) + ")";
    std::string draws_text = "Draws: " + std::to_string(draws_);

    DrawText(p1_text.c_str(), 4, 4, 14, BLACK);
    DrawText(p2_text.c_str(), 4, 24, 14, BLACK);
    int draws_width = MeasureText(draws_text.c_str(), 12);
    DrawText(draws_text.c_str(), (width_ - draws_width) / 2, 44, 12, DARKGRAY);
}

void MigoyugoUI::draw_game_over_overlay()
{
    Rectangle panel = { (width_ - 380) / 2.0f, header_height_ + (height_ - header_height_ - 24 - 170) / 2.0f, 380, 170 };
    DrawRectangleRec(panel, Fade(BLACK, 0.85f));
    DrawRectangleLinesEx(panel, 2, WHITE);

    std::string title;
    std::string subtitle;
    if (winner_ == -1)
    {
        title = "Draw!";
    }
    else
    {
        title = (winner_ == 0 ? "White Wins!" : "Black Wins!");
        std::string name = winner_ < static_cast<int>(players_.size()) ? players_[winner_]->name_ : "?";
        subtitle = "Player " + std::to_string(winner_ + 1) + ": " + name;
    }

    int title_width = MeasureText(title.c_str(), 28);
    DrawText(title.c_str(), panel.x + (panel.width - title_width) / 2, panel.y + 16, 28, WHITE);

    if (!subtitle.empty())
    {
        int subtitle_width = MeasureText(subtitle.c_str(), 16);
        DrawText(subtitle.c_str(), panel.x + (panel.width - subtitle_width) / 2, panel.y + 52, 16, LIGHTGRAY);
    }

    DrawRectangleRec(exit_button_rect_, GRAY);
    DrawText("Exit to Menu", exit_button_rect_.x + 10, exit_button_rect_.y + 9, 14, BLACK);

    DrawRectangleRec(rematch_button_rect_, GREEN);
    DrawText("Rematch", rematch_button_rect_.x + 10, rematch_button_rect_.y + 4, 14, BLACK);
    DrawText("(Switch Sides)", rematch_button_rect_.x + 10, rematch_button_rect_.y + 18, 10, BLACK);
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
    Rectangle duration_rect = { left, top, input_width, input_height };
    DrawRectangleRec(duration_rect, LIGHTGRAY);
    if (duration_input_focused_) DrawRectangleLinesEx(duration_rect, 2, BLUE);
    DrawText(duration_input_.c_str(), left + 5, top + 5, 14, BLACK);
    top += input_height + 10;

    // Load name label and input (only for network)
    if (uses_load_name(selected_player_type_)) {
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
        if (i < wins_.size()) {
            player_text += "  (Wins: " + std::to_string(wins_[i]) + ")";
        }
        DrawText(player_text.c_str(), left, top, 14, BLACK);
        top += 20;
    }
    if (!players_.empty()) {
        std::string draws_text = "Draws: " + std::to_string(draws_);
        DrawText(draws_text.c_str(), left, top, 14, BLACK);
    }
}

void MigoyugoUI::handle_board_events()
{
    if (!game_over_)
    {
        int current_player_ind = state_ptr_->player_turn();
        auto& current_player_info = players_.at(current_player_ind);
        auto player_p = dynamic_cast<const rl::players::HumanPlayer*>(current_player_info->player_ptr_.get());
        if (player_p != nullptr)
        {
            if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT))
            {
                Vector2 mouse_position = GetMousePosition();
                int rel_x = static_cast<int>(mouse_position.x) - left_margin_;
                int rel_y = static_cast<int>(mouse_position.y) - header_height_;
                if (rel_x >= 0 && rel_y >= 0)
                {
                    int row = rel_y / cell_size_;
                    int col = rel_x / cell_size_;
                    if (row <= 7 && col <= 7)
                    {
                        perform_player_action(row, col);
                    }
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
            game_over_ = true;
            winner_ = compute_winner();
            if (winner_ == -1)
            {
                draws_++;
                PlaySound(draw_sound_);
            }
            else
            {
                wins_.at(winner_)++;
                PlaySound(win_sound_);
            }

            std::cout << "Actions History: ";
            for (int action : history_) {
                std::cout << action << ' ';
            }
            std::cout << "-1\n";
        }
    }
    else
    {
        if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT))
        {
            Vector2 mouse_pos = GetMousePosition();
            if (CheckCollisionPointRec(mouse_pos, exit_button_rect_))
            {
                current_window_ = MigoyugoWindow::menu;
                game_over_ = false;
                winner_ = -2;
            }
            else if (CheckCollisionPointRec(mouse_pos, rematch_button_rect_))
            {
                std::reverse(players_.begin(), players_.end());
                std::reverse(wins_.begin(), wins_.end());
                reset_state();
                history_.clear();
                game_over_ = false;
                winner_ = -2;
            }
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
    Rectangle duration_rect = { left, top, 120, 25 };
    if (mouse_clicked && CheckCollisionPointRec(mouse_pos, duration_rect)) {
        duration_input_focused_ = true;
        loadname_input_focused_ = false;
    }
    else if (uses_load_name(selected_player_type_)) {
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
            size_t prev_player_count = players_.size();
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
                        // Default load name if empty
                        loadname_input_ = "migoyugo_strongest_480.pt";
                    }
                    players_.push_back(get_network_amcts2_player(state_ptr_.get(), 2, duration, loadname_input_));
                }
                else if (selected_player_type_ == "nnue") {
                    if (loadname_input_.empty()) {
                        // Default load name if empty
                        loadname_input_ = "nnue_weights.bin";
                    }
                    players_.push_back(get_nnue_player(state_ptr_.get(), duration, loadname_input_));
                }

                else if (selected_player_type_ == "nnue_mcts") {
                    players_.push_back(get_nnue_mcts_player(state_ptr_.get(), duration));
                }

                else if (selected_player_type_ == "nnue_layerstacks") {
                    if (loadname_input_.empty()) {
                        // Default load name if empty
                        loadname_input_ = "nnue_layerstacks_weights.bin";
                    }
                    players_.push_back(get_nnue_layerstacks_player(state_ptr_.get(), duration, loadname_input_));
                }

                else if (selected_player_type_ == "nnue_layerstacks_v2") {
                    if (loadname_input_.empty()) {
                        // Default load name if empty
                        loadname_input_ = "nnue_layerstacks_v2_weights.bin";
                    }
                    players_.push_back(get_nnue_layerstacks_v2_player(state_ptr_.get(), duration, loadname_input_));
                }

                else if (selected_player_type_ == "migoyugo_grave") {
                    // Deliberately no default: an empty name means "no
                    // network", which for GRAVE is the even-game heuristic and
                    // a perfectly good bot, not a misconfiguration.
                    players_.push_back(get_migoyugo_grave_player(state_ptr_.get(), duration, loadname_input_));
                }
            }
            catch (const std::invalid_argument&) {
                // Invalid duration, ignore
            }
            if (players_.size() != prev_player_count) {
                wins_.assign(players_.size(), 0);
                draws_ = 0;
            }
        }
        // Clear players button
        else if (CheckCollisionPointRec(mouse_pos, std::get<0>(buttons_[2]))) {
            players_.clear();
            wins_.clear();
            draws_ = 0;
        }
        // Start game button (only if >=2 players)
        else if (players_.size() >= 2 && CheckCollisionPointRec(mouse_pos, std::get<0>(buttons_[3]))) {
            reset_state();
            history_.clear();
            current_window_ = MigoyugoWindow::game;
            game_over_ = false;
            winner_ = -2;
        }
    }
}

void MigoyugoUI::perform_action(int action)
{
    auto actions_legality = state_ptr_->actions_mask();
    if (action < actions_legality.size() && actions_legality.at(action) && state_ptr_->is_terminal() == false)
    {
        history_.push_back(action);
        set_state(state_ptr_->step_state(action));
        PlaySound(move_sound_);
    }
}

int MigoyugoUI::compute_winner() const
{
    float r = state_ptr_->get_reward(); // relative to state_ptr_->player_turn()
    int to_move = state_ptr_->player_turn();
    if (r > 0.0f) return to_move;
    if (r < 0.0f) return 1 - to_move;
    return -1; // draw
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
    float cx = left + inner_cell_size_ / 2.0f;
    float cy = top + inner_cell_size_ / 2.0f;
    float radius = inner_cell_size_ * 0.38f;

    Color base = player == 0 ? RAYWHITE : Color{ 30, 30, 30, 255 };
    Color outline = player == 0 ? Color{ 60, 60, 60, 255 } : Color{ 10, 10, 10, 255 };

    // Drop shadow
    DrawCircle(cx + 3, cy + 3, radius, Fade(BLACK, 0.35f));
    // Base disc
    DrawCircle(cx, cy, radius, base);
    // Glossy highlight
    DrawEllipse(cx - radius * 0.35f, cy - radius * 0.35f, radius * 0.4f, radius * 0.28f, Fade(WHITE, 0.5f));
    // Crisp outline
    DrawRing({ cx, cy }, radius - 1.5f, radius, 0, 360, 32, outline);

    if (is_yugo)
    {
        Color emblem = player == 0 ? Color{ 180, 30, 30, 255 } : Color{ 220, 60, 60, 255 };
        // Accent ring marking this as a yugo
        DrawRing({ cx, cy }, radius + 2, radius + 5, 0, 360, 32, GOLD);
        // Diamond emblem
        DrawPoly({ cx, cy }, 4, radius * 0.42f, 45.0f, emblem);
        DrawPolyLines({ cx, cy }, 4, radius * 0.42f, 45.0f, BLACK);
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
            left = left_margin_ + col * cell_size_ + padding_;
            top = header_height_ + row * cell_size_ + padding_;
            Rectangle ol = { left, top, inner_cell_size_, inner_cell_size_ };
            DrawRectangleLinesEx(ol, 2.0f, GREEN);
        }
    }
}

Wave MigoyugoUI::synthesize_tone(float freq_hz, float duration_sec, ToneWave shape, float amplitude) const
{
    constexpr float kPi = 3.14159265358979323846f;
    unsigned int sample_rate = 44100;
    unsigned int frame_count = static_cast<unsigned int>(duration_sec * sample_rate);
    int16_t* samples = static_cast<int16_t*>(std::malloc(frame_count * sizeof(int16_t)));

    unsigned int fade_frames = static_cast<unsigned int>(0.008f * sample_rate);

    for (unsigned int i = 0; i < frame_count; i++)
    {
        float t = static_cast<float>(i) / sample_rate;
        float raw = sinf(2.0f * kPi * freq_hz * t);
        if (shape == ToneWave::Square)
        {
            raw = raw >= 0.0f ? 1.0f : -1.0f;
        }

        float env = 1.0f;
        if (fade_frames > 0 && frame_count > 2 * fade_frames)
        {
            if (i < fade_frames)
            {
                env = static_cast<float>(i) / fade_frames;
            }
            else if (i >= frame_count - fade_frames)
            {
                env = static_cast<float>(frame_count - i) / fade_frames;
            }
        }

        samples[i] = static_cast<int16_t>(raw * amplitude * env * 32000.0f);
    }

    Wave wave{};
    wave.frameCount = frame_count;
    wave.sampleRate = sample_rate;
    wave.sampleSize = 16;
    wave.channels = 1;
    wave.data = samples;
    return wave;
}

Wave MigoyugoUI::synthesize_sequence(const std::vector<std::pair<float, float>>& notes, ToneWave shape, float amplitude) const
{
    std::vector<Wave> note_waves;
    unsigned int total_frames = 0;
    for (const auto& note : notes)
    {
        Wave w = synthesize_tone(note.first, note.second, shape, amplitude);
        note_waves.push_back(w);
        total_frames += w.frameCount;
    }

    int16_t* combined = static_cast<int16_t*>(std::malloc(total_frames * sizeof(int16_t)));
    unsigned int offset = 0;
    for (Wave& w : note_waves)
    {
        std::memcpy(combined + offset, w.data, w.frameCount * sizeof(int16_t));
        offset += w.frameCount;
        UnloadWave(w);
    }

    Wave result{};
    result.frameCount = total_frames;
    result.sampleRate = 44100;
    result.sampleSize = 16;
    result.channels = 1;
    result.data = combined;
    return result;
}

Sound MigoyugoUI::load_tone_sound(Wave wave) const
{
    Sound sound = LoadSoundFromWave(wave);
    UnloadWave(wave);
    return sound;
}

} // namespace rl::ui
