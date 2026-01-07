#include "gobblet_goblers_ui.hpp"
#include "../players_utils.hpp"

namespace rl::ui {

GobbletGoblersUI::GobbletGoblersUI(int width, int height)
    :width_{ width },
    height_{ height },
    padding_{ 2 },
    state_ptr_{ rl::games::GobbletGoblersState::initialize_state() },
    selected_row_{ -1 },
    selected_col_{ -1 },
    selected_size_{ 0 },
    players_{},
    board_{},
    current_window_{ GobbletGoblersWindow::menu }
{
    cell_size_ = width_ / (GobbletGoblersState::ROWS + 2);
    inner_cell_size_ = cell_size_ - 2 * padding_;
    initialize_buttons();
    reset_state();
}
GobbletGoblersUI::~GobbletGoblersUI() = default;

void GobbletGoblersUI::draw_game()
{
    if (current_window_ == GobbletGoblersWindow::menu)
    {
        draw_menu();
    }
    else if (current_window_ == GobbletGoblersWindow::game)
    {
        draw_game();
    }
}

void GobbletGoblersUI::handle_events()
{
    if (current_window_ == GobbletGoblersWindow::menu)
    {
        handle_menu_events();
    }
    else if (current_window_ == GobbletGoblersWindow::game)
    {
        handle_board_events();
    }
}

void GobbletGoblersUI::set_state(GobbletGoblersStatePtr new_state_ptr)
{
    state_ptr_ = std::move(new_state_ptr);
    obs_ = state_ptr_->get_observation();
    actions_legality_ = state_ptr_->actions_mask();
    selected_row_ = -1;
    selected_col_ = -1;

    constexpr int selection_channel = GobbletGoblersState::SELECTED_PIECE_ONBOARD_CHANNEL;
    constexpr int channel_size = GobbletGoblersState::ROWS * GobbletGoblersState::COLS;
    constexpr int cell_start = selection_channel * channel_size;
    constexpr int cell_end = cell_start + channel_size;

    for (int cell = cell_start; cell < cell_end; cell++)
    {
        if (obs_.at(cell) == 1.0f)
        {
            int index = cell - cell_start;
            selected_row_ = index / GobbletGoblersState::COLS;
            selected_col_ = index % GobbletGoblersState::COLS;
            break;
        }
    }

    constexpr int selected_small_size_channel = GobbletGoblersState::SELECTED_PIECE_SMALL_CHANNEL;
    constexpr int selected_small_size_cell = selected_small_size_channel * channel_size;
    constexpr int selected_medium_size_cell = selected_small_size_cell + channel_size;
    constexpr int selected_large_size_cell = selected_medium_size_cell + channel_size;

    if (obs_.at(selected_small_size_cell))
    {
        selected_size_ = 1;
    }
    else if (obs_.at(selected_medium_size_cell))
    {
        selected_size_ = 2;
    }
    else if (obs_.at(selected_large_size_cell))
    {
        selected_size_ = 3;
    }


    auto board = state_ptr_->get_board();
    if (state_ptr_->player_turn() == 0)
    {
        board_ = board;
    }
    else {
        for (int row = 0;row < rl::games::GobbletGoblersState::ROWS;row++)
        {
            for (int col = 0;col < rl::games::GobbletGoblersState::COLS;col++)
            {
                int cell = board.at(row).at(col);

                board_.at(row).at(col) = -cell;
            }
        }
    }
}

void GobbletGoblersUI::reset_state()
{
    set_state(state_ptr_->reset_state());
}


void GobbletGoblersUI::initialize_buttons()
{
    float button_width = 100;
    float button_height = 20;

    float top, left;
    left = (width_ - button_width) / 2;
    top = 20;
    buttons_.push_back(std::make_pair<Rectangle, Color>(Rectangle{ left, top, button_width, button_height }, GRAY));
}
void GobbletGoblersUI::draw_board()
{
    constexpr int ROWS = GobbletGoblersState::ROWS;
    constexpr int COLS = GobbletGoblersState::COLS;
    constexpr int CURRENT_PLAYER_CHANNEL = 0;
    constexpr int OPPONENT_PLAYER_CHANNEL = CURRENT_PLAYER_CHANNEL + 3;


    int left, top, width, height, left_center, top_center;

    int current_player = state_ptr_->player_turn();


    DrawRectangle(0, 0, width_, height_, BLACK);

    for (int row = 0; row < ROWS; row++)
    {
        for (int col = 0; col < COLS; col++)
        {
            left = (col + 1) * cell_size_ + padding_;
            top = (row + 1) * cell_size_ + padding_;
            left_center = left + inner_cell_size_ / 2;
            top_center = top + inner_cell_size_ / 2;

            draw_ground(left_center, top_center);
            int cell = board_.at(row).at(col);


            if (cell != 0)
            {
                int size = cell > 0 ? cell : -cell;
                bool player = static_cast<int>(cell > 0);
                draw_piece(left_center, top_center, player);
            }
        }
    }

    // draw outer pieces 
    // draw selected pieces

    draw_legal_actions();
}

void GobbletGoblersUI::draw_menu()
{
    DrawRectangleRec(std::get<0>(buttons_.at(0)), std::get<1>(buttons_.at(0)));
}

void GobbletGoblersUI::handle_board_events()
{
    if (!paused_)
    {
        int current_player_ind = state_ptr_->player_turn();
        auto& current_player_ptr_ref = players_.at(current_player_ind);
        auto player_p = dynamic_cast<const rl::players::HumanPlayer*>(current_player_ptr_ref->player_ptr_.get());
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
            // int action = current_player_ptr_ref-> ->choose_action(state_ptr_->clone_state());
            int action = current_player_ptr_ref->player_ptr_->choose_action(state_ptr_->clone_state());
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
            current_window_ = GobbletGoblersWindow::menu;
            paused_ = false;
        }
    }
}


void GobbletGoblersUI::handle_menu_events()
{
    if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON))
    {
        auto mouse_pos = GetMousePosition();
        auto [rec, col] = buttons_.at(0);
        auto is_button_pressed = CheckCollisionPointRec(mouse_pos, rec);
        if (is_button_pressed)
        {
            reset_state();
            current_window_ = GobbletGoblersWindow::game;
            players_.clear();
            auto players_duration = std::chrono::milliseconds(1000);

            players_.push_back(get_human_player(state_ptr_.get()));
            players_.push_back(get_network_amcts2_player(state_ptr_.get(), 3, players_duration, "gobblet_strongest_240.pt"));
        }
    }
}

void GobbletGoblersUI::perform_action(int action)
{
    auto actions_legality = state_ptr_->actions_mask();
    if (action < actions_legality.size() && actions_legality.at(action) && state_ptr_->is_terminal() == false)
    {
        set_state(state_ptr_->step_state(action));
    }
}

void GobbletGoblersUI::perform_player_action(int row, int col)
{
    int action = -1;
    if (row < 1 || row > 3)
    {
        return;
    }

    if (col == 0 || col == 4) {
        // process picking a peice action
        int player = state_ptr_->player_turn();
        if (player == 0 && col == 0)
        {
            action = row;
        }
        else if (player == 1 && col == 4)
        {
            action = row;
        }
        action += GobbletGoblersState::COLS * GobbletGoblersState::ROWS;
    }

    if (row > 0 && row < 4 && col >0 && col < 4)
    {
        row--;
        col--;

        action = row * GobbletGoblersState::COLS + col;
    }

    if (action >= 0 && action < state_ptr_->get_n_actions() && actions_legality_.at(action))
    {
        perform_action(action);
    }
}

}
