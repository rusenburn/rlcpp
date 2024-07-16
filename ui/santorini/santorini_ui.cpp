#include "santorini_ui.hpp"
#include <iostream>
#include <players/random_rollout_evaluator.hpp>
#include <deeplearning/alphazero/networks/shared_res_nn.hpp>
#include <filesystem>
namespace rl::ui
{
    SantoriniUI::SantoriniUI(int width, int height)
        : width_{width}, height_{height}, padding_{2}, state_ptr_{rl::games::SantoriniState::initialize_state()},
          current_window_{SantoriniWindow::menu},
          players_{},
          selected_row_{-1},
          selected_col_{-1}

    {
        cell_size_ = width_ / SantoriniState::ROWS;
        inner_cell_size_ = cell_size_ - 2 * padding_;
        initialize_buttons();
        reset_state();
    }

    SantoriniUI::~SantoriniUI() = default;

    void SantoriniUI::draw_game()
    {
        if (current_window_ == SantoriniWindow::game)
        {
            draw_board();
        }
        else if (current_window_ == SantoriniWindow::menu)
        {
            draw_menu();
        }
    }

    void SantoriniUI::handle_events()
    {

        if (current_window_ == SantoriniWindow::game)
        {
            handle_board_events();
        }
        else if (current_window_ == SantoriniWindow::menu)
        {
            handle_menu_events();
        }
    }

    void SantoriniUI::set_state(SantoriniStatePtr new_state_ptr)
    {
        state_ptr_ = std::move(new_state_ptr);
        obs_ = state_ptr_->get_observation();
        actions_legality_ = state_ptr_->actions_mask();
        phase_ = state_ptr_->get_current_phase();
        selected_row_ = -1;
        selected_col_ = -1;

        constexpr int selection_channel = 2;
        constexpr int channel_size = SantoriniState::ROWS * SantoriniState::COLS;
        constexpr int cell_start = channel_size * 2;
        constexpr int cell_end = cell_start + channel_size;
        for (int cell = cell_start; cell < cell_end; cell++)
        {
            if (obs_.at(cell) == 1.0f)
            {
                int index = cell - cell_start;
                selected_row_ = index / SantoriniState::COLS;
                selected_col_ = index % SantoriniState::COLS;
                break;
            }
        }
    }

    void SantoriniUI::reset_state()
    {
        set_state(state_ptr_->reset_state());
    }

    void SantoriniUI::initialize_buttons()
    {
        float button_width = 100;
        float button_height = 20;

        float top, left;
        left = (width_ - button_width) / 2;
        top = 20;
        buttons_.push_back(std::make_pair<Rectangle, Color>(Rectangle{left, top, button_width, button_height}, GRAY));
    }

    void SantoriniUI::draw_board()
    {
        constexpr int ROWS = SantoriniState::ROWS;
        constexpr int COLS = SantoriniState::COLS;
        constexpr int CURRENT_PLAYER_CHANNEL = 0;
        constexpr int OPPONENT_PLAYER_CHANNEL = 1;
        constexpr int SELECTION_CHANNEL = 2;
        constexpr int GROUND_HEIGHT_CHANNEL = 3;
        constexpr int FLOOR1_CHANNEL = GROUND_HEIGHT_CHANNEL + 1;
        constexpr int FLOOR2_CHANNEL = FLOOR1_CHANNEL + 1;
        constexpr int FLOOR3_CHANNEL = FLOOR2_CHANNEL + 1;
        constexpr int DOME_CHANNEL = FLOOR3_CHANNEL + 1;
        constexpr int CHANNEL_SIZE = ROWS * COLS;
        int left, top, width, height, left_center, top_center;

        int current_player = state_ptr_->player_turn();

        for (int row = 0; row < ROWS; row++)
        {
            for (int col = 0; col < COLS; col++)
            {
                left = col * cell_size_ + padding_;
                top = row * cell_size_ + padding_;
                left_center = left + inner_cell_size_ / 2;
                top_center = top + inner_cell_size_ / 2;

                int ground_ind = GROUND_HEIGHT_CHANNEL * CHANNEL_SIZE + row * COLS + col;
                int floor1_ind = FLOOR1_CHANNEL * CHANNEL_SIZE + row * COLS + col;
                int floor2_ind = FLOOR2_CHANNEL * CHANNEL_SIZE + row * COLS + col;
                int floor3_ind = FLOOR3_CHANNEL * CHANNEL_SIZE + row * COLS + col;
                int dome_ind = DOME_CHANNEL * CHANNEL_SIZE + row * COLS + col;

                if (obs_.at(ground_ind) == 1.0f)
                {
                    draw_ground(left_center, top_center);
                }
                else if (obs_.at(floor1_ind) == 1.0f)
                {
                    draw_floor1(left_center, top_center);
                }
                else if (obs_.at(floor2_ind) == 1.0f)
                {
                    draw_floor2(left_center, top_center);
                }
                else if (obs_.at(floor3_ind) == 1.0f)
                {
                    draw_floor3(left_center, top_center);
                }
                else if (obs_.at(dome_ind) == 1.0f)
                {
                    draw_dome(left_center, top_center);
                }
                int player_0_channel = current_player == 0 ? 0 : 1;
                int player_1_channel = current_player == 0 ? 1 : 0;
                int player_0_ind = player_0_channel * CHANNEL_SIZE + row * COLS + col;
                int player_1_ind = player_1_channel * CHANNEL_SIZE + row * COLS + col;

                if (obs_.at(player_0_ind) == 1.0f)
                {
                    draw_piece(left, top, 0, false);
                }
                else if (obs_.at(player_1_ind) == 1.0f)
                {
                    draw_piece(left, top, 1, false);
                }

                Rectangle ol = {left, top, inner_cell_size_, inner_cell_size_};
                if (selected_row_ == row && selected_col_ == col)
                {
                    DrawRectangleLinesEx(ol, 2.0f, RED);
                }
            }
        }
        draw_legal_actions();
    }

    void SantoriniUI::draw_menu()
    {
        DrawRectangleRec(std::get<0>(buttons_.at(0)), std::get<1>(buttons_.at(0)));
    }

    void SantoriniUI::handle_board_events()
    {
        if (!paused_)
        {
            int current_player_ind = state_ptr_->player_turn();
            IPlayerPtr &current_player_ptr_ref = players_.at(current_player_ind);
            auto player_p = dynamic_cast<const rl::players::HumanPlayer *>(current_player_ptr_ref.get());
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
                current_window_ = SantoriniWindow::menu;
                paused_ = false;
            }
        }
    }

    void SantoriniUI::handle_menu_events()
    {
        if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON))
        {
            auto mouse_pos = GetMousePosition();
            auto [rec, col] = buttons_.at(0);
            auto is_button_pressed = CheckCollisionPointRec(mouse_pos, rec);
            if (is_button_pressed)
            {
                reset_state();
                current_window_ = SantoriniWindow::game;
                players_.clear();
                auto players_duration = std::chrono::duration<int, std::milli>(1000);
                // players_.push_back(std::make_unique<rl::players::HumanPlayer>());
                // players_.push_back(get_random_rollout_player_ptr(3, player_g_duration));
                players_.push_back(get_network_amcts_player(3, players_duration,"santorini_strongest_340.pt"));
                players_.push_back(get_network_amcts_player(3, players_duration*5,"santorini_strongest_220.pt"));
                // players_.push_back(get_network_amcts_player(3, players_duration*2,"santorini_strongest_120.pt"));
                // players_.push_back(get_random_rollout_player_ptr(3, player_g_duration));
            }
        }
    }

    void SantoriniUI::perform_action(int action)
    {
        auto actions_legality = state_ptr_->actions_mask();
        if (action < actions_legality.size() && actions_legality.at(action) && state_ptr_->is_terminal() == false)
        {
            set_state(state_ptr_->step_state(action));
        }
    }

    void SantoriniUI::perform_player_action(int row, int col)
    {
        int action = SantoriniState::encode_action(row, col);
        if (action < state_ptr_->get_n_actions() && actions_legality_.at(action))
        {
            perform_action(action);
        }
    }

    void SantoriniUI::draw_ground(int left_center, int top_center)
    {
        int left = left_center - inner_cell_size_ / 2;
        int top = top_center - inner_cell_size_ / 2;
        DrawRectangle(left, top, inner_cell_size_, inner_cell_size_, DARKGREEN);
    }

    void SantoriniUI::draw_floor1(int left_center, int top_center)
    {
        draw_ground(left_center, top_center);
        int size = inner_cell_size_ * 0.8;
        int left = left_center - size / 2;
        int top = top_center - size / 2;
        DrawRectangle(left, top, size, size, GRAY);
    }

    void SantoriniUI::draw_floor2(int left_center, int top_center)
    {
        draw_floor1(left_center, top_center);
        int size = inner_cell_size_ * 0.8 * 0.8;
        int left = left_center - size / 2;
        int top = top_center - size / 2;
        DrawRectangle(left, top, size, size, DARKGRAY);
    }

    void SantoriniUI::draw_floor3(int left_center, int top_center)
    {
        draw_floor2(left_center, top_center);
        int size = inner_cell_size_ * 0.8 * 0.8 * 0.8;
        int left = left_center - size / 2;
        int top = top_center - size / 2;
        DrawRectangle(left, top, size, size, RED);
    }

    void SantoriniUI::draw_dome(int left_center, int top_center)
    {
        draw_floor3(left_center, top_center);
        int size = inner_cell_size_ * 0.8 * 0.8 * 0.8;
        // DrawRectangle(left, top, size, size, DARKBLUE);
        DrawCircle(left_center, top_center, size / 2, DARKBLUE);
    }

    void SantoriniUI::draw_piece(int left, int top, int player, bool is_fade)
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
                DrawCircle(left_center, top_center, inner_cell_size_ / 4, FADE_BLACK);
            }
            else
            {
                DrawCircle(left_center, top_center, inner_cell_size_ / 4, BLACK);
            }
        }
        else
        {
            if (is_fade)
            {
                DrawCircle(left_center, top_center, inner_cell_size_ / 4, FADE_WHITE);
            }
            else
            {
                DrawCircle(left_center, top_center, inner_cell_size_ / 4, WHITE);
            }
        }
    }

    void SantoriniUI::draw_legal_actions()
    {
        int top, left;
        for (int action = 0; action < actions_legality_.size(); action++)
        {
            if (actions_legality_.at(action))
            {
                auto [row, col] = SantoriniState::decode_action(action);
                left = col * cell_size_ + padding_;
                top = row * cell_size_ + padding_;
                Rectangle ol = {left, top, inner_cell_size_, inner_cell_size_};
                DrawRectangleLinesEx(ol, 2.0f, GREEN);
            }
        }
    }

    std::unique_ptr<rl::players::GPlayer> SantoriniUI::get_default_g_player(int n_sims, std::chrono::duration<int, std::milli> minimum_duration)
    {
        return std::make_unique<rl::players::GPlayer>(n_sims, minimum_duration, 15, 0.04f);
    }

    std::unique_ptr<rl::players::MctsPlayer> SantoriniUI::get_random_rollout_player_ptr(int n_sims, std::chrono::duration<int, std::milli> minimum_duration)
    {
        auto ev_ptr = std::make_unique<rl::players::RandomRolloutEvaluator>(state_ptr_->get_n_actions());
        return std::make_unique<rl::players::MctsPlayer>(state_ptr_->get_n_actions(), ev_ptr->copy(), n_sims, minimum_duration, 1.0f, 2.0f);
    }

    std::unique_ptr<rl::players::AmctsPlayer> SantoriniUI::get_network_amcts_player(int n_sims, std::chrono::duration<int, std::milli> minimum_duration,std::string load_name)
    {
        auto network_ptr = std::make_unique<rl::deeplearning::alphazero::SharedResNetwork>(state_ptr_->get_observation_shape(), state_ptr_->get_n_actions(),
                                                                                           128, 512, 5);
        auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        // const std::string load_name = "santorini_strongest_120.pt";
        const std::string folder_name = "../checkpoints";
        std::filesystem::path folder(folder_name);
        std::filesystem::path file_path;
        file_path = folder / load_name;
        network_ptr->load(file_path.string());
        network_ptr->to(device);
        auto ev_ptr = std::make_unique<rl::deeplearning::NetworkEvaluator>(std::move(network_ptr), state_ptr_->get_n_actions(), state_ptr_->get_observation_shape());
        ev_ptr->evaluate(state_ptr_->clone());
        auto player_ptr = std::make_unique<rl::players::AmctsPlayer>(state_ptr_->get_n_actions(), std::move(ev_ptr), n_sims, minimum_duration, 0.5f, 2.0f, 8);
        return player_ptr;
    }

    std::unique_ptr<rl::players::MctsPlayer> SantoriniUI::get_network_mcts_player(int n_sims, std::chrono::duration<int, std::milli> minimum_duration,std::string load_name)
    {
        auto network_ptr = std::make_unique<rl::deeplearning::alphazero::SharedResNetwork>(state_ptr_->get_observation_shape(), state_ptr_->get_n_actions(),
                                                                                           128, 512, 5);
        auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        // const std::string load_name = "santorini_strongest_120.pt";
        const std::string folder_name = "../checkpoints";
        std::filesystem::path folder(folder_name);
        std::filesystem::path file_path;
        file_path = folder / load_name;
        network_ptr->load(file_path.string());
        network_ptr->to(device);
        auto ev_ptr = std::make_unique<rl::deeplearning::NetworkEvaluator>(std::move(network_ptr), state_ptr_->get_n_actions(), state_ptr_->get_observation_shape());
        ev_ptr->evaluate(state_ptr_->clone());
        auto player_ptr = std::make_unique<rl::players::MctsPlayer>(state_ptr_->get_n_actions(), std::move(ev_ptr), n_sims, minimum_duration, 0.5f, 2.0f);
        return player_ptr;
    }

    std::unique_ptr<rl::players::LMMctsPlayer> SantoriniUI::get_network_lm_mcts_player(int n_sims, std::chrono::duration<int, std::milli> minimum_duration, std::string load_name)
    {
            auto network_ptr = std::make_unique<rl::deeplearning::alphazero::SharedResNetwork>(state_ptr_->get_observation_shape(), state_ptr_->get_n_actions(),
                                                                                           128, 512, 5);
        auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        // const std::string load_name = "santorini_strongest_120.pt";
        const std::string folder_name = "../checkpoints";
        std::filesystem::path folder(folder_name);
        std::filesystem::path file_path;
        file_path = folder / load_name;
        network_ptr->load(file_path.string());
        network_ptr->to(device);
        auto ev_ptr = std::make_unique<rl::deeplearning::NetworkEvaluator>(std::move(network_ptr), state_ptr_->get_n_actions(), state_ptr_->get_observation_shape());
        ev_ptr->evaluate(state_ptr_->clone());
        auto player_ptr = std::make_unique<rl::players::LMMctsPlayer>(state_ptr_->get_n_actions(), std::move(ev_ptr), n_sims, minimum_duration, 0.5f, 2.0f);
        return player_ptr;
    }

} // namespace rl::ui
