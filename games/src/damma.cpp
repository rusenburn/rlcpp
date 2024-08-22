#include <games/damma.hpp>
#include <common/exceptions.hpp>
#include <sstream>
#include <iostream>

namespace rl::games
{

DammaState::DammaState(Board board, int n_no_capture_rounds, std::optional<std::pair<int, int>> last_jump, std::vector<bool> last_jump_action_mask, int current_player)
    : board_(board),
    n_no_capture_rounds_{ n_no_capture_rounds },
    last_jump_{ last_jump },
    last_jump_action_mask_(last_jump_action_mask),
    current_player_{ current_player },
    cached_actions_masks_{},
    cached_is_terminal_{},
    cached_result_{},
    cached_observation_{}
{
}

DammaState::~DammaState() = default;

std::unique_ptr<DammaState> DammaState::initialize_state()
{
    Board board{
        {{0, 0, 0, 0, 0, 0, 0, 0},
         {-1, -1, -1, -1, -1, -1, -1, -1},
         {-1, -1, -1, -1, -1, -1, -1, -1},
         {0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0},
         {1, 1, 1, 1, 1, 1, 1, 1},
         {1, 1, 1, 1, 1, 1, 1, 1},
         {0, 0, 0, 0, 0, 0, 0, 0}} };
    constexpr int N_NO_CAPTURE_ROUNDS = 0;
    std::optional<std::pair<int, int>> last_jump{};
    std::vector<bool> last_jump_action_mask{};
    constexpr int STARTING_PLAYER = 0;
    return std::make_unique<DammaState>(board, N_NO_CAPTURE_ROUNDS, last_jump, last_jump_action_mask, STARTING_PLAYER);
}

std::unique_ptr<rl::common::IState> DammaState::initialize()
{
    return initialize_state();
}

std::unique_ptr<DammaState> DammaState::reset_state() const
{
    return initialize_state();
}

std::unique_ptr<rl::common::IState> DammaState::reset() const
{
    return reset_state();
}

std::unique_ptr<DammaState> DammaState::step_state(int action) const
{
    if (is_terminal())
    {
        std::stringstream ss;
        ss << "Stepping a terminal Damma state";
        throw rl::common::SteppingTerminalStateException(ss.str());
    }
    if (actions_mask().at(action) == false)
    {
        std::stringstream ss;
        ss << "Stepping a Damma state with an illegal action " << action;
        auto [row, col, target_row, target_col] = decode_action(action);
        ss << "\nDecoded action row: " << row << " col " << col << "target row " << target_row << "target col " << target_col << "\n";
        ss << to_short();
        std::vector<int> legal_actions{};
        auto am = actions_mask();
        if (last_jump_.has_value())
        {
            auto [jumprow, jumpcol] = last_jump_.value();
            ss << "\nlast jump is (" << jumprow << "," << jumpcol << ")";
        }
        else
        {
            ss << "\nno last jump , " << last_jump_action_mask_.size();
        }
        ss << "\nlegal actions are: ";
        for (int a = 0; a < am.size(); a++)
        {
            if (am.at(action))
            {
                ss << a << ",";
            }
        }

        throw rl::common::IllegalActionException(ss.str());
    }

    auto [row, col, target_row, target_col] = decode_action(action);
    Board new_board = board_;
    if (new_board.at(row).at(col) <= 0)
    {
        std::stringstream ss;
        ss << "Stepping a Damma state with action " << action << " moving from row " << row << " col " << col << "but player has no pieces there";
        throw rl::common::UnreachableCodeException(ss.str());
    }

    // move current piece
    new_board.at(target_row).at(target_col) = new_board.at(row).at(col);

    // check if there is a promotion
    if (target_row == ROWS - 1 || target_row == 0)
    {
        new_board.at(target_row).at(target_col) = PLAYER_K_CELL;
    }

    // remove moved piece
    new_board.at(row).at(col) = 0;
    int row_dir, col_dir; // can be 0 or 1 or -1;
    row_dir = target_row - row == 0 ? 0 : (target_row - row) / abs(target_row - row);
    col_dir = target_col - col == 0 ? 0 : (target_col - col) / abs(target_col - col);

    int current_row = row + row_dir;
    int current_col = col + col_dir;
    bool capture = false;
    while (current_row != target_row || current_col != target_col)
    {
        if (new_board.at(current_row).at(current_col) != 0)
        {
            capture = true;
            if (new_board.at(current_row).at(current_col) >= 0)
            {
                throw rl::common::UnreachableCodeException("Damma state assertion failed");
            }
        }

        // clear the path

        new_board.at(current_row).at(current_col) = 0;
        current_col += col_dir;
        current_row += row_dir;
    }
    int next_player;
    std::optional<std::pair<int, int>> new_last_jump{};
    std::vector<bool> new_last_jump_action_mask{};
    int new_no_capture_rounds = 0;
    if (capture)
    {
        std::vector<bool> new_capture_actions_mask(N_ACTIONS, 0);
        std::vector<bool> new_action_mask_no_capture{};
        assign_legal_action(new_board, target_row, target_col, true, new_action_mask_no_capture, new_capture_actions_mask);
        bool can_double_jump = false;
        for (bool i : new_capture_actions_mask)
        {
            if (i)
            {
                can_double_jump = true;
                break;
            }
        }

        if (can_double_jump)
        {
            next_player = current_player_;
            new_last_jump.emplace(std::make_pair(target_row, target_col));
            new_last_jump_action_mask = new_capture_actions_mask;
        }
        else
        {
            next_player = 1 - current_player_;
            swap_board_view(new_board);
        }
    }
    else
    {
        next_player = 1 - current_player_;
        new_no_capture_rounds = n_no_capture_rounds_ + 1;
        swap_board_view(new_board);
    }

    return std::make_unique<DammaState>(new_board, new_no_capture_rounds, new_last_jump, new_last_jump_action_mask, next_player);
}

std::unique_ptr<rl::common::IState> DammaState::step(int action) const
{
    return step_state(action);
}

void DammaState::render() const
{
    Board board = board_;
    if (current_player_ == 1)
    {
        swap_board_view(board);
    }

    std::stringstream ss;
    int player_0_pawn = 1;
    int player_0_king = 2;
    int player_1_pawn = -1;
    int player_1_king = -2;

    for (int row = 0; row < ROWS; row++)
    {
        for (int col = 0; col < COLS; col++)
        {
            int cell = board.at(row).at(col);
            if (cell == 0)
            {
                ss << " . ";
            }
            else if (cell == player_0_pawn)
            {
                ss << " x ";
            }
            else if (cell == player_0_king)
            {
                ss << " X ";
            }
            else if (cell == player_1_pawn)
            {
                ss << " o ";
            }
            else if (cell == player_1_king)
            {
                ss << " O ";
            }
        }
        ss << "\n";
    }
    if (is_terminal())
    {
        ss << "Game ended";
    }
    else if (current_player_ == 0)
    {
        ss << "Player X has to move";
    }
    else if (current_player_ == 1)
    {
        ss << "Player O has to move";
    }

    std::cout << ss.str() << std::endl;
}

bool DammaState::is_terminal() const
{
    if (cached_is_terminal_.has_value())
    {
        return cached_is_terminal_.value();
    }
    if (is_opponent_win() || is_draw())
    {
        cached_is_terminal_.emplace(true);
        return cached_is_terminal_.value();
    }
    cached_is_terminal_.emplace(false);
    return cached_is_terminal_.value();
}

float DammaState::get_reward() const
{
    if (!is_terminal())
    {
        return 0.0f;
    }

    if (cached_result_.has_value())
    {
        return cached_result_.value();
    }

    if (is_opponent_win())
    {
        cached_result_.emplace(-1.0f);
        return cached_result_.value();
    }
    if (is_draw())
    {
        cached_result_.emplace(0.0f);
        return cached_result_.value();
    }
    throw rl::common::UnreachableCodeException("DammaState get result unreachable error");
    cached_result_.emplace(1.0f);
    return cached_result_.value();
}

std::vector<float> DammaState::get_observation() const
{
    if (cached_observation_.size())
    {
        return cached_observation_;
    }

    std::vector<float> cached_observation_ = std::vector<float>(CHANNELS * ROWS * COLS, 0);
    constexpr int player_0_pawn = 1;
    constexpr int player_0_king = 2;
    constexpr int player_1_pawn = -1;
    constexpr int player_1_king = -2;
    constexpr int PLAYER_P_CHANNEL = 0;
    constexpr int PLAYER_K_CHANNEL = 1;
    constexpr int OPPONENT_P_CHANNEL = 2;
    constexpr int OPPONENT_K_CHANNEL = 3;

    const int channel_size = ROWS * COLS;
    for (int row{ 0 }; row < ROWS; row++)
    {
        for (int col{ 0 }; col < COLS; col++)
        {
            int cell_value = board_.at(row).at(col);
            int current_channel{ -1 };
            if (cell_value == player_0_pawn)
            {
                current_channel = PLAYER_P_CHANNEL;
            }
            else if (cell_value == player_0_king)
            {
                current_channel = PLAYER_K_CHANNEL;
            }
            else if (cell_value == player_1_pawn)
            {
                current_channel = OPPONENT_P_CHANNEL;
            }
            else if (cell_value == player_1_king)
            {
                current_channel = OPPONENT_K_CHANNEL;
            }
            else
            {
                continue;
            }
            if (current_channel == -1)
            {
                continue;
            }
            int observation_cell_id = channel_size * current_channel + row * COLS + col;
            cached_observation_.at(observation_cell_id) = 1.0f;
        }
    }
    add_no_capture_rounds_observation_(cached_observation_);
    add_last_jump_observation_(cached_observation_);
    // add_current_player_turn_observation_(cached_observation_);

    return cached_observation_;
}

std::string DammaState::to_short() const
{
    std::stringstream ss;
    int empty_count = 0;
    for (int row{ 0 }; row < ROWS; row++)
    {
        for (int col{ 0 }; col < COLS; col++)
        {
            if (board_.at(row).at(col) == 0)
            {
                empty_count++;
            }
            else
            {
                if (empty_count)
                {
                    ss << empty_count;
                    empty_count = 0;
                }
                if (board_.at(row).at(col) == 2)
                {
                    ss << 'X';
                }
                else if (board_.at(row).at(col) == 1)
                {
                    ss << 'x';
                }
                else if (board_.at(row).at(col) == -2)
                {
                    ss << 'O';
                }
                else if (board_.at(row).at(col) == -1)
                {
                    ss << 'o';
                }
                else
                {
                    throw rl::common::UnreachableCodeException("");
                }
            }
        }
    }
    if (empty_count)
    {
        ss << empty_count;
        empty_count = 0;
    }

    ss << "#" << n_no_capture_rounds_ << "#" << current_player_;
    if (last_jump_.has_value())
    {
        auto [row, col] = last_jump_.value();
        ss << "#" << row << "," << col;
    }
    return ss.str();
}

std::array<int, 3> DammaState::get_observation_shape() const
{
    return { CHANNELS, ROWS, COLS };
}

int DammaState::get_n_actions() const
{
    return N_ACTIONS;
}

int DammaState::player_turn() const
{
    return current_player_;
}

std::vector<bool> DammaState::actions_mask() const
{
    if (cached_actions_masks_.size())
    {
        return cached_actions_masks_;
    }
    if (last_jump_action_mask_.size())
    {
        cached_actions_masks_ = last_jump_action_mask_;
        return cached_actions_masks_;
    }

    if (last_jump_.has_value())
    {
        throw rl::common::UnreachableCodeException("Last Jump has value");
    }

    bool no_capture = true;
    std::vector<bool> actions_legality_no_capture = std::vector<bool>(N_ACTIONS);
    std::vector<bool> actions_legality_capture = std::vector<bool>(N_ACTIONS);

    for (int row = 0; row < ROWS; row++)
    {
        for (int col = 0; col < COLS; col++)
        {
            if (board_.at(row).at(col) == 0)
                continue;

            else if (board_.at(row).at(col) == PLAYER_P_CELL)
            {
                int cell_action_index = (row * COLS + col) * (ROWS + COLS - 2);
                int target_index;
                if (col - 1 >= 0)
                {
                    if (board_.at(row).at(col - 1) == 0 && no_capture)
                    {
                        // left is empty
                        target_index = col - 1;
                        int legal_action = cell_action_index + target_index;
                        actions_legality_no_capture.at(legal_action) = 1;
                    }
                    else if (col - 2 >= 0 && board_.at(row).at(col - 1) < 0 && board_.at(row).at(col - 2) == 0)
                    {
                        // left has an opponent that can be captured
                        target_index = col - 2;
                        int legal_action = cell_action_index + target_index;
                        no_capture = false;
                        actions_legality_capture.at(legal_action) = 1;
                    }
                }
                if (col + 1 < COLS)
                {
                    if (board_.at(row).at(col + 1) == 0 && no_capture)
                    {
                        // right is empty
                        target_index = col + 1 - 1;
                        int legal_action = cell_action_index + target_index;
                        actions_legality_no_capture.at(legal_action) = 1;
                    }
                    else if (col + 2 < COLS && board_.at(row).at(col + 1) < 0 && board_.at(row).at(col + 2) == 0)
                    {
                        // right can be captured
                        target_index = col + 2 - 1;
                        int legal_action = cell_action_index + target_index;
                        no_capture = false;
                        actions_legality_capture.at(legal_action) = 1;
                    }
                }
                if (row - 1 >= 0)
                {
                    if (board_.at(row - 1).at(col) == 0 && no_capture)
                    {
                        target_index = COLS - 1 + row - 1;
                        int legal_action = cell_action_index + target_index;
                        actions_legality_no_capture.at(legal_action) = 1;
                    }
                    else if (row - 2 >= 0 && board_.at(row - 1).at(col) < 0 && board_.at(row - 2).at(col) == 0)
                    {
                        target_index = COLS - 1 + row - 2;
                        int legal_action = cell_action_index + target_index;
                        no_capture = false;
                        actions_legality_capture.at(legal_action) = 1;
                    }
                }
            }
            else if (board_.at(row).at(col) == PLAYER_K_CELL)
            {
                int cell_action_index = (row * COLS + col) * (ROWS + COLS - 2);
                for (int col_dir = 1; col_dir < COLS; col_dir++)
                {
                    int target_col = col - col_dir;
                    if (target_col < 0)
                        break;

                    if (board_.at(row).at(target_col) == 0)
                    {
                        if (no_capture)
                        {
                            int target_index = target_col;
                            int legal_action = cell_action_index + target_index;
                            actions_legality_no_capture.at(legal_action) = 1;
                        }
                    }
                    else
                    {
                        if (board_.at(row).at(target_col) < 0)
                        {
                            int second_target = target_col - 1;
                            while (second_target >= 0 && board_.at(row).at(second_target) == 0)
                            {
                                int target_index = second_target;
                                int legal_action = cell_action_index + target_index;
                                no_capture = false;
                                actions_legality_capture.at(legal_action) = 1;
                                second_target--;
                            }
                        }
                        break;
                    }
                }
                for (int col_dir = 1; col_dir < COLS; col_dir++)
                {
                    int target_col = col + col_dir;
                    if (target_col >= COLS)
                        break;

                    if (board_.at(row).at(target_col) == 0)
                    {
                        if (no_capture)
                        {
                            int target_index = target_col - 1;
                            int legal_action = cell_action_index + target_index;
                            actions_legality_no_capture.at(legal_action) = 1;
                        }
                    }
                    else
                    {
                        if (board_.at(row).at(target_col) < 0)
                        {
                            int second_target = target_col + 1;
                            while (second_target < COLS && board_.at(row).at(second_target) == 0)
                            {
                                int target_index = second_target - 1;
                                int legal_action = cell_action_index + target_index;
                                no_capture = false;
                                actions_legality_capture.at(legal_action) = 1;
                                second_target++;
                            }
                        }
                        break;
                    }
                }
                for (int row_dir = 1; row_dir < ROWS; row_dir++)
                {
                    int target_row = row - row_dir;
                    if (target_row < 0)
                        break;

                    if (board_.at(target_row).at(col) == 0)
                    {
                        if (no_capture)
                        {
                            int target_index = COLS - 1 + target_row;
                            int legal_action = cell_action_index + target_index;
                            actions_legality_no_capture.at(legal_action) = 1;
                        }
                    }
                    else
                    {
                        if (board_.at(target_row).at(col) < 0)
                        {
                            int second_target = target_row - 1;
                            while (second_target >= 0 && board_.at(second_target).at(col) == 0)
                            {
                                int target_index = COLS - 1 + second_target;
                                int legal_action = cell_action_index + target_index;
                                actions_legality_capture.at(legal_action) = 1;
                                no_capture = false;
                                second_target--;
                            }
                        }
                        break;
                    }
                }
                for (int row_dir = 1; row_dir < ROWS; row_dir++)
                {
                    int target_row = row + row_dir;
                    if (target_row >= ROWS)
                        break;

                    if (board_.at(target_row).at(col) == 0)
                    {
                        if (no_capture)
                        {
                            int target_index = COLS - 1 + target_row - 1;
                            int legal_action = cell_action_index + target_index;
                            actions_legality_no_capture.at(legal_action) = 1;
                        }
                    }
                    else
                    {
                        if (board_.at(target_row).at(col) < 0)
                        {
                            int second_target = target_row + 1;
                            while (second_target < ROWS && board_.at(second_target).at(col) == 0)
                            {
                                int target_index = COLS - 1 + second_target - 1;
                                int legal_action = cell_action_index + target_index;
                                actions_legality_capture.at(legal_action) = 1;
                                no_capture = false;
                                second_target++;
                            }
                        }
                        break;
                    }
                }
            }
        }
    }

    if (no_capture)
    {
        cached_actions_masks_ = actions_legality_no_capture;
    }
    else
    {
        cached_actions_masks_ = actions_legality_capture;
    }
    return cached_actions_masks_;
}

std::unique_ptr<DammaState> DammaState::clone_state() const
{
    return std::unique_ptr<DammaState>(new DammaState(*this));
}

std::unique_ptr<rl::common::IState> DammaState::clone() const
{
    return clone_state();
}

void DammaState::get_symmetrical_obs_and_actions(std::vector<float> const& obs, std::vector<float> const& actions_distribution, std::vector<std::vector<float>>& out_syms, std::vector<std::vector<float>>& out_actions_distribution) const
{
    out_syms.clear();
    out_actions_distribution.clear();
}
std::tuple<int, int, int, int> DammaState::decode_action(int action)
{
    int target_index = action % (ROWS + COLS - 2);
    int cell_index = action / (ROWS + COLS - 2);
    int row = cell_index / COLS;
    int col = cell_index % COLS;
    int target_row;
    int target_col;
    if (target_index < COLS - 1)
    {
        target_row = row;
        target_col = target_index >= col ? target_index + 1 : target_index;
    }
    else
    {
        target_index -= (COLS - 1);
        target_col = col;
        target_row = target_index >= row ? target_index + 1 : target_index;
    }
    return std::make_tuple(row, col, target_row, target_col);
}

int DammaState::encode_action(int row, int col, int target_row, int target_col)
{
    int row_dir, col_dir;
    row_dir = target_row - row;
    col_dir = target_col - col;
    int cell_action_index = (row * COLS + col) * (ROWS + COLS - 2);
    if (row_dir != 0 && col_dir != 0)
    {
        throw rl::common::UnreachableCodeException("[DammaState] either row direction or col direction is 0 but none is");
    }

    if (col_dir != 0)
    {
        int target_index = col_dir > 0 ? target_col - 1 : target_col;
        int action = cell_action_index + target_index;
        return action;
    }
    if (row_dir != 0)
    {
        int target_index = row_dir > 0 ? COLS - 1 + target_row - 1 : COLS - 1 + target_row;
        int action = cell_action_index + target_index;
        return action;
    }
    throw rl::common::UnreachableCodeException("[DammaState] either row direction or col direction is 0 but both are");
}
bool DammaState::assign_legal_action(Board& board, int row, int col, bool capture_only, std::vector<bool>& actions_legality_no_capture_out, std::vector<bool>& actions_legality_capture_out)
{
    bool no_capture = !capture_only;

    if (board.at(row).at(col) == 0)
        return no_capture;

    else if (board.at(row).at(col) == PLAYER_P_CELL)
    {
        int cell_action_index = (row * COLS + col) * (ROWS + COLS - 2);
        int target_index;
        if (col - 1 >= 0)
        {
            if (board.at(row).at(col - 1) == 0 && no_capture)
            {
                // left is empty
                target_index = col - 1;
                int legal_action = cell_action_index + target_index;
                actions_legality_no_capture_out.at(legal_action) = 1;
            }
            else if (col - 2 >= 0 && board.at(row).at(col - 1) < 0 && board.at(row).at(col - 2) == 0)
            {
                // left has an opponent that can be captured
                target_index = col - 2;
                int legal_action = cell_action_index + target_index;
                no_capture = false;
                actions_legality_capture_out.at(legal_action) = 1;
            }
        }
        if (col + 1 < COLS)
        {
            if (board.at(row).at(col + 1) == 0 && no_capture)
            {
                // right is empty
                target_index = col + 1 - 1;
                int legal_action = cell_action_index + target_index;
                actions_legality_no_capture_out.at(legal_action) = 1;
            }
            else if (col + 2 < COLS && board.at(row).at(col + 1) < 0 && board.at(row).at(col + 2) == 0)
            {
                // right can be captured
                target_index = col + 2 - 1;
                int legal_action = cell_action_index + target_index;
                no_capture = false;
                actions_legality_capture_out.at(legal_action) = 1;
            }
        }
        if (row - 1 >= 0)
        {
            if (board.at(row - 1).at(col) == 0 && no_capture)
            {
                target_index = COLS - 1 + row - 1;
                int legal_action = cell_action_index + target_index;
                actions_legality_no_capture_out.at(legal_action) = 1;
            }
            else if (row - 2 >= 0 && board.at(row - 1).at(col) < 0 && board.at(row - 2).at(col) == 0)
            {
                target_index = COLS - 1 + row - 2;
                int legal_action = cell_action_index + target_index;
                no_capture = false;
                actions_legality_capture_out.at(legal_action) = 1;
            }
        }
    }
    else if (board.at(row).at(col) == PLAYER_K_CELL)
    {
        int cell_action_index = (row * COLS + col) * (ROWS + COLS - 2);
        for (int col_dir = 1; col_dir < COLS; col_dir++)
        {
            int target_col = col - col_dir;
            if (target_col < 0)
                break;

            if (board.at(row).at(target_col) == 0)
            {
                if (no_capture)
                {
                    int target_index = target_col;
                    int legal_action = cell_action_index + target_index;
                    actions_legality_no_capture_out.at(legal_action) = 1;
                }
            }
            else
            {
                if (board.at(row).at(target_col) < 0)
                {
                    int second_target = target_col - 1;
                    while (second_target >= 0 && board.at(row).at(second_target) == 0)
                    {
                        int target_index = second_target;
                        int legal_action = cell_action_index + target_index;
                        no_capture = false;
                        actions_legality_capture_out.at(legal_action) = 1;
                        second_target--;
                    }
                }
                break;
            }
        }
        for (int col_dir = 1; col_dir < COLS; col_dir++)
        {
            int target_col = col + col_dir;
            if (target_col >= COLS)
                break;

            if (board.at(row).at(target_col) == 0)
            {
                if (no_capture)
                {
                    int target_index = target_col - 1;
                    int legal_action = cell_action_index + target_index;
                    actions_legality_no_capture_out.at(legal_action) = 1;
                }
            }
            else
            {
                if (board.at(row).at(target_col) < 0)
                {
                    int second_target = target_col + 1;
                    while (second_target < COLS && board.at(row).at(second_target) == 0)
                    {
                        int target_index = second_target - 1;
                        int legal_action = cell_action_index + target_index;
                        no_capture = false;
                        actions_legality_capture_out.at(legal_action) = 1;
                        second_target++;
                    }
                }
                break;
            }
        }
        for (int row_dir = 1; row_dir < ROWS; row_dir++)
        {
            int target_row = row - row_dir;
            if (target_row < 0)
                break;

            if (board.at(target_row).at(col) == 0)
            {
                if (no_capture)
                {
                    int target_index = COLS - 1 + target_row;
                    int legal_action = cell_action_index + target_index;
                    actions_legality_no_capture_out.at(legal_action) = 1;
                }
            }
            else
            {
                if (board.at(target_row).at(col) < 0)
                {
                    int second_target = target_row - 1;
                    while (second_target >= 0 && board.at(second_target).at(col) == 0)
                    {
                        int target_index = COLS - 1 + second_target;
                        int legal_action = cell_action_index + target_index;
                        actions_legality_capture_out.at(legal_action) = 1;
                        no_capture = false;
                        second_target--;
                    }
                }
                break;
            }
        }
        for (int row_dir = 1; row_dir < ROWS; row_dir++)
        {
            int target_row = row + row_dir;
            if (target_row >= ROWS)
                break;

            if (board.at(target_row).at(col) == 0)
            {
                if (no_capture)
                {
                    int target_index = COLS - 1 + target_row - 1;
                    int legal_action = cell_action_index + target_index;
                    actions_legality_no_capture_out.at(legal_action) = 1;
                }
            }
            else
            {
                if (board.at(target_row).at(col) < 0)
                {
                    int second_target = target_row + 1;
                    while (second_target < ROWS && board.at(second_target).at(col) == 0)
                    {
                        int target_index = COLS - 1 + second_target - 1;
                        int legal_action = cell_action_index + target_index;
                        actions_legality_capture_out.at(legal_action) = 1;
                        no_capture = false;
                        second_target++;
                    }
                }
                break;
            }
        }
    }
    return no_capture;
}

void DammaState::swap_board_view(Board& board)
{
    int row_lo, row_hi;
    row_lo = 0;
    row_hi = ROWS - 1;
    while (row_lo < row_hi)
    {
        auto tmp = board.at(row_lo);
        board.at(row_lo) = board.at(row_hi);
        board.at(row_hi) = tmp;
        for (int col = 0; col < COLS; col++)
        {
            // TODO:  mb we can swap and inverse at the same time
            board.at(row_lo).at(col) = -board.at(row_lo).at(col);
            board.at(row_hi).at(col) = -board.at(row_hi).at(col);
        }
        row_lo++;
        row_hi--;
    }
}

bool DammaState::is_opponent_win() const
{
    // check if we have units or not then check if we have valid actions or not
    // if we do not have any units means we lost , and if we do not have valid actions means we lost too
    bool has_units = false;
    for (int row = 0; row < ROWS && !has_units; row++)
    {
        for (int col = 0; col < COLS && !has_units; col++)
        {
            if (board_.at(row).at(col) > 0)
            {
                has_units = true;
            }
        }
    }

    if (!has_units)
    {
        return true;
    }

    const auto mask = actions_mask();

    // check if player has any valid action
    for (int i{ 0 }; i < mask.size(); i++)
    {
        if (mask[i])
        {
            return false;
        }
    }

    // it does not have any valid move , player loses , opponent win
    return true;
}

bool DammaState::is_draw() const
{
    return n_no_capture_rounds_ == MAX_NO_CAPTURE_ROUNDS;
}

void DammaState::add_no_capture_rounds_observation_(std::vector<float>& observation_out) const
{
    constexpr int no_capture_rounds_obs_channel_id = 4;
    constexpr int channel_size = ROWS * COLS;
    constexpr int channel_start = no_capture_rounds_obs_channel_id * channel_size;
    constexpr int channel_end = channel_start + channel_size;
    for (int i = { channel_start }; i < channel_end; i++)
    {
        observation_out.at(i) = static_cast<float>(n_no_capture_rounds_) / MAX_NO_CAPTURE_ROUNDS;
    }
}

void DammaState::add_current_player_turn_observation_(std::vector<float>& observation_out) const
{
    constexpr int current_player_turn_channel_id{ 6 };
    constexpr int channel_size = ROWS * COLS;
    constexpr int channel_start = current_player_turn_channel_id * channel_size;
    constexpr int channel_end = channel_start + channel_size;

    int player = current_player_;
    for (int i{ channel_start }; i < channel_end; i++)
    {
        observation_out.at(i) = player;
    }
}

void DammaState::add_last_jump_observation_(std::vector<float>& observation_out) const
{
    if (!last_jump_.has_value())
    {
        return;
    }
    constexpr int last_jump_obs_channel_id = 5;
    constexpr int channel_size = ROWS * COLS;
    constexpr int channel_start = last_jump_obs_channel_id * channel_size;
    constexpr int channel_end = channel_start + channel_size;

    auto [row, col] = last_jump_.value();

    int cell_to_be_modified = channel_start + row * COLS + col;
    observation_out.at(cell_to_be_modified) = 1.0f;
}

} // namespace rl::games
