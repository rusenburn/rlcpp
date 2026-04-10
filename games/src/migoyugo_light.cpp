#include <sstream>
#include <stdexcept>
#include <iomanip>
#include <iostream>
#include <array>
#include <string>
#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <cctype>
#include <games/migoyugo_light.hpp>
#include <games/migoyugo.hpp>
#include <common/exceptions.hpp>


namespace rl::games
{


inline int ctz64(uint64_t x) {
#if defined(_MSC_VER)
    unsigned long idx;
    _BitScanForward64(&idx, x);
    return (int)idx;
#else
    return __builtin_ctzll(x);
#endif
}
MigoyugoLightState::MigoyugoLightState(std::array<std::array<int8_t, COLS>, ROWS> board, int player, int step, int last_action)
    : board_(board),
    current_player_(player),
    step_(step),
    last_action_(last_action)
{
}

MigoyugoLightState::~MigoyugoLightState() = default;

std::unique_ptr<MigoyugoLightState> MigoyugoLightState::initialize_state()
{
    std::array<std::array<int8_t, COLS>, ROWS> obs;

    for (int row = 0; row < ROWS; row++)
    {
        for (int col = 0; col < COLS; col++)
        {
            obs.at(row).at(col) = 0;
        }
    }

    int player_0 = 0;
    auto state_ptr = std::make_unique<MigoyugoLightState>(obs, player_0, 0, -1);
    return state_ptr;
}

std::unique_ptr<rl::common::IState> MigoyugoLightState::initialize()
{
    return initialize_state();
}




std::unique_ptr<MigoyugoLightState> MigoyugoLightState::from_short(const std::string& short_str) {
    // 1. Split board part from player part
    size_t space_pos = short_str.find_last_of(' ');
    std::string board_part = short_str.substr(0, space_pos);
    int player = std::stoi(short_str.substr(space_pos + 1));

    std::array<std::array<int8_t, COLS>, ROWS> new_board{};
    int r = 0, c = 0;

    // 2. Parse the FEN-style board string
    for (char ch : board_part) {
        if (ch == '/') {
            r++;
            c = 0;
            continue;
        }

        if (std::isdigit(ch)) {
            // It's an empty square count (e.g., '5')
            c += (ch - '0');
        }
        else {
            // It's a piece
            int8_t val = 0;
            switch (ch) {
            case 'x': val = 1;  break; // Migo
            case 'X': val = 2;  break; // Yugo
            case 'o': val = -1; break; // Opponent Migo
            case 'O': val = -2; break; // Opponent Yugo
            }

            if (r < ROWS && c < COLS) {
                new_board[r][c] = val;
            }
            c++;
        }
    }

    // 3. Create the light state (assuming step 0 and no last action)
    return std::make_unique<MigoyugoLightState>(new_board, player, 0, -1);
}
std::unique_ptr<rl::common::IState> MigoyugoLightState::reset() const
{
    return reset_state();
}
std::unique_ptr<MigoyugoLightState> MigoyugoLightState::reset_state() const
{
    return initialize_state();
}


std::unique_ptr<MigoyugoLightState> MigoyugoLightState::step_state(int action) const
{
    if (is_terminal())
    {
        throw rl::common::SteppingTerminalStateException("Trying to step a terminal state");
    }
    auto am = actions_mask_2();
    bool action_legality = am.at(action);
    if (action_legality == false)
    {
        std::stringstream ss;
        ss << "Trying to perform an illegal action of " << action;
        throw rl::common::IllegalActionException(ss.str());
    }

    int player = current_player_;
    int other = 1 - player;

    std::array<std::array<int8_t, COLS>, ROWS> new_board(board_);

    // turn action into (row,col) action
    int row_id = action / COLS;
    int col_id = action % COLS;

    std::array<std::pair<int8_t, int8_t>, 12> change_buffer;

    int change_count = 0;

    bool creates_yugo = false;

    constexpr std::array<std::array<std::pair<int, int>, 2>, 4> directions = { {
   {{ {0, 1}, {0, -1} }},
   {{ {1, 0}, {-1, 0} }},
   {{ {1, 1}, {-1, -1} }},
   {{ {1, -1}, {-1, 1} }}
        } };


    for (const auto& opposites : directions) // Use your constexpr directions array
    {
        int streak_len = 0;
        // Check both sides of the axis
        for (const auto& dir : opposites) {
            streak_len += get_streak_count(row_id, col_id, dir.first, dir.second, current_player_);
        }

        if (streak_len == 3) {
            creates_yugo = true;
            // Record these cells to clear them later
            for (const auto& dir : opposites) {
                int r = row_id + dir.first;
                int c = col_id + dir.second;
                // Walk the streak and add to buffer
                while (is_in_board(r, c) && (current_player_ == 0 ? board_[r][c] > 0 : board_[r][c] < 0)) {
                    change_buffer[change_count++] = { static_cast<int8_t>(r), static_cast<int8_t>(c) };
                    r += dir.first;
                    c += dir.second;
                }
            }
        }
    }
    if (creates_yugo) {
        new_board[row_id][col_id] = (current_player_ == 0) ? 2 : -2;
        // Clear the completed pieces
        for (int i = 0; i < change_count; ++i) {
            auto& cell = change_buffer[i];
            int8_t val = new_board[cell.first][cell.second];
            // Only clear if it's a Migo (1/-1), not a Yugo (2/-2)
            if (val == 1 || val == -1) {
                new_board[cell.first][cell.second] = 0;
            }
        }
    }
    else {
        new_board[row_id][col_id] = (current_player_ == 0) ? 1 : -1;
    }

    return std::make_unique<MigoyugoLightState>(new_board, other, step_ + 1, action);
}


std::unique_ptr<MigoyugoLightState> MigoyugoLightState::step_state_light(int action, NNUEUpdate& update_out) const
{
    if (is_terminal())
    {
        throw rl::common::SteppingTerminalStateException("Trying to step a terminal state");
    }
    auto am = actions_mask_2();
    bool action_legality = am.at(action);
    if (action_legality == false)
    {
        std::stringstream ss;
        ss << "Trying to perform an illegal action of " << action;
        throw rl::common::IllegalActionException(ss.str());
    }

    int player = current_player_;
    int other = 1 - player;

    std::array<std::array<int8_t, COLS>, ROWS> new_board(board_);

    // turn action into (row,col) action
    int row_id = action / COLS;
    int col_id = action % COLS;

    std::array<std::pair<int8_t, int8_t>, 12> change_buffer;

    int change_count = 0;

    bool creates_yugo = false;

    constexpr std::array<std::array<std::pair<int, int>, 2>, 4> directions = { {
   {{ {0, 1}, {0, -1} }},
   {{ {1, 0}, {-1, 0} }},
   {{ {1, 1}, {-1, -1} }},
   {{ {1, -1}, {-1, 1} }}
        } };


    for (const auto& opposites : directions) // Use your constexpr directions array
    {
        int streak_len = 0;
        // Check both sides of the axis
        for (const auto& dir : opposites) {
            streak_len += get_streak_count(row_id, col_id, dir.first, dir.second, current_player_);
        }

        if (streak_len == 3) {
            creates_yugo = true;
            // Record these cells to clear them later
            for (const auto& dir : opposites) {
                int r = row_id + dir.first;
                int c = col_id + dir.second;
                // Walk the streak and add to buffer
                while (is_in_board(r, c) && (current_player_ == 0 ? board_[r][c] > 0 : board_[r][c] < 0)) {
                    change_buffer[change_count++] = { static_cast<int8_t>(r), static_cast<int8_t>(c) };
                    r += dir.first;
                    c += dir.second;
                }
            }
        }
    }

    auto get_feature_id = [](int r, int c, int8_t piece, int perspective) {
        int square = r * 8 + c;
        bool is_white = (piece > 0);
        bool is_yugo = (std::abs(piece) == 2);

        // Map piece to one of the 4 channels (64 features each)
        // Perspective 0: White is "Current Player" (Channels 0,1)
        // Perspective 1: Black is "Current Player" (Channels 0,1)
        int channel;
        if (perspective == 0) { // White's View
            if (is_white) channel = is_yugo ? 1 : 0; // My Migo/Yugo
            else          channel = is_yugo ? 3 : 2; // Enemy Migo/Yugo
        }
        else { // Black's View
            if (!is_white) channel = is_yugo ? 1 : 0; // My Migo/Yugo
            else           channel = is_yugo ? 3 : 2; // Enemy Migo/Yugo
        }
        return (channel * 64) + square;
        };

    if (creates_yugo) {
        int8_t yugo_piece = (current_player_ == 0) ? 2 : -2;
        new_board[row_id][col_id] = yugo_piece;

        // 1. Record the Yugo being ADDED
        // Perspective 0 (White) and 1 (Black)
        update_out.white_added.push_back(get_feature_id(row_id, col_id, yugo_piece, 0));
        update_out.black_added.push_back(get_feature_id(row_id, col_id, yugo_piece, 1));

        // Clear the completed pieces
        for (int i = 0; i < change_count; ++i) {
            auto& cell = change_buffer[i];
            int8_t val = new_board[cell.first][cell.second];

            // Only clear if it's a Migo (1/-1), not a Yugo (2/-2)
            if (val == 1 || val == -1) {
                // 2. Record the Migo being REMOVED before we set the board to 0
                update_out.white_removed.push_back(get_feature_id(cell.first, cell.second, val, 0));
                update_out.black_removed.push_back(get_feature_id(cell.first, cell.second, val, 1));

                new_board[cell.first][cell.second] = 0;
            }
        }
    }
    else {
        int8_t migo_piece = (current_player_ == 0) ? 1 : -1;
        new_board[row_id][col_id] = migo_piece;

        // 3. Record the Migo being ADDED (Standard Move)
        update_out.white_added.push_back(get_feature_id(row_id, col_id, migo_piece, 0));
        update_out.black_added.push_back(get_feature_id(row_id, col_id, migo_piece, 1));
    }

    return std::make_unique<MigoyugoLightState>(new_board, other, step_ + 1, action);
}
void MigoyugoLightState::render() const
{
    auto legal_actions = actions_mask();
    std::array<std::array<bool, COLS>, ROWS> legal_actions_2d{};
    for (int row = 0; row < ROWS; row++)
    {
        for (int col = 0; col < COLS; col++)
        {
            int action = row * COLS + col;
            legal_actions_2d.at(row).at(col) = legal_actions.at(action);
        }
    }

    std::cout << "\n";
    std::cout << "   0  1  2  3  4  5  6  7\n";

    for (int row = 0; row < ROWS; row++)
    {
        std::cout << std::setw(3) << std::setfill(' ') << row;
        std::cout << ' ';
        for (int col = 0; col < COLS; col++)
        {
            int cell = board_.at(row).at(col);
            std::string display = ".";

            if (cell == 1)
            {
                display = "x";
            }
            else if (cell == 2)
            {
                display = "X";
            }
            else if (cell == -1)
            {
                display = "o";
            }
            else if (cell == -2)
            {
                display = "O";
            }
            else if (legal_actions_2d.at(row).at(col))
            {
                display = std::to_string(row * COLS + col);
            }

            std::cout << std::setw(3) << std::setfill(' ') << display;
        }
        std::cout << '\n';
    }
    char player_char = 'x';
    if (current_player_ == 1)
    {
        player_char = 'o';
    }

    std::cout << "\nPlayer " << player_char << " to move" << std::endl;
}
std::unique_ptr<rl::common::IState> MigoyugoLightState::step(int action) const
{
    return step_state(action);
}

int MigoyugoLightState::get_n_actions()const
{
    return ROWS * COLS;
}

int MigoyugoLightState::player_turn() const
{
    return current_player_;
}

std::array<int, 3> MigoyugoLightState::get_observation_shape() const
{
    return { CHANNELS, ROWS, COLS };
}




bool MigoyugoLightState::is_in_board(int row, int col) const
{
    return row < ROWS && row >= 0 && col < COLS && col >= 0;
}


bool MigoyugoLightState::is_terminal() const {
    if (cached_is_terminal_.has_value())
    {
        return cached_is_terminal_.value();
    }

    if (is_opponent_won())
    {
        cached_is_terminal_.emplace(true);
        return true;
    }

    // check if there is no legal action
    if (has_legal_action())
    {
        cached_is_terminal_.emplace(false);
        return false;
    }

    // has no legal action
    cached_is_terminal_.emplace(true);
    return true;

}
bool MigoyugoLightState::has_legal_action()const {
    auto am = actions_mask_2();
    for (auto is_legal : am)
    {
        if (is_legal)
        {
            return true;
        }
    }
    return false;
}


bool MigoyugoLightState::is_opponent_won() const
{
    const int opponent = 1 - current_player_;
    const int8_t opponent_yugo_flag = (opponent == 0) ? 2 : -2;

    // --- FAST PATH: For 99.9% of the search tree nodes ---
    if (last_action_ >= 0) {
        int row_id = last_action_ / COLS;
        int col_id = last_action_ % COLS;

        // If the piece just placed isn't an opponent's Yugo, they didn't win by streak.
        if (board_[row_id][col_id] != opponent_yugo_flag) return false;

        constexpr std::array<std::array<std::pair<int, int>, 2>, 4> directions = { {
   {{ {0, 1}, {0, -1} }},
   {{ {1, 0}, {-1, 0} }},
   {{ {1, 1}, {-1, -1} }},
   {{ {1, -1}, {-1, 1} }}
        } };

        for (const auto& axis : directions) {
            int streak_len = 1; // Start with the last_action piece
            for (const auto& dir : axis) {
                streak_len += get_yugo_streak_count(row_id, col_id, dir.first, dir.second, opponent);
            }
            if (streak_len >= 4) return true;
        }
        return false;
    }

    // --- SLOW PATH: Only for Root Node (from_short) ---
    // If last_action is -1, we must scan the board once to see if the state is already terminal.
    for (int r = 0; r < ROWS; ++r) {
        for (int c = 0; c < COLS; ++c) {
            if (board_[r][c] == opponent_yugo_flag) {
                // Check 4 directions (only forward-looking to avoid double counting)
                static constexpr int dir_row[] = { 0, 1, 1, 1 };
                static constexpr int dir_col[] = { 1, 0, 1, -1 };

                for (int i = 0; i < 4; ++i) {
                    int count = 1;
                    for (int step = 1; step < 4; ++step) {
                        int new_row = r + dir_row[i] * step;
                        int new_col = c + dir_col[i] * step;
                        if (is_in_board(new_row, new_col) && board_[new_row][new_col] == opponent_yugo_flag) {
                            count++;
                        }
                        else {
                            break;
                        }
                    }
                    if (count >= 4) return true;
                }
            }
        }
    }
    return false;
}


bool MigoyugoLightState::check_row_winning(std::array<std::array<int8_t, COLS>, ROWS> const& opponent_yugos_board, int row, int col)const
{
    if (!is_in_board(row + 3, col))
    {
        return false;
    };

    if (
        // opponent_yugos_board.at(row).at(col) &&
        opponent_yugos_board[row + 1][col] &&
        opponent_yugos_board[row + 2][col] &&
        opponent_yugos_board[row + 3][col]
        )
    {
        return true;
    }

    return false;
}


bool MigoyugoLightState::check_col_winning(std::array<std::array<int8_t, COLS>, ROWS> const& opponent_yugos_board, int row, int col)const
{
    if (!is_in_board(row, col + 3))
    {
        return false;
    };

    if (
        // opponent_yugos_board.at(row).at(col) &&
        opponent_yugos_board[row][col + 1] &&
        opponent_yugos_board[row][col + 2] &&
        opponent_yugos_board[row][col + 3]
        )
    {
        return true;
    }

    return false;
}

bool MigoyugoLightState::check_forward_diagonal_winning(std::array<std::array<int8_t, COLS>, ROWS> const& opponent_yugos_board, int row, int col)const
{
    if (!is_in_board(row + 3, col + 3))
    {
        return false;
    };

    if (
        // opponent_yugos_board.at(row).at(col) &&
        opponent_yugos_board[row + 1][col + 1] &&
        opponent_yugos_board[row + 2][col + 2] &&
        opponent_yugos_board[row + 3][col + 3]
        )
    {
        return true;
    }
    return false;
}


bool MigoyugoLightState::check_backward_diagonal_winning(std::array<std::array<int8_t, COLS>, ROWS> const& opponent_yugos_board, int row, int col)const
{
    if (!is_in_board(row + 3, col - 3))
    {
        return false;
    };

    if (
        // opponent_yugos_board.at(row).at(col) &&
        opponent_yugos_board[row + 1][col - 1] &&
        opponent_yugos_board[row + 2][col - 2] &&
        opponent_yugos_board[row + 3][col - 3]
        )
    {
        return true;
    }
    return false;
}


std::vector<bool> MigoyugoLightState::actions_mask() const
{
    if (!cached_actions_masks_.empty()) return cached_actions_masks_;

    std::vector<bool> result(get_n_actions(), true);

    constexpr std::array<std::array<std::pair<int, int>, 2>, 4> directions = { {
   {{ {0, 1}, {0, -1} }},
   {{ {1, 0}, {-1, 0} }},
   {{ {1, 1}, {-1, -1} }},
   {{ {1, -1}, {-1, 1} }}
        } };

    for (int row = 0; row < ROWS; row++) {
        for (int col = 0; col < COLS; col++) {
            int action = row * COLS + col;

            if (board_[row][col] != 0) {
                result[action] = false;
                continue;
            }

            // Check all 4 axes using the unified helper
            for (const auto& axis : directions) {
                int total_streak = 0;
                for (const auto& dir : axis) {
                    total_streak += get_streak_count(row, col, dir.first, dir.second, current_player_);
                }

                // Rule: Line of 5 or more (3 existing + 1 placed + 1 more) is illegal
                if (total_streak >= 4) {
                    result[action] = false;
                    break;
                }
            }
        }
    }
    cached_actions_masks_ = result;
    return cached_actions_masks_;
}

std::vector<int> MigoyugoLightState::actions_mask_2() const
{
    if (!cached_actions_masks_2_.empty()) return cached_actions_masks_2_;

    std::vector<int> result(get_n_actions(), true);

    constexpr std::array<std::array<std::pair<int, int>, 2>, 4> directions = { {
   {{ {0, 1}, {0, -1} }},
   {{ {1, 0}, {-1, 0} }},
   {{ {1, 1}, {-1, -1} }},
   {{ {1, -1}, {-1, 1} }}
        } };

    for (int row = 0; row < ROWS; row++) {
        for (int col = 0; col < COLS; col++) {
            int action = row * COLS + col;

            if (board_[row][col] != 0) {
                result[action] = false;
                continue;
            }

            // Check all 4 axes using the unified helper
            for (const auto& axis : directions) {
                int total_streak = 0;
                for (const auto& dir : axis) {
                    total_streak += get_streak_count(row, col, dir.first, dir.second, current_player_);
                }

                // Rule: Line of 5 or more (3 existing + 1 placed + 1 more) is illegal
                if (total_streak >= 4) {
                    result[action] = false;
                    break;
                }
            }
        }
    }
    cached_actions_masks_2_ = result;
    return result;
}

// This is the only streak helper you need in the class
int MigoyugoLightState::get_streak_count(int row, int col, int row_dir, int col_dir, int player) const
{
    int count = 0;
    row += row_dir;
    col += col_dir;

    while (is_in_board(row, col))
    {
        int8_t cell = board_[row][col]; // No .at() for speed

        if (player == 0) {
            if (cell <= 0) break; // Not a player 0 piece (1 or 2)
        }
        else {
            if (cell >= 0) break; // Not a player 1 piece (-1 or -2)
        }

        count++;
        row += row_dir;
        col += col_dir;
    }
    return count;
}


int MigoyugoLightState::get_yugo_streak_count(int row, int col, int row_dir, int col_dir, int player) const
{
    int count = 0;
    row += row_dir;
    col += col_dir;
    while (is_in_board(row, col))
    {
        int8_t cell = board_[row][col]; // No .at() for speed

        if (player == 0) {
            if (cell != 2) break; // Not a player 0 yugo (2)
        }
        else {
            if (cell != -2) break; // Not a player 1 yugo (-2)
        }

        count++;
        row += row_dir;
        col += col_dir;
    }
    return count;
}


float MigoyugoLightState::get_reward() const
{
    if (!is_terminal())
    {
        return 0.0f;
    }
    if (cached_result_.has_value())
    {
        return cached_result_.value();
    }

    if (is_opponent_won())
    {
        cached_result_.emplace(-1.0f);
        return cached_result_.value();
    }

    if (has_legal_action())
    {
        std::stringstream ss;
        ss << "Reaching unreachable code after getting rewards";
        throw rl::common::UnreachableCodeException(ss.str());
    }

    int player = current_player_;
    int opponent = 1 - player;

    int player_0_yugos = 0;
    int player_1_yugos = 0;
    int cell = 0;
    for (size_t row = 0; row < ROWS; row++)
    {
        for (size_t col = 0; col < COLS; col++)
        {
            cell = board_[row][col];
            if (cell > 1)
            {
                player_0_yugos++;
            }
            else if (cell < -1)
            {
                player_1_yugos++;
            }
        }
    }

    float player_0_score = 0.0f;

    if (player_0_yugos > player_1_yugos)
    {
        player_0_score = 1.0f;
    }
    else if (player_0_yugos < player_1_yugos)
    {
        player_0_score = -1.0f;
    }
    else {
        player_0_score = 0.0f;
    }

    float player_score = player == 0 ? player_0_score : -player_0_score;

    cached_result_.emplace(player_score);
    return cached_result_.value();
}

int MigoyugoLightState::encode_action(int row, int col)
{
    int action = row * COLS + col;
    return action;
}


std::unique_ptr<MigoyugoLightState> MigoyugoLightState::clone_state() const
{
    return std::unique_ptr<MigoyugoLightState>(new MigoyugoLightState(*this));
}

std::unique_ptr<rl::common::IState> MigoyugoLightState::clone() const
{
    return clone_state();
}

std::string MigoyugoLightState::to_short() const
{
    if (cached_short_.has_value())
    {
        return cached_short_.value();
    }

    std::stringstream ss;
    for (int row = 0; row < ROWS; row++)
    {
        int empty_count = 0;
        for (int col = 0; col < COLS; col++)
        {
            int cell = board_.at(row).at(col);
            if (cell == 0)
            {
                empty_count++;
            }
            else
            {
                if (empty_count > 0)
                {
                    ss << empty_count;
                    empty_count = 0;
                }
                if (cell == 1)
                {
                    ss << 'x';
                }
                else if (cell == 2)
                {
                    ss << 'X';
                }
                else if (cell == -1)
                {
                    ss << 'o';
                }
                else if (cell == -2)
                {
                    ss << 'O';
                }
            }
        }
        if (empty_count > 0)
        {
            ss << empty_count;
        }
        if (row < ROWS - 1)
        {
            ss << '/';
        }
    }
    ss << ' ' << current_player_;
    cached_short_.emplace(ss.str());
    return cached_short_.value();
}

void MigoyugoLightState::get_symmetrical_obs_and_actions(
    std::vector<float> const& obs,
    std::vector<float> const& actions_distribution,
    std::vector<std::vector<float>>& out_syms,
    std::vector<std::vector<float>>& out_actions_distribution) const
{
    const size_t obs_size = CHANNELS * ROWS * COLS;
    if (obs.size() != obs_size)
    {
        std::stringstream ss;
        ss << "get_symmetrical_obs_and_actions requires an observation with size of " << obs_size;
        ss << " but a size of " << obs.size() << " was passed.";
        throw std::runtime_error(ss.str());
    }

    out_syms.clear();
    out_actions_distribution.clear();
    // We expect 7 symmetries
    out_syms.reserve(7);
    out_actions_distribution.reserve(7);

    // Helper lambda to map a vector using a symmetry array
    auto apply_sym = [](const std::vector<float>& source, const auto& mapping_array) {
        std::vector<float> result(mapping_array.size());
        for (size_t i = 0; i < mapping_array.size(); ++i) {
            // Using .at() on both for maximum safety as requested
            result.at(i) = source.at(mapping_array.at(i));
        }
        return result;
        };

    // Define a helper macro or local function to bundle the additions
    auto add_symmetry = [&](const auto& obs_map, const auto& act_map) {
        out_syms.push_back(apply_sym(obs, obs_map));
        out_actions_distribution.push_back(apply_sym(actions_distribution, act_map));
        };

    // --- Apply all 7 symmetries ---
    using namespace miguyugo_syms;

    add_symmetry(ROT90_OBS_SYM, ROT90_ACTIONS_SYM);
    add_symmetry(ROT180_OBS_SYM, ROT180_ACTIONS_SYM);
    add_symmetry(ROT270_OBS_SYM, ROT270_ACTIONS_SYM);
    add_symmetry(FLIP_LR_OBS_SYM, FLIP_LR_ACTIONS_SYM);
    add_symmetry(FLIP_UD_OBS_SYM, FLIP_UD_ACTIONS_SYM);
    add_symmetry(TRANSPOSE_OBS_SYM, TRANSPOSE_ACTIONS_SYM);
    add_symmetry(ANTI_TRANSPOSE_OBS_SYM, ANTI_TRANSPOSE_ACTIONS_SYM);
}

std::vector<float> MigoyugoLightState::get_observation() const
{
    if (cached_observation_.size())
    {
        return cached_observation_;
    }

    int player = current_player_;
    std::vector<float> true_obs;

    true_obs.resize(CHANNELS * ROWS * COLS, 0);
    int index = 0;

    for (size_t channel = 0; channel < CHANNELS; channel++)
    {
        for (int row{ 0 }; row < ROWS; row++)
        {
            for (int col{ 0 }; col < COLS; col++)
            {
                int channel_flag = 100;
                if (player == 0)
                {
                    if (channel == 0)
                    {
                        channel_flag = 1;
                    }
                    else if (channel == 1)
                    {
                        channel_flag = 2;
                    }
                    else if (channel == 2)
                    {
                        channel_flag = -1;
                    }
                    else if (channel == 3)
                    {
                        channel_flag = -2;
                    }
                }
                else // player == 1
                {
                    if (channel == 0)
                    {
                        channel_flag = -1;
                    }
                    else if (channel == 1)
                    {
                        channel_flag = -2;
                    }
                    else if (channel == 2)
                    {
                        channel_flag = 1;
                    }
                    else if (channel == 3)
                    {
                        channel_flag = 2;
                    }
                }


                if (board_.at(row).at(col) == channel_flag)
                {
                    true_obs.at(index) = 1.0;
                }
                index++;
            }
        }
    }

    cached_observation_ = true_obs;
    return true_obs;
}


void MigoyugoLightState::get_active_features(NNUEUpdate& update_out) const
{
    update_out.white_added.clear();
    update_out.black_added.clear();

    auto get_feature_id = [](int r, int c, int8_t piece, int perspective) {
        int square = r * 8 + c;
        bool is_white = (piece > 0);
        bool is_yugo = (std::abs(piece) == 2);

        int channel;
        if (perspective == 0) { // White's Perspective
            if (is_white) channel = is_yugo ? OUR_YUGO_CHANNEL : OUR_MIGO_CHANNEL; // My Migo/Yugo
            else          channel = is_yugo ? OPPONENT_YUGO_CHANNEL : OPPONENT_MIGO_CHANNEL;// Enemy Migo/Yugo
        }
        else { // Black's Perspective
            if (!is_white) channel = is_yugo ? OUR_YUGO_CHANNEL : OUR_MIGO_CHANNEL; // My Migo/Yugo
            else           channel = is_yugo ? OPPONENT_YUGO_CHANNEL : OPPONENT_MIGO_CHANNEL; // Enemy Migo/Yugo
        }
        return (channel * ROWS * COLS) + square;
        };

    // 2. Scan the physical board
    for (int r = 0; r < ROWS; ++r) {
        for (int c = 0; c < COLS; ++c) {
            int8_t piece = board_[r][c];
            if (piece != 0) {
                // Add this piece's feature ID to the White view
                update_out.white_added.push_back(get_feature_id(r, c, piece, 0));

                // Add this piece's feature ID to the Black view
                update_out.black_added.push_back(get_feature_id(r, c, piece, 1));
            }
        }
    }
}

std::string MigoyugoLightState::to_string_key() const {
    // 1. Bitboards for: [MyMigo, TheirMigo, MyYugo, TheirYugo]
    std::array<uint64_t, 4> boards = { 0, 0, 0, 0 };

    for (int r = 0; r < ROWS; ++r) {
        for (int c = 0; c < COLS; ++c) {
            int8_t piece = board_[r][c];
            if (piece == 0) continue;

            int bit_idx = r * COLS + c;
            uint64_t bit = 1ULL << bit_idx;

            // Perspective Check: Is this piece "Mine" relative to current_player_?
            bool is_mine = (current_player_ == 0) ? (piece > 0) : (piece < 0);
            bool is_yugo = (std::abs(piece) == 2);

            // Channel Mapping:
            // 0: My Migo, 1: Their Migo, 2: My Yugo, 3: Their Yugo
            int channel = is_yugo ? (is_mine ? 2 : 3) : (is_mine ? 0 : 1);
            boards[channel] |= bit;
        }
    }

    return std::string(reinterpret_cast<const char*>(boards.data()), 32);
}


int MigoyugoLightState::get_last_action()const {
    return last_action_;
}


float MigoyugoLightState::calculate_feature_weight() const {
    float total = 0;
    for (int r = 0; r < ROWS; ++r) {
        for (int c = 0; c < COLS; ++c) {
            int8_t val = board_[r][c];
            if (val == 1 || val == -1) total += 1.0f; // Migo 1 Move
            else if (val == 2 || val == -2) total += 4.0f; // Yugo 4 Moves
        }
    }
    return total;
}

std::vector<int> MigoyugoLightState::detect_threats() const
{
    auto am = actions_mask_2();

    uint64_t white_migo = 0;
    uint64_t white_yugo = 0;
    uint64_t black_migo = 0;
    uint64_t black_yugo = 0;

    uint64_t legal_moves = 0;

    // Build bitboards
    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            int v = board_[r][c];
            int idx = r * 8 + c;

            if (v == 1)  white_migo |= (1ULL << idx);
            if (v == 2)  white_yugo |= (1ULL << idx);
            if (v == -1) black_migo |= (1ULL << idx);
            if (v == -2) black_yugo |= (1ULL << idx);
        }
    }

    // Legal moves bitboard
    for (int idx = 0; idx < 64; idx++) {
        if (am[idx]) {
            legal_moves |= (1ULL << idx);
        }
    }

    uint64_t occupied = white_migo | white_yugo | black_migo | black_yugo;
    uint64_t empty = ~occupied; // fine for 64-bit

    // Threat calculations
    uint64_t white_yugo_moves = find_extend_3_to_4(white_yugo, empty);
    uint64_t black_yugo_moves = find_extend_3_to_4(black_yugo, empty);

    uint64_t white_all = white_migo | white_yugo;
    uint64_t black_all = black_migo | black_yugo;

    uint64_t white_combo_moves = find_extend_3_to_4(white_all, empty);
    uint64_t black_combo_moves = find_extend_3_to_4(black_all, empty);

    std::vector<int> priority_actions;

    if (current_player_ == 0)
    {
        uint64_t win_threats = white_yugo_moves & legal_moves;

        uint64_t loss_threats = (black_yugo_moves & legal_moves) & ~win_threats;

        uint64_t win_tempo = (white_combo_moves & legal_moves)
            & ~(win_threats | loss_threats);

        uint64_t loss_tempo = (black_combo_moves & legal_moves)
            & ~(win_threats | loss_threats | win_tempo);

        // Convert bitboards → vector
        auto extract = [&](uint64_t bb) {
            while (bb) {
                int idx = ctz64(bb);
                priority_actions.push_back(idx);
                bb &= bb - 1;
            }
            };

        extract(win_threats);
        extract(loss_threats);
        extract(win_tempo);
        extract(loss_tempo);
    }
    else
    {
        // Same logic but swapped roles
        uint64_t win_threats = black_yugo_moves & legal_moves;

        uint64_t loss_threats = (white_yugo_moves & legal_moves) & ~win_threats;

        uint64_t win_tempo = (black_combo_moves & legal_moves)
            & ~(win_threats | loss_threats);

        uint64_t loss_tempo = (white_combo_moves & legal_moves)
            & ~(win_threats | loss_threats | win_tempo);

        auto extract = [&](uint64_t bb) {
            while (bb) {
                int idx = ctz64(bb);
                priority_actions.push_back(idx);
                bb &= bb - 1;
            }
            };

        extract(win_threats);
        extract(loss_threats);
        extract(win_tempo);
        extract(loss_tempo);
    }

    return priority_actions;
}

uint64_t MigoyugoLightState::find_extend_3_to_4(uint64_t b, uint64_t empty) {
    constexpr uint64_t notA = 0xfefefefefefefefeULL;
    constexpr uint64_t notH = 0x7f7f7f7f7f7f7f7fULL;

    uint64_t result = 0;

    // Horizontal
    {
        uint64_t three = b & ((b & notH) >> 1) & ((b & notH & (notH >> 1)) >> 2);
        result |= ((three & notH) >> 3) | ((three & notA) << 1);
    }

    // Vertical
    {
        uint64_t three = b & (b >> 8) & (b >> 16);
        result |= (three >> 24) | (three << 8);
    }

    // Diagonal ↘
    {
        uint64_t three = b & ((b & notH) >> 9) & ((b & notH & (notH >> 1)) >> 18);
        result |= ((three & notH) >> 27) | ((three & notA) << 9);
    }

    // Diagonal ↙
    {
        uint64_t three = b & ((b & notA) >> 7) & ((b & notA & (notA << 1)) >> 14);
        result |= ((three & notA) >> 21) | ((three & notH) << 7);
    }

    return result & empty;
}
}


