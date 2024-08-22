#include <sstream>
#include <cassert>
#include <iostream>

#include <games/english_draughts.hpp>
#include <common/exceptions.hpp>

namespace rl::games
{
EnglishDraughtState::EnglishDraughtState(Board board,
    int n_no_capture_rounds,
    std::vector<bool> last_jump_actions_mask,
    std::vector<int> last_jump,
    int current_player)
    : board_(board),
    n_no_capture_rounds_{ n_no_capture_rounds },
    last_jump_actions_mask_{ last_jump_actions_mask },
    last_jump_{ last_jump },
    current_player_{ current_player }
{
}

EnglishDraughtState::~EnglishDraughtState()
{
}

std::unique_ptr<EnglishDraughtState> EnglishDraughtState::initialize_state()
{
    std::array<std::array<int8_t, COLS>, ROWS> board{
        {{0, -1, 0, -1, 0, -1, 0, -1},
         {-1, 0, -1, 0, -1, 0, -1, 0},
         {0, -1, 0, -1, 0, -1, 0, -1},
         {0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0},
         {1, 0, 1, 0, 1, 0, 1, 0},
         {0, 1, 0, 1, 0, 1, 0, 1},
         {1, 0, 1, 0, 1, 0, 1, 0}} };
    return std::make_unique<EnglishDraughtState>(board, 0, std::vector<bool>{}, std::vector<int>{}, 0);
}

std::unique_ptr<rl::common::IState> EnglishDraughtState::initialize()
{
    return initialize_state();
}
std::unique_ptr<EnglishDraughtState> EnglishDraughtState::reset_state() const
{
    return initialize_state();
}

std::unique_ptr<rl::common::IState> EnglishDraughtState::reset() const
{
    return reset_state();
}

std::unique_ptr<EnglishDraughtState> EnglishDraughtState::step_state(int action) const
{
    if (is_terminal())
    {
        std::stringstream ss;
        ss << "Stepping a terminal EnglishDraught state";
        throw rl::common::SteppingTerminalStateException(ss.str());
    }
    if (actions_mask().at(action) == false)
    {
        std::stringstream ss;
        ss << "Stepping an EnglishDraught state with an illegal action " << action;
        throw rl::common::IllegalActionException(ss.str());
    }

    RowColDirectionAction rcd_action = get_row_col_direction_from_action_(action);
    int target_row = rcd_action.row + rcd_action.row_direction;
    int target_col = rcd_action.col + rcd_action.col_direction;
    Board new_board{ board_ };
    int new_n_no_capture_round{ n_no_capture_rounds_ };
    bool is_moving_piece_a_king = is_king_(new_board, rcd_action.row, rcd_action.col);
    int current_player_flag{ is_moving_piece_a_king ? PLAYERS_K_FLAGS.at(current_player_) : PLAYERS_P_FLAGS.at(current_player_) };
    bool is_player_switching{ true };
    int next_player{ 1 - current_player_ };
    std::vector<int> new_last_jump{};
    std::vector<bool> new_last_jump_actions_mask{};

    if (is_empty_position_(new_board, target_row, target_col))
    {
        // remove the piece
        new_board.at(rcd_action.row).at(rcd_action.col) = 0;

        if (target_row == 0 || target_row == ROWS - 1)
        {
            // add a king for the current player (promote moving piece)
            new_board.at(target_row).at(target_col) = PLAYERS_K_FLAGS.at(current_player_);
        }
        else
        {
            // add a piece for the current player without changing its rank
            new_board.at(target_row).at(target_col) = current_player_flag;
        }
        new_n_no_capture_round++;
    }
    else // not empty then we have an opponent piece , capture and jump
    {
        // remove out piece
        new_board.at(rcd_action.row).at(rcd_action.col) = 0;

        // remove opponent piece
        new_board.at(target_row).at(target_col) = 0;

        int jumpto_row = target_row + rcd_action.row_direction;
        int jumpto_col = target_col + rcd_action.col_direction;

        // check if our piece distination is promotion distination
        if (jumpto_row == 0 || jumpto_row == ROWS - 1)
        {
            // add a king for the current player (promote moving piece)
            new_board.at(jumpto_row).at(jumpto_col) = PLAYERS_K_FLAGS.at(current_player_);
        }
        else
        {
            // add a piece for the current player without changing its rank
            new_board.at(jumpto_row).at(jumpto_col) = current_player_flag;
        }
        new_n_no_capture_round = 0;

        // check if it can multi jump , DO NOT DOUBLE JUMP JUST CHECK IF IT IS POSSIBLE

        new_last_jump_actions_mask = std::vector<bool>(N_ACTIONS, false);

        bool can_multi_jump{ false };

        for (int direction_id{ 0 }; direction_id < DIRECTIONS.size(); direction_id++)
        {
            const std::array<int, 2>& d = DIRECTIONS.at(direction_id);
            // disable 180 multi jump ,, if we do not disable it the king can perform 180degree jump which is illegal
            if (d.at(0) == -rcd_action.row && d.at(1) == -rcd_action.col)
            {
                continue;
            }

            // jumpto is our new starting position in multi jump

            // each direction is an action , check if the action performed from new position can capture an opponent piece
            // only capture moves are allowed in multi jump
            int jump_action = encode_action_(jumpto_row, jumpto_col, direction_id);

            std::tuple<bool, bool> is_legal_is_capture = is_legal_is_capture_(new_board, jump_action, current_player_);
            const bool& is_capture = std::get<1>(is_legal_is_capture);
            if (is_capture)
            {
                new_last_jump_actions_mask.at(jump_action) = true;
                can_multi_jump = true;
                next_player = current_player_;
                is_player_switching = false;
                new_last_jump = std::vector<int>{ {jumpto_row, jumpto_col} };
            }
        }

        if (can_multi_jump == false)
        {
            new_last_jump_actions_mask = {};
            new_last_jump = {};
        }
    }
    return std::make_unique<EnglishDraughtState>(new_board, new_n_no_capture_round, new_last_jump_actions_mask, new_last_jump, next_player);
}

std::unique_ptr<rl::common::IState> EnglishDraughtState::step(int action) const
{
    return step_state(action);
}

void EnglishDraughtState::render() const
{
    std::stringstream ss;

    int player_0_pawn_flag = PLAYERS_P_FLAGS.at(0);
    int player_0_king_flag = PLAYERS_K_FLAGS.at(0);
    int player_1_pawn_flag = PLAYERS_P_FLAGS.at(1);
    int player_1_king_flag = PLAYERS_K_FLAGS.at(1);
    for (int row = 0; row < ROWS; row++)
    {
        for (int col{ 0 }; col < COLS; col++)
        {
            int cell = board_.at(row).at(col);
            if (cell == 0)
            {
                ss << " . ";
            }
            else if (cell == player_0_pawn_flag)
            {
                ss << " x ";
            }
            else if (cell == player_0_king_flag)
            {
                ss << " X ";
            }
            else if (cell == player_1_pawn_flag)
            {
                ss << " o ";
            }
            else if (cell == player_1_king_flag)
            {
                ss << " O ";
            }
        }
        ss << "\n";
    }

    if (current_player_ == 0)
    {
        ss << "Player x has to move";
    }
    if (current_player_ == 1)
    {
        ss << "Player o has to move";
    }
    std::cout << ss.str() << std::endl;
}

bool EnglishDraughtState::is_terminal() const
{
    if (cached_is_terminal_.has_value())
    {
        return cached_is_terminal_.value();
    }

    if (is_opponent_win_())
    {
        cached_is_terminal_.emplace(true);
        return cached_is_terminal_.value();
    }
    if (is_draw_())
    {
        cached_is_terminal_.emplace(true);
        return cached_is_terminal_.value();
    }
    cached_is_terminal_.emplace(false);
    return cached_is_terminal_.value();
}

float EnglishDraughtState::get_reward() const
{
    if (!is_terminal())
    {
        return 0.0f;
    }

    if (cached_result_.has_value())
    {
        return cached_result_.value();
    }

    if (is_opponent_win_())
    {
        cached_result_.emplace(-1.0f);
        return cached_result_.value();
    }
    if (is_draw_())
    {
        cached_result_.emplace(0.0f);
        return cached_result_.value();
    }
    throw rl::common::UnreachableCodeException("EnglishDraughtState get result unreachable error");
    cached_result_.emplace(1.0f);
    return cached_result_.value();
}

std::string EnglishDraughtState::to_short() const
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
                if (board_.at(row).at(col) == PLAYERS_K_FLAGS.at(0))
                {
                    ss << 'X';
                }
                else if (board_.at(row).at(col) == PLAYERS_P_FLAGS.at(0))
                {
                    ss << 'x';
                }
                else if (board_.at(row).at(col) == PLAYERS_K_FLAGS.at(1))
                {
                    ss << 'O';
                }
                else if (board_.at(row).at(col) == PLAYERS_P_FLAGS.at(1))
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
    if (last_jump_.size())
    {
        int row = last_jump_.at(0);
        int col = last_jump_.at(1);
        ss << "#" << row << "," << col;
    }
    return ss.str();
}

std::vector<float> EnglishDraughtState::get_observation() const
{
    if (cached_observation_.size())
    {
        return cached_observation_;
    }
    // NOTE : obviously not thread safe

    std::vector<float> cached_observation_ = std::vector<float>(CHANNELS * ROWS * COLS);
    constexpr int player_0_pawn = std::get<0>(PLAYERS_P_FLAGS);
    constexpr int player_0_king = std::get<0>(PLAYERS_K_FLAGS);
    constexpr int player_1_pawn = std::get<1>(PLAYERS_P_FLAGS);
    constexpr int player_1_king = std::get<1>(PLAYERS_K_FLAGS);

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
    add_current_player_turn_observation_(cached_observation_);

    return cached_observation_;
}

std::array<int, 3> EnglishDraughtState::get_observation_shape() const
{
    return { CHANNELS, ROWS, COLS };
}

int EnglishDraughtState::get_n_actions() const
{
    return N_ACTIONS;
}

int EnglishDraughtState::player_turn() const
{
    return current_player_;
}

std::vector<bool> EnglishDraughtState::actions_mask() const
{
    if (cached_actions_masks_.size())
    {
        return cached_actions_masks_;
    }

    if (last_jump_actions_mask_.size())
    {
        if (last_jump_.size() == 0)
        {
            throw rl::common::UnreachableCodeException("multi jump did not provide the last jump");
        }
        cached_actions_masks_ = last_jump_actions_mask_;
        return cached_actions_masks_;
    }

    assert(last_jump_.size() == 0);

    std::vector<bool> no_capture_actions_mask(N_ACTIONS);
    std::vector<bool> capture_actions_mask(N_ACTIONS);
    bool can_capture = false;

    for (int action{ 0 }; action < N_ACTIONS; action++)
    {
        std::tuple<bool, bool> is_legal_is_capture = is_legal_is_capture_(board_, action, current_player_);
        const bool& is_legal_action = std::get<0>(is_legal_is_capture);
        const bool& is_capture_action = std::get<1>(is_legal_is_capture);
        if (is_legal_action)
        {
            no_capture_actions_mask.at(action) = true;
        }
        if (is_capture_action)
        {
            capture_actions_mask.at(action) = true;
            can_capture = true;
        }
    }

    if (can_capture)
    {
        cached_actions_masks_ = capture_actions_mask;
        return cached_actions_masks_;
    }
    else
    {
        cached_actions_masks_ = no_capture_actions_mask;
        return cached_actions_masks_;
    }
}

EnglishDraughtState::RowColDirectionAction EnglishDraughtState::get_row_col_direction_from_action_(int action)
{
    int a = action / 4; // 4 is the number of directions a piece can move
    int row = a / (COLS / 2);
    int col = (a % (COLS / 2)) * 2 + (1 - (row % 2));

    int d_index = action % 4;
    const auto& direction = DIRECTIONS.at(d_index);
    return RowColDirectionAction{ row, col, std::get<0>(direction), std::get<1>(direction) };
}

bool EnglishDraughtState::is_king_(const Board& board, int row, int col)
{
    int player0_king = PLAYERS_K_FLAGS.at(0);
    int player1_king = PLAYERS_K_FLAGS.at(1);
    int piece_flag = board.at(row).at(col);
    return piece_flag == player0_king || piece_flag == player1_king;
}

bool EnglishDraughtState::is_empty_position_(const Board& board, int row, int col)
{
    return board.at(row).at(col) == 0;
}

int EnglishDraughtState::encode_action_(int row, int col, int direction_id)
{
    int a{ row * (COLS / 2) + col / 2 };
    int action{ a * 4 + direction_id };
    return action;
}

std::tuple<bool, bool> EnglishDraughtState::is_legal_is_capture_(const Board& board, int action, int player)
{
    auto rcd_action = get_row_col_direction_from_action_(action);
    int row = rcd_action.row;
    int col = rcd_action.col;
    int player_pawn_flag = PLAYERS_P_FLAGS.at(player);
    int player_king_flag = PLAYERS_K_FLAGS.at(player);
    if (board.at(row).at(col) != player_pawn_flag && board.at(row).at(col) != player_king_flag)
    {
        // we have no piece here to move
        return { false, false };
    }

    // we established that we have a piece on the board

    int row_dir = rcd_action.row_direction;
    int col_dir = rcd_action.col_direction;
    if (!is_king_(board, row, col) && is_backward_move_(row_dir, col_dir, player))
    {
        // our piece is not a king and the action is a moving backward action , which cannot be.
        return { false, false };
    }

    int target_row = row + row_dir;
    int target_col = col + col_dir;
    if (target_row < 0 || target_row >= ROWS || target_col < 0 || target_col >= ROWS)
    {
        // action is out of boundary
        return { false, false };
    }

    if (is_empty_position_(board, target_row, target_col))
    {
        // empty then can just move
        return { true, false };
    }

    if (can_capture_(board, row, col, row_dir, col_dir, player))
    {
        // can capture
        return { true, true };
    }

    return { false, false };
}

bool EnglishDraughtState::is_backward_move_(int row_direction, int col_direction, int player)
{
    if (player == 0)
    {
        return row_direction == 1;
    }
    if (player == 1)
    {
        return row_direction == -1;
    }
    throw rl::common::UnreachableCodeException("");
}

bool EnglishDraughtState::can_capture_(const Board& board, int row, int col, int row_direction, int col_direction, int player)
{
    int capture_row = row + row_direction;
    int capture_col = col + col_direction;
    int jump_row = capture_row + row_direction;
    int jump_col = capture_col + col_direction;

    if (jump_row < 0 || jump_row >= ROWS || jump_col < 0 || jump_col >= COLS)
    {
        // player piece cannot jump outside the board return false
        return false;
    }

    int opponent = 1 - player;
    int opponent_pawn_flag = PLAYERS_P_FLAGS.at(opponent);
    int opponent_king_flag = PLAYERS_K_FLAGS.at(opponent);
    if (board.at(capture_row).at(capture_col) == opponent_pawn_flag || board.at(capture_row).at(capture_col) == opponent_king_flag)
    {
        // an opponent piece is at capture position , check if the jump position is empty
        if (is_empty_position_(board, jump_row, jump_col))
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    // no opponent piece then the player cannot capture
    return false;
}

bool EnglishDraughtState::is_opponent_win_() const
{
    // check if the player has any piece on board
    int player = current_player_;
    int player_pawn_flag = PLAYERS_P_FLAGS.at(player);
    int player_king_flag = PLAYERS_K_FLAGS.at(player);
    bool has_units = false;
    for (int row{ 0 }; row < ROWS && has_units == false; row++)
    {
        for (int col{ 0 }; col < COLS && has_units == false; col++)
        {
            if (board_.at(row).at(col) == player_pawn_flag || board_.at(row).at(col) == player_king_flag)
            {
                has_units = true;
            }
        }
    }
    if (has_units == false)
    {
        // if player does not have units then the opponent won
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
    return true;
}

bool EnglishDraughtState::is_draw_() const
{
    return n_no_capture_rounds_ == MAX_NO_CAPTURE_ROUNDS;
}

void EnglishDraughtState::add_no_capture_rounds_observation_(std::vector<float>& observation_out) const
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

void EnglishDraughtState::add_last_jump_observation_(std::vector<float>& observation_out) const
{
    if (last_jump_.size() == 0)
    {
        return;
    }
    constexpr int last_jump_obs_channel_id = 5;
    constexpr int channel_size = ROWS * COLS;
    constexpr int channel_start = last_jump_obs_channel_id * channel_size;
    constexpr int channel_end = channel_start + channel_size;

    int row = last_jump_.at(0);
    int col = last_jump_.at(1);
    int cell_to_be_modified = channel_start + row * COLS + col;
    observation_out.at(cell_to_be_modified) = 1.0f;
}

void EnglishDraughtState::add_current_player_turn_observation_(std::vector<float>& observation_out) const
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

std::unique_ptr<EnglishDraughtState> EnglishDraughtState::clone_state() const
{
    return std::unique_ptr<EnglishDraughtState>(new EnglishDraughtState(*this));
}
std::unique_ptr<rl::common::IState> EnglishDraughtState::clone() const
{
    return clone_state();
}

void EnglishDraughtState::get_symmetrical_obs_and_actions(std::vector<float> const& obs, std::vector<float> const& actions_distribution, std::vector<std::vector<float>>& out_syms, std::vector<std::vector<float>>& out_actions_distribution)const
{
    out_syms.clear();
    out_actions_distribution.clear();
}
} // namespace rl::games
