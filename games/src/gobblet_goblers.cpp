#include <iostream>
#include <sstream>

#include <games/gobblet_goblers.hpp>
#include <common/exceptions.hpp>

namespace rl::games
{
GobbletGoblersState::GobbletGoblersState(std::array<std::array<std::array<int8_t, COLS>, ROWS>, CHANNELS> board, int8_t player, int turn)
    :board_(board),
    player_{ player },
    turn_{ turn },
    legal_actions_{},
    cached_observation_{},
    cached_is_terminal_{},
    cached_result_{}
{
}

std::unique_ptr<GobbletGoblersState> GobbletGoblersState::initialize_state()
{
    std::array<std::array<std::array<int8_t, COLS>, ROWS>, CHANNELS> array{};
    // mark it as selection phase
    fill_channel(array, SELECTION_PHASE_CHANNEL, 1);
    int player = 0;
    return std::make_unique<GobbletGoblersState>(array, player, 0);
}


std::unique_ptr<rl::common::IState> GobbletGoblersState::initialize()
{
    return GobbletGoblersState::initialize_state();
}

std::unique_ptr<GobbletGoblersState> GobbletGoblersState::reset_state() const
{
    return GobbletGoblersState::initialize_state();
}



std::unique_ptr<rl::common::IState> GobbletGoblersState::reset() const
{
    return reset_state();
}

std::unique_ptr<GobbletGoblersState> GobbletGoblersState::step_state(int action) const
{
    std::vector<bool>mask = actions_mask();
    if (!mask.at(action))
    {
        throw rl::common::SteppingTerminalStateException("");
    }

    int current_player = player_;
    int opponent = 1 - current_player;
    int is_selection_phase = board_.at(SELECTION_PHASE_CHANNEL).at(0).at(0);
    int new_is_selection_phase = 1 - is_selection_phase;
    int new_player = new_is_selection_phase == 1 ? opponent : current_player;

    std::array<std::array<std::array<int8_t, COLS>, ROWS>, CHANNELS> new_board{};
    for (int c = 0;c < 6;c++)
    {
        // copy the first 6 channels
        new_board.at(c) = board_.at(c);
    }
    if (is_selection_phase)
    {
        if (action < ROWS * COLS)
        {
            // selecting a piece inside the board
            int row = action / COLS;
            int col = action % COLS;
            new_board.at(SELECTED_PIECE_ONBOARD_CHANNEL).at(row).at(col) = 1;
        }
        else {
            // selecting a piece outside the board
            int size = action - ROWS * COLS;
            int channel = SELECTED_PIECE_SMALL_CHANNEL + size;
            fill_channel(new_board, channel, 1);
        }
    }
    else { // moving phase
        if (action < ROWS * COLS)
        {
            // moving must be inside the board
            int target_row = action / COLS;
            int target_col = action % COLS;

            int size = -1;
            int src_row = -1;
            int src_col = -1;

            // get the moving unit place and size
            if (board_.at(SELECTED_PIECE_SMALL_CHANNEL).at(0).at(0) == 1)
            {
                size = 0;
            }
            else if (board_.at(SELECTED_PIECE_MEDIUM_CHANNEL).at(0).at(0) == 1)
            {
                size = 1;
            }
            else if (board_.at(SELECTED_PIECE_LARGE_CHANNEL).at(0).at(0) == 1)
            {
                size = 2;
            }
            else {
                // inside the board
                // get the size
                for (int row = 0;row < ROWS;row++)
                {
                    for (int col = 0;col < COLS;col++)
                    {
                        if (board_.at(SELECTED_PIECE_ONBOARD_CHANNEL).at(row).at(col) == 1)
                        {
                            src_row = row;
                            src_col = col;
                        }
                    }

                }
                if (src_col == -1 || src_row == -1)
                {
                    std::stringstream ss;
                    ss << "assertion failed , src " << src_row << ", " << src_col;
                    throw rl::common::UnreachableCodeException(ss.str());
                }

                for (int s = 0; s < 3;s++)
                {
                    if (board_.at(s).at(src_row).at(src_col) == 1)
                    {
                        size = s;
                    }
                }
            }

            if (size == -1)
            {
                std::stringstream ss;
                ss << "assertion failed , size " << size;
                throw rl::common::UnreachableCodeException(ss.str());
            }


            // if it is inside the board , remove it
            if (src_row >= 0 && src_col >= 0)
            {
                new_board.at(size).at(src_row).at(src_col) = 0;
            }

            // place it according to its size
            new_board.at(size).at(target_row).at(target_col) = 1;
        }
        else {
            std::stringstream ss;
            ss << "assertion failed , Moving action outside the board with action " << action;
            throw rl::common::UnreachableCodeException(ss.str());
        }
    }
    int new_turn = turn_;
    if (new_player != current_player)
    {
        // swap the channels , so the current player always have the first 3 channels
        for (int i = 0;i < 3;i++)
        {
            auto temp = new_board.at(i);
            new_board.at(i) = new_board.at(i + 3);
            new_board.at(i + 3) = temp;
        }

        new_turn++;
    }
    if (new_is_selection_phase)
    {
        fill_channel(new_board, SELECTION_PHASE_CHANNEL, static_cast<int>(static_cast<bool>(new_is_selection_phase)));
    }
    return std::make_unique<GobbletGoblersState>(new_board, new_player, new_turn);
}

std::unique_ptr<rl::common::IState> GobbletGoblersState::step(int action) const
{
    return step_state(action);
}

void GobbletGoblersState::render() const
{
    std::stringstream ss;

    int is_selection_phase = board_.at(SELECTION_PHASE_CHANNEL).at(0).at(0);

    int size = -1;
    int src_row = -1;
    int src_col = -1;

    if (!is_selection_phase)
    {
        // moving phase
        if (board_.at(SELECTED_PIECE_SMALL_CHANNEL).at(0).at(0) == 1)
        {
            size = 0;
        }
        else if (board_.at(SELECTED_PIECE_MEDIUM_CHANNEL).at(0).at(0) == 1)
        {
            size = 1;
        }
        else if (board_.at(SELECTED_PIECE_LARGE_CHANNEL).at(0).at(0) == 1)
        {
            size = 2;
        }
        else {
            // inside the board
            // get the size
            for (int row = 0;row < ROWS;row++)
            {
                for (int col = 0;col < COLS;col++)
                {
                    if (board_.at(SELECTED_PIECE_ONBOARD_CHANNEL).at(row).at(col) == 1)
                    {
                        src_row = row;
                        src_col = col;
                    }
                }

            }
            if (src_col == -1 || src_row == -1)
            {
                std::stringstream error_ss;
                error_ss << "assertion failed in render, src " << src_row << ", " << src_col;
                throw rl::common::UnreachableCodeException(error_ss.str());
            }

            for (int s = 0; s < 3;s++)
            {
                if (board_.at(s).at(src_row).at(src_col) == 1)
                {
                    size = s;
                }
            }
        }

        if (size == -1)
        {
            std::stringstream error_ss;
            error_ss << "assertion failed , size " << size;
            throw rl::common::UnreachableCodeException(error_ss.str());
        }
    }
    std::array<std::array<int, COLS>, ROWS> top_board{};

    get_top_board(board_, top_board);
    auto am = actions_mask();
    char player_rep, opponent_rep;
    if (player_ == 0)
    {
        player_rep = 'X';
        opponent_rep = 'O';
    }
    else {
        player_rep = 'O';
        opponent_rep = 'X';
    }

    ss << "****************************\n";
    ss << "*** Player " << player_rep << " has to " << (is_selection_phase ? "select" : "move") << "***\n";
    ss << "****************************\n";
    for (int row = 0;row < ROWS;row++)
    {
        ss << "+----+----+----+ +----+----+----+\n";
        for (int col = 0;col < COLS;col++)
        {
            ss << "| ";
            int cell = top_board.at(row).at(col);
            if (cell > 0)
            {
                if (row == src_row && col == src_col)
                {
                    ss << RED << player_rep << RESET << cell;
                }
                else {
                    ss << player_rep << cell;
                }

            }
            else if (cell < 0) {
                ss << opponent_rep << -cell;
            }
            else {
                ss << "  ";
            }
            ss << ' ';
        }
        ss << "| ";
        for (int col = 0;col < COLS;col++)
        {
            int action = row * COLS + col;
            ss << "| ";
            if (am.at(action))
            {
                ss << "0" << action;
            }
            else {
                ss << "  ";
            }
            ss << ' ';
        }
        ss << "|\n";
    }
    ss << "+----+----+----+ +----+----+----+\n";

    ss << "Player: " << player_rep << " Turn " << turn_ << " ";
    if (is_selection_phase)
    {
        ss << "Select a piece";
    }
    else {
        ss << "Move a piece";
        if (src_row >= 0 && src_col >= 0)
        {
            ss << "\nSelected Piece is " << src_row << "," << src_col;
        }
        ss << "\nSize" << size;
    }
    ss << "\n";
    ss << "legal actions are :";
    for (int i = 0;i < N_ACTIONS;i++)
    {
        if (am.at(i))
        {
            ss << i << " ,";
        }

    }
    ss << "\n";
    std::cout << ss.str();
}

// void GobbletGoblersState::render() const
// {
//     std::stringstream ss;

//     int is_selection_phase = board_.at(SELECTION_PHASE_CHANNEL).at(0).at(0);

//     int size = -1;
//     int src_row = -1;
//     int src_col = -1;

//     if (!is_selection_phase)
//     {
//         // moving phase
//         if (board_.at(SELECTED_PIECE_SMALL_CHANNEL).at(0).at(0) == 1)
//         {
//             size = 0;
//         }
//         else if (board_.at(SELECTED_PIECE_MEDIUM_CHANNEL).at(0).at(0) == 1)
//         {
//             size = 1;
//         }
//         else if (board_.at(SELECTED_PIECE_LARGE_CHANNEL).at(0).at(0) == 1)
//         {
//             size = 2;
//         }
//         else {
//             // inside the board
//             // get the size
//             for (int row = 0;row < ROWS;row++)
//             {
//                 for (int col = 0;col < COLS;col++)
//                 {
//                     if (board_.at(SELECTED_PIECE_ONBOARD_CHANNEL).at(row).at(col) == 1)
//                     {
//                         src_row = row;
//                         src_col = col;
//                     }
//                 }

//             }
//             if (src_col == -1 || src_row == -1)
//             {
//                 std::stringstream error_ss;
//                 error_ss << "assertion failed in render, src " << src_row << ", " << src_col;
//                 throw rl::common::UnreachableCodeException(error_ss.str());
//             }

//             for (int s = 0; s < 3;s++)
//             {
//                 if (board_.at(s).at(src_row).at(src_col) == 1)
//                 {
//                     size = s;
//                 }
//             }
//         }

//         if (size == -1)
//         {
//             std::stringstream error_ss;
//             error_ss << "assertion failed , size " << size;
//             throw rl::common::UnreachableCodeException(error_ss.str());
//         }
//     }
//     std::array<std::array<int, COLS>, ROWS> top_board{};

//     get_top_board(board_, top_board);

//     char player_rep, opponent_rep;
//     if (player_ == 0)
//     {
//         player_rep = 'X';
//         opponent_rep = 'O';
//     }
//     else {
//         player_rep = 'O';
//         opponent_rep = 'X';
//     }

//     ss << "****************************\n";
//     ss << "*** Player " << player_rep << " has to " << (is_selection_phase ? "select" : "move") << "***\n";
//     ss << "****************************\n";
//     for (int row = 0;row < ROWS;row++)
//     {
//         ss << "+----+----+----+\n";
//         for (int col = 0;col < COLS;col++)
//         {
//             ss << "| ";
//             int cell = top_board.at(row).at(col);
//             if (cell > 0)
//             {
//                 if (row == src_row && col == src_col)
//                 {
//                     ss << RED << player_rep << RESET << cell;
//                 }
//                 else {
//                     ss << player_rep << cell;
//                 }

//             }
//             else if (cell < 0) {
//                 ss << opponent_rep << -cell;
//             }
//             else {
//                 ss << "  ";
//             }
//             ss << ' ';
//         }
//         ss << "|\n";
//     }
//     ss << "+----+----+----+\n";

//     ss << "Player: " << player_rep << " Turn " << turn_ << " ";
//     if (is_selection_phase)
//     {
//         ss << "Select a piece";
//     }
//     else {
//         ss << "Move a piece";
//         if (src_row >= 0 && src_col >= 0)
//         {
//             ss << "\nSelected Piece is " << src_row << "," << src_col;
//         }
//         ss << "\nSize" << size;
//     }
//     ss << "\n";
//     ss << "legal actions are :";
//     auto am = actions_mask();
//     for (int i = 0;i < N_ACTIONS;i++)
//     {
//         if (am.at(i))
//         {
//             ss << i << " ,";
//         }

//     }
//     ss << "\n";
//     std::cout << ss.str();
// }

bool GobbletGoblersState::is_terminal() const
{
    if (cached_is_terminal_.has_value())
    {
        return cached_is_terminal_.value();
    }
    std::array<std::array<int, COLS>, ROWS> top_board{};
    for (int size = 0; size < 3;size++)
    {
        for (int row = 0;row < ROWS;row++)
        {
            for (int col = 0;col < COLS;col++)
            {
                if (board_.at(size).at(row).at(col))
                {
                    top_board.at(row).at(col) = size + 1;
                }
                else if (board_.at(size + 3).at(row).at(col))
                {
                    top_board.at(row).at(col) = -size - 1;
                }
            }
        }
    }

    int current_player = 0;
    int opponent = 1;
    bool is_current_player_winning = is_winning(top_board, current_player);
    bool is_opponent_winning = is_winning(top_board, opponent);

    if (is_current_player_winning && is_opponent_winning)
    {
        // if both can win then decide that the loser is the last one that moved

        // then opponent was the last one to move , our player is the winner
        if (!cached_result_.has_value())
        {
            cached_result_.emplace(1.0f);
        }
        cached_is_terminal_.emplace(true);
        return cached_is_terminal_.value();
    }

    if (is_current_player_winning)
    {
        if (!cached_result_.has_value())
        {
            cached_result_.emplace(1.0f);
        }
        cached_is_terminal_.emplace(true);
        return cached_is_terminal_.value();
    }

    if (is_opponent_winning)
    {
        if (!cached_result_.has_value())
        {
            cached_result_.emplace(-1.0f);
        }
        cached_is_terminal_.emplace(true);
        return cached_is_terminal_.value();
    }

    if (turn_ == MAX_TURNS)
    {
        if (!cached_result_.has_value())
        {
            cached_result_.emplace(0.0f);
        }
        cached_is_terminal_.emplace(true);
        return cached_is_terminal_.value();
    }

    auto masks = actions_mask();
    bool has_legal_actions = 0;
    for (bool a : masks)
    {
        if (a)
        {
            has_legal_actions = true;
            break;
        }
    }

    if (!has_legal_actions)
    {
        if (!cached_result_.has_value())
        {
            cached_result_.emplace(-1.0f);
        }
        cached_is_terminal_.emplace(true);
        return cached_is_terminal_.value();
    }


    if (!cached_result_.has_value())
    {
        cached_result_.emplace(0.0f);
    }
    cached_is_terminal_.emplace(false);
    return cached_is_terminal_.value();
}

float GobbletGoblersState::get_reward() const
{
    if (cached_result_.has_value())
    {
        return cached_result_.value();
    }

    // it should calculate the winner
    bool terminal = is_terminal();
    if (!terminal)
    {
        return 0.0f;
    }
    if (terminal && !cached_result_.has_value())
    {
        std::stringstream ss;
        ss << "assertion failed , game is terminal but cached result was not calculated ";
        throw rl::common::UnreachableCodeException(ss.str());
    }
    return cached_result_.value();
}


void GobbletGoblersState::get_top_board(std::array<std::array<std::array<int8_t, COLS>, ROWS>, CHANNELS> const& board, std::array<std::array<int, COLS>, ROWS>& top_board)
{
    for (int size = 0; size < 3;size++)
    {
        for (int row = 0;row < ROWS;row++)
        {
            for (int col = 0;col < COLS;col++)
            {
                if (board.at(size).at(row).at(col))
                {
                    top_board.at(row).at(col) = size + 1;
                }
                else if (board.at(size + 3).at(row).at(col))
                {
                    top_board.at(row).at(col) = -size - 1;
                }
            }
        }
    }
}

bool GobbletGoblersState::is_winning(std::array<std::array<int, COLS>, ROWS> const& top_board, int player)
{
    return is_vertical_win(top_board, player) || is_horizontal_win(top_board, player) || is_forward_diagonal_win(top_board, player) || is_backward_diagonal_win(top_board, player);
}

std::vector<float> GobbletGoblersState::get_observation() const
{
    if (cached_observation_.size())
    {
        return cached_observation_;
    }


    cached_observation_.resize(CHANNELS * ROWS * COLS);

    // copy the all channels except turn
    for (int channel = 0;channel < CHANNELS;channel++)
    {
        if (channel == TURN_CHANNEL)
        {
            continue;
        }
        for (int row = 0;row < ROWS;row++)
        {
            for (int col = 0;col < COLS;col++)
            {
                int cell = channel * ROWS * COLS + row * COLS + col;
                cached_observation_.at(cell) = board_.at(channel).at(row).at(col);
            }
        }
    }

    for (int row = 0;row < ROWS;row++)
    {
        for (int col = 0;col < COLS;col++)
        {
            int cell = TURN_CHANNEL * ROWS * COLS + row * COLS + col;
            cached_observation_.at(cell) = turn_ / static_cast<float>(MAX_TURNS);
        }
    }

    return cached_observation_;
}

std::string GobbletGoblersState::to_short() const
{
    // TODO later
    auto obs = get_observation();
    std::stringstream ss;
    for (auto a : obs)
    {
        ss << static_cast<int>(a);
    }
    ss << "," << player_ << "," << turn_;
    return ss.str();
}





void GobbletGoblersState::fill_channel(std::array<std::array<std::array<int8_t, COLS>, ROWS>, CHANNELS>& array, int channel, int fill_value)
{
    for (int row = 0;row < ROWS;row++)
    {
        for (int col = 0;col < COLS;col++)
        {
            array.at(channel).at(row).at(col) = fill_value;
        }
    }
}

std::array<int, 3> GobbletGoblersState::get_observation_shape() const
{
    return { CHANNELS,ROWS,COLS };
}

int GobbletGoblersState::get_n_actions() const
{
    return N_ACTIONS;
}

int GobbletGoblersState::player_turn() const
{
    return player_;
}

std::vector<bool> GobbletGoblersState::actions_mask() const
{
    if (legal_actions_.size())
    {
        return legal_actions_;
    }

    std::vector<bool> legal_actions{};

    legal_actions.reserve(N_ACTIONS);

    bool is_selection_phase = board_.at(SELECTION_PHASE_CHANNEL).at(0).at(0);

    if (is_selection_phase)
    {
        std::array<int, 3> unplayed_sizes_count{ 2,2,2 };
        std::array<std::array<int, COLS>, ROWS> final_board{};
        for (int size = 0; size < 3;size++)
        {
            for (int row = 0;row < ROWS;row++)
            {
                for (int col = 0;col < COLS;col++)
                {
                    if (board_.at(size).at(row).at(col))
                    {
                        final_board.at(row).at(col) = 1;
                        unplayed_sizes_count.at(size)--;
                    }
                    else if (board_.at(size + 3).at(row).at(col))
                    {
                        final_board.at(row).at(col) = -1;
                    }
                }
            }
        }

        for (int a = 0;a < N_ACTIONS;a++)
        {
            if (a < ROWS * COLS)
            {
                // inside the board selection
                int row = a / COLS;
                int col = a % COLS;
                // if players current piece are on the top then it is legal to pick it
                legal_actions.emplace_back(final_board.at(row).at(col) > 0);
            }
            else {
                // selection outside the board
                int size = a - ROWS * COLS;
                legal_actions.emplace_back(unplayed_sizes_count.at(size) > 0);
            }
        }
    }
    else {
        // moving piece

        // get the moving unit place and size
        int size = -1;
        int selected_row = -1;
        int selected_col = -1;

        if (board_.at(SELECTED_PIECE_SMALL_CHANNEL).at(0).at(0) == 1)
        {
            size = 0;
        }
        else if (board_.at(SELECTED_PIECE_MEDIUM_CHANNEL).at(0).at(0) == 1)
        {
            size = 1;
        }
        else if (board_.at(SELECTED_PIECE_LARGE_CHANNEL).at(0).at(0) == 1)
        {
            size = 2;
        }
        else {
            // inside the board
            // get the size
            for (int row = 0;row < ROWS;row++)
            {
                for (int col = 0;col < COLS;col++)
                {
                    if (board_.at(SELECTED_PIECE_ONBOARD_CHANNEL).at(row).at(col) == 1)
                    {
                        selected_row = row;
                        selected_col = col;
                    }
                }

            }
            if (selected_col == -1 || selected_row == -1)
            {
                std::stringstream ss;
                ss << "assertion failed , src " << selected_row << ", " << selected_col;
                throw rl::common::UnreachableCodeException(ss.str());
            }

            for (int s = 0; s < 3;s++)
            {
                if (board_.at(s).at(selected_row).at(selected_col) == 1)
                {
                    size = s;
                }
            }
        }

        if (size == -1)
        {
            std::stringstream ss;
            ss << "assertion failed , size " << size;
            throw rl::common::UnreachableCodeException(ss.str());
        }

        std::array<std::array<int, COLS>, ROWS> final_board_sizes{};
        for (int size = 0; size < 3;size++)
        {
            for (int row = 0;row < ROWS;row++)
            {
                for (int col = 0;col < COLS;col++)
                {
                    if (board_.at(size).at(row).at(col))
                    {
                        final_board_sizes.at(row).at(col) = size + 1;
                    }
                    else if (board_.at(size + 3).at(row).at(col))
                    {
                        final_board_sizes.at(row).at(col) = size + 1;
                    }
                }
            }
        }

        for (int a = 0;a < N_ACTIONS;a++)
        {
            int row = a / COLS;
            int col = a % COLS;
            if (row >= ROWS)
            {
                legal_actions.emplace_back(false);
                continue;
            }
            if (selected_row == row && selected_col == col)
            {
                // cannot move at the same place
                legal_actions.emplace_back(false);
                continue;
            }

            legal_actions.emplace_back(size + 1 > final_board_sizes.at(row).at(col));
        }
    }

    // cache legal_actions_
    legal_actions_ = legal_actions;
    if (legal_actions_.size() != N_ACTIONS)
    {
        std::stringstream ss;
        ss << "assertion failed , legal actions size:" << legal_actions_.size() << " is not equal to the number of action: " << N_ACTIONS;
        throw rl::common::UnreachableCodeException(ss.str());
    }
    return legal_actions_;
}
GobbletGoblersState::~GobbletGoblersState() = default;

std::unique_ptr<rl::common::IState> GobbletGoblersState::clone() const
{
    return clone_state();
}

std::unique_ptr<GobbletGoblersState> GobbletGoblersState::clone_state() const
{
    return std::unique_ptr<GobbletGoblersState>(new GobbletGoblersState(*this));
}

void GobbletGoblersState::get_symmetrical_obs_and_actions(std::vector<float> const& obs, std::vector<float> const& actions_distribution, std::vector<std::vector<float>>& out_syms, std::vector<std::vector<float>>& out_actions_distribution) const
{
    if (obs.size() != CHANNELS * ROWS * COLS)
    {
        std::stringstream ss;
        ss << "get_symmetrical_obs_and_actions requires an observation with size of " << CHANNELS * ROWS * COLS;
        ss << " but a size of " << obs.size() << " was passed.";
        throw std::runtime_error(ss.str());
    }


    out_syms.clear();
    out_actions_distribution.clear();

    // add first sym
    out_syms.push_back({});
    std::vector<float>& first_obs = out_syms.at(0);
    first_obs.reserve(obs.size());
    for (int i = 0; i < CHANNELS * ROWS * COLS; i++)
    {
        float value = obs.at(gobblet_syms::FIRST_OBS_SYM.at(i));
        first_obs.emplace_back(value);
    }

    // add first actions sym
    out_actions_distribution.push_back({});
    std::vector<float>& first_actions = out_actions_distribution.at(0);
    first_actions.reserve(N_ACTIONS);
    for (int i = 0; i < N_ACTIONS; i++)
    {
        float value = actions_distribution.at(gobblet_syms::FIRST_ACTIONS.at(i));
        first_actions.emplace_back(value);
    }

    // add second sym
    out_syms.push_back({});
    std::vector<float>& second_obs = out_syms.at(1);
    second_obs.reserve(obs.size());
    for (int i = 0; i < CHANNELS * ROWS * COLS; i++)
    {
        float value = obs.at(gobblet_syms::SECOND_OBS_SYM.at(i));
        second_obs.emplace_back(value);
    }

    // add second action sym
    out_actions_distribution.push_back({});
    std::vector<float>& second_actions = out_actions_distribution.at(1);
    second_actions.reserve(N_ACTIONS);
    for (int i = 0; i < N_ACTIONS; i++)
    {
        float value = actions_distribution.at(gobblet_syms::SECOND_ACTIONS.at(i));
        second_actions.emplace_back(value);
    }

    // add third sym
    out_syms.push_back({});
    std::vector<float>& third_obs = out_syms.at(2);
    third_obs.reserve(obs.size());
    for (int i = 0; i < CHANNELS * ROWS * COLS; i++)
    {
        float value = obs.at(gobblet_syms::THIRD_OBS_SYM.at(i));
        third_obs.emplace_back(value);
    }

    // add third action sym
    out_actions_distribution.push_back({});
    std::vector<float>& third_actions = out_actions_distribution.at(2);
    third_actions.reserve(N_ACTIONS);
    for (int i = 0; i < N_ACTIONS; i++)
    {
        float value = actions_distribution.at(gobblet_syms::THIRD_ACTIONS.at(i));
        third_actions.emplace_back(value);
    }
}

std::array<std::array<int, 3>, 3> GobbletGoblersState::get_board() const
{
    std::array<std::array<int, COLS>, ROWS> top_board{};
    get_top_board(board_, top_board);
    return top_board;
}
bool GobbletGoblersState::is_horizontal_win(std::array<std::array<int, COLS>, ROWS> const& top_board, int player)
{
    if (player == 0)
    {
        for (int row = 0;row < ROWS;row++)
        {
            if (top_board.at(row).at(0) > 0 && top_board.at(row).at(1) > 0 && top_board.at(row).at(2) > 0)
            {
                return true;
            }
        }
        return false;
    }
    else {
        for (int row = 0;row < ROWS;row++)
        {
            if (top_board.at(row).at(0) < 0 && top_board.at(row).at(1) < 0 && top_board.at(row).at(2) < 0)
            {
                return true;
            }
        }
        return false;
    }
}

bool GobbletGoblersState::is_vertical_win(std::array<std::array<int, COLS>, ROWS> const& top_board, int player)
{
    if (player == 0)
    {
        for (int col = 0;col < ROWS;col++)
        {
            if (top_board.at(0).at(col) > 0 && top_board.at(1).at(col) > 0 && top_board.at(2).at(col) > 0)
            {
                return true;
            }
        }
        return false;
    }
    else {
        for (int col = 0;col < ROWS;col++)
        {
            if (top_board.at(0).at(col) < 0 && top_board.at(1).at(col) < 0 && top_board.at(2).at(col) < 0)
            {
                return true;
            }
        }
        return false;
    }
}

bool GobbletGoblersState::is_forward_diagonal_win(std::array<std::array<int, COLS>, ROWS> const& top_board, int player)
{
    if (player == 0)
    {

        if (top_board.at(0).at(0) > 0 && top_board.at(1).at(1) > 0 && top_board.at(2).at(2) > 0)
        {
            return true;
        }

        return false;
    }
    else {
        if (top_board.at(0).at(0) < 0 && top_board.at(1).at(1) < 0 && top_board.at(2).at(2) < 0)
        {
            return true;
        }

        return false;
    }
}

bool GobbletGoblersState::is_backward_diagonal_win(std::array<std::array<int, COLS>, ROWS> const& top_board, int player)
{
    if (player == 0)
    {
        if (top_board.at(2).at(0) > 0 && top_board.at(1).at(1) > 0 && top_board.at(0).at(2) > 0)
        {
            return true;
        }

        return false;
    }
    else {
        if (top_board.at(2).at(0) < 0 && top_board.at(1).at(1) < 0 && top_board.at(0).at(2) < 0)
        {
            return true;
        }
        return false;
    }
}

}

