

#include <iostream>
#include <sstream>
#include <iomanip>

#include <games/ultimate_tictactoe.hpp>
#include <common/exceptions.hpp>


namespace rl::games
{



UltimateTicTacToeState::~UltimateTicTacToeState() = default;



UltimateTicTacToeState::UltimateTicTacToeState(std::array<std::array<std::array<int, COLS>, ROWS>, BOARDS> board, std::array<std::array<int, COLS>, ROWS> ultimate_board, std::array<std::array<int, COLS>, ROWS> ultimate_board_terminals, int player, int last_action)
    : board_(board),
    ultimate_board_(ultimate_board),
    ultimate_board_terminals_(ultimate_board_terminals),
    player_{ player },
    last_action_{ last_action },
    legal_actions_{},
    is_game_over_cached_{},
    game_result_cached_{}
{
}


std::unique_ptr<rl::common::IState> UltimateTicTacToeState::initialize()
{
    return UltimateTicTacToeState::initialize_state();
}


std::unique_ptr<UltimateTicTacToeState> UltimateTicTacToeState::initialize_state()
{

    std::array<std::array<std::array<int, COLS>, ROWS>, BOARDS> board{};
    std::array<std::array<int, COLS>, ROWS> ultimate_board{};
    std::array<std::array<int, COLS>, ROWS> ultimate_board_terminals{};
    int player = 0;
    int last_action = -1;
    return std::make_unique<UltimateTicTacToeState>(board, ultimate_board, ultimate_board_terminals, player, last_action);
}

std::unique_ptr<UltimateTicTacToeState> UltimateTicTacToeState::reset_state() const
{
    return UltimateTicTacToeState::initialize_state();
}



std::unique_ptr<rl::common::IState> UltimateTicTacToeState::reset()const {
    return reset_state();
}


std::unique_ptr<UltimateTicTacToeState> UltimateTicTacToeState::step_state(int action) const
{
    std::vector<bool> mask = actions_mask();
    if (!mask.at(action))
    {
        throw rl::common::IllegalActionException("Stepping a state with invalid action");
    }
    if (is_terminal())
    {
        throw rl::common::SteppingTerminalStateException("Stepping a terminal state");
    }

    int last_action = action;
    int current_player = player_;
    int current_player_flag = FLAGS.at(current_player);
    int next_player = 1 - current_player;

    std::array<std::array<std::array<int, ROWS>, COLS>, BOARDS> next_board(board_);

    int action_board_no = action / (ROWS * COLS);
    int action_row = action % (ROWS * COLS) / COLS;
    int action_col = action % (ROWS * COLS) % COLS;

    next_board.at(action_board_no).at(action_row).at(action_col) = current_player_flag;

    std::array<std::array<int, ROWS>, COLS> next_ultimate_board = ultimate_board_;
    std::array<std::array<int, ROWS>, COLS> next_ultimate_board_terminals = ultimate_board_terminals_;
    int ultimate_board_row = action_board_no / COLS;
    int ultimate_board_col = action_board_no % COLS;
    if (is_winning(current_player, next_board.at(action_board_no)))
    {
        next_ultimate_board.at(ultimate_board_row).at(ultimate_board_col) = current_player_flag;
        next_ultimate_board_terminals.at(ultimate_board_row).at(ultimate_board_col) = 1;
    }
    else if (is_full(next_board.at(action_board_no))) {
        next_ultimate_board_terminals.at(ultimate_board_row).at(ultimate_board_col) = 1;
    }


    return std::make_unique<UltimateTicTacToeState>(next_board, next_ultimate_board, next_ultimate_board_terminals, next_player, last_action);
}

void UltimateTicTacToeState::render() const
{
    auto legal_actions = actions_mask();

    std::cout << "\n";
    std::cout << "    b0    b1    b2       b3    b4    b5       b6    b7    b8\n";
    std::cout << "  0 1 2  0 1 2  0 1 2   0 1 2  0 1 2  0 1 2   0 1 2  0 1 2  0 1 2\n";

    for (int row = 0; row < ROWS; row++)
    {
        std::cout << row << "  ";

        for (int meta_col = 0; meta_col < COLS; meta_col++)
        {
            for (int board_in_row = 0; board_in_row < COLS; board_in_row++)
            {
                int board_no = meta_col * COLS + board_in_row;
                for (int col = 0; col < COLS; col++)
                {
                    int cell = board_.at(board_no).at(row).at(col);
                    int ultimate_owner = ultimate_board_.at(board_no / COLS).at(board_no % COLS);

                    if (ultimate_owner != 0)
                    {
                        std::cout << (ultimate_owner == 1 ? "X" : "O") << " ";
                    }
                    else if (cell == 1)
                    {
                        std::cout << "x ";
                    }
                    else if (cell == -1)
                    {
                        std::cout << "o ";
                    }
                    else if (legal_actions.at(board_no * ROWS * COLS + row * COLS + col))
                    {
                        int action = board_no * ROWS * COLS + row * COLS + col;
                        std::cout << std::setw(2) << std::setfill('0') << action << "";
                    }
                    else
                    {
                        std::cout << ". ";
                    }
                }
                if (board_in_row < COLS - 1)
                {
                    std::cout << " ";
                }
            }
            if (meta_col < COLS - 1)
            {
                std::cout << "  ";
            }
        }
        std::cout << "\n";
    }

    // Print ultimate board status
    std::cout << "\nUltimate Board:\n";
    for (int row = 0; row < ROWS; row++)
    {
        std::cout << "  ";
        for (int col = 0; col < COLS; col++)
        {
            int cell = ultimate_board_.at(row).at(col);
            if (cell == 1)
            {
                std::cout << "X ";
            }
            else if (cell == -1)
            {
                std::cout << "O ";
            }
            else
            {
                if (ultimate_board_terminals_.at(row).at(col))
                {
                    std::cout << "# "; // terminal but drawn
                }
                else
                {
                    std::cout << ". ";
                }
            }
        }
        std::cout << "\n";
    }

    char player_char = 'X';
    if (player_ == 1)
    {
        player_char = 'O';
    }

    std::cout << "\nPlayer " << player_char << " to move" << std::endl;
}

std::unique_ptr<rl::common::IState> UltimateTicTacToeState::step(int action) const
{
    return step_state(action);
}



std::vector<float> UltimateTicTacToeState::get_observation() const
{

    if (observation_cached_.size())
    {
        return observation_cached_;
    }
    /*
    observation consists of 30 channels , YES 30 channels
    [0,9] channels are for the current player (9 for each board and 1 for the ultimate board)
    [10,19] channels are for the opponent player (9 for each board and 1 for the ultimate board)
    channels [20-28] comes the legal actions for the 9 boards( ultimate channel has no actions)
    the [29] channel is all ones, makes the neural network understand the difference between real inputs and paddings
    */

    int current_player{ player_ };
    int opponent{ 1 - current_player };
    int player_flag{ FLAGS.at(current_player) };
    int opponent_flag{ FLAGS.at(opponent) };

    std::vector<float> observation;
    constexpr int CELLS = CHANNELS * ROWS * COLS;
    observation.resize(CELLS);

    for (size_t board_no = 0; board_no < BOARDS; board_no++)
    {
        for (size_t row = 0; row < ROWS; row++)
        {
            for (size_t col = 0; col < COLS; col++)
            {
                int cell = board_.at(board_no).at(row).at(col);
                if (cell)
                {
                    int relative_board = board_no;

                    if (cell == opponent_flag)
                    {
                        relative_board += 10; // opponent first board is 10
                    }

                    int cell_index = relative_board * (ROWS * COLS) + row * COLS + col;
                    observation.at(cell_index) = 1.0;
                }
            }
        }
    }

    // ULTIMATE CHANNEL

    for (size_t board_no = BOARDS; board_no < BOARDS + 1; board_no++)
    {
        for (size_t row = 0; row < ROWS; row++)
        {
            for (size_t col = 0; col < COLS; col++)
            {
                int cell = ultimate_board_.at(row).at(col);
                if (cell)
                {
                    int relative_board = board_no;

                    if (cell == opponent_flag)
                    {
                        relative_board += 10; // opponent ultimate board is 19
                    }

                    int cell_index = relative_board * (ROWS * COLS) + row * COLS + col;
                    observation.at(cell_index) = 1.0;
                }
            }
        }
    }

    auto masks = actions_mask();

    constexpr int ACTIONS_INITIAL_CELL = ((BOARDS + 1) * 2) * ROWS * COLS;

    for (size_t i = 0; i < masks.size(); i++)
    {
        observation.at(ACTIONS_INITIAL_CELL + i) = static_cast<float>(masks.at(i));
    }

    // ALL ONES CHANNEL

    constexpr int ALL_ONES_CHANNEL = ((BOARDS + 1) * 2) + BOARDS;

    constexpr int ALL_ONES_INITIAL_CELL_INDEX = ALL_ONES_CHANNEL * (ROWS * COLS);

    for (size_t i = 0; i < ROWS * COLS; i++)
    {
        observation.at(ALL_ONES_INITIAL_CELL_INDEX + i) = 1.0f;
    }

    observation_cached_ = observation;
    return observation;
}


std::string UltimateTicTacToeState::to_short() const
{
    std::stringstream ss;
    for (size_t board_no = 0; board_no < BOARDS; board_no++)
    {
        for (int row{ 0 }; row < ROWS; row++)
        {
            for (int col{ 0 }; col < COLS; col++)
            {
                ss << (board_.at(board_no).at(row).at(col) == 1 ? 'X' : board_.at(board_no).at(row).at(col) == -1 ? 'O'
                    : ' ');
            }
        }
    }
    ss << '#' << last_action_;
    return ss.str();
}


std::array<int, 3> UltimateTicTacToeState::get_observation_shape() const
{
    return { {CHANNELS,ROWS,COLS } };
}


bool UltimateTicTacToeState::is_legal_action(int action) const
{
    int action_board_no = action / (ROWS * COLS);
    int action_row = action % (ROWS * COLS) / COLS;
    int action_col = action % (ROWS * COLS) % COLS;

    int ultimate_board_row = action_board_no / COLS;
    int ultimate_board_col = action_board_no % COLS;
    if (board_.at(action_board_no).at(action_row).at(action_col) != 0 || ultimate_board_.at(ultimate_board_row).at(ultimate_board_col) != 0)
    {
        return false;
    }

    if (last_action_ >= 0)
    {
        int last_action_board_no = last_action_ / (ROWS * COLS);
        int last_action_row = last_action_ % (ROWS * COLS) / COLS;
        int last_action_col = last_action_ % (ROWS * COLS) % COLS;
        if (!ultimate_board_terminals_.at(last_action_row).at(last_action_col))
        {
            if (ultimate_board_row != last_action_row || ultimate_board_col != last_action_col)
            {
                return false;
            }
        }
    }
    if (ultimate_board_terminals_.at(ultimate_board_row).at(ultimate_board_col))
    {
        return false;
    }

    return true;
}


int UltimateTicTacToeState::get_n_actions() const
{
    return N_ACTIONS;
}





int UltimateTicTacToeState::player_turn() const
{
    return player_;
}

int UltimateTicTacToeState::get_last_action() const
{
    return last_action_;
}


std::vector<bool> UltimateTicTacToeState::actions_mask() const

{
    if (legal_actions_.size())
    {
        return legal_actions_;
    }

    std::vector<bool> legal_actions{};

    legal_actions_.reserve(get_n_actions());

    for (size_t action = 0; action < get_n_actions(); action++)
    {
        legal_actions_.emplace_back(is_legal_action(action));
    }
    return legal_actions_;
}





bool UltimateTicTacToeState::is_terminal()const
{
    if (is_game_over_cached_.has_value())
    {
        return is_game_over_cached_.value();
    }

    int current_player = player_turn();
    int opponent = 1 - current_player;

    // check if oppponent won
    if (is_winning(opponent, ultimate_board_))
    {
        is_game_over_cached_.emplace(true);
        return is_game_over_cached_.value();
    }


    // check if all boards are terminals
    for (size_t row = 0; row < ROWS; row++)
    {
        for (size_t col = 0; col < COLS; col++)
        {
            if (!ultimate_board_terminals_.at(row).at(col))
            {
                is_game_over_cached_.emplace(false);
                return is_game_over_cached_.value();
            }
        }
    }

    is_game_over_cached_.emplace(true);
    return is_game_over_cached_.value();;
}


float UltimateTicTacToeState::get_reward() const
{
    if (!is_terminal())
    {
        return 0.0f;
    }

    if (game_result_cached_.has_value())
    {
        return game_result_cached_.value();
    }

    int player = player_;
    int opponent = 1 - player;

    if (is_winning(opponent))
    {
        game_result_cached_.emplace(-1.0f);
    }
    else {
        game_result_cached_.emplace(0.0f);
    }
    return game_result_cached_.value();
}


bool UltimateTicTacToeState::is_winning(int player) const
{
    return is_winning(player, ultimate_board_);
}

bool UltimateTicTacToeState::is_winning(int player, const std::array<std::array<int, COLS>, ROWS>& board) const
{
    return is_horizontal_win(player, board) || is_vertical_win(player, board) || is_forward_diagonal_win(player, board) || is_backward_diagonal_win(player, board);
}

bool UltimateTicTacToeState::is_full(const std::array<std::array<int, COLS>, ROWS>& board) const
{
    for (size_t row = 0; row < ROWS; row++)
    {
        for (size_t col = 0; col < COLS; col++)
        {
            if (board.at(row).at(col) == 0)
            {
                return false;
            }
        }
    }
    return true;
}

std::unique_ptr<rl::common::IState> UltimateTicTacToeState::clone() const
{
    return clone_state();
}

std::unique_ptr<UltimateTicTacToeState> UltimateTicTacToeState::clone_state() const
{
    return std::unique_ptr<UltimateTicTacToeState>(new UltimateTicTacToeState(*this));
}

void UltimateTicTacToeState::get_symmetrical_obs_and_actions(std::vector<float> const& obs, std::vector<float> const& actions_distribution, std::vector<std::vector<float>>& out_syms, std::vector<std::vector<float>>& out_actions_distribution) const
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
            result.at(i) = source.at(mapping_array.at(i));
        }
        return result;
    };

    auto add_symmetry = [&](const auto& obs_map, const auto& act_map) {
        out_syms.push_back(apply_sym(obs, obs_map));
        out_actions_distribution.push_back(apply_sym(actions_distribution, act_map));
    };

    using namespace ultimate_tictactoe_syms;

    add_symmetry(ROT90_OBS_SYM, ROT90_ACTIONS_SYM);
    add_symmetry(ROT180_OBS_SYM, ROT180_ACTIONS_SYM);
    add_symmetry(ROT270_OBS_SYM, ROT270_ACTIONS_SYM);
    add_symmetry(FLIP_LR_OBS_SYM, FLIP_LR_ACTIONS_SYM);
    add_symmetry(FLIP_UD_OBS_SYM, FLIP_UD_ACTIONS_SYM);
    add_symmetry(TRANSPOSE_OBS_SYM, TRANSPOSE_ACTIONS_SYM);
    add_symmetry(ANTI_TRANSPOSE_OBS_SYM, ANTI_TRANSPOSE_ACTIONS_SYM);
}


bool UltimateTicTacToeState::is_horizontal_win(int player, const std::array<std::array<int, COLS>, ROWS>& board) const
{
    int player_flag = FLAGS.at(player);
    bool horizontal = (board.at(0).at(0) == player_flag && board.at(0).at(1) == player_flag && board.at(0).at(2) == player_flag) || ((board.at(1).at(0) == player_flag && board.at(1).at(1) == player_flag && board.at(1).at(2) == player_flag)) || (board.at(2).at(0) == player_flag && board.at(2).at(1) == player_flag && board.at(2).at(2) == player_flag);
    return horizontal;
}

bool UltimateTicTacToeState::is_vertical_win(int player, const std::array<std::array<int, COLS>, ROWS>& board) const
{
    int player_flag = FLAGS.at(player);
    bool vertical = (board.at(0).at(0) == player_flag && board.at(1).at(0) == player_flag && board.at(2).at(0) == player_flag) || (board.at(0).at(1) == player_flag && board.at(1).at(1) == player_flag && board.at(2).at(1) == player_flag) || (board.at(0).at(2) == player_flag && board.at(1).at(2) == player_flag && board.at(2).at(2) == player_flag);
    return vertical;
}

bool UltimateTicTacToeState::is_forward_diagonal_win(int player, const std::array<std::array<int, COLS>, ROWS>& board) const
{
    int player_flag = FLAGS.at(player);
    bool forward = (board.at(0).at(0) == player_flag && board.at(1).at(1) == player_flag && board.at(2).at(2) == player_flag);
    return forward;
}

bool UltimateTicTacToeState::is_backward_diagonal_win(int player, const std::array<std::array<int, COLS>, ROWS>& board) const
{
    int player_flag = FLAGS.at(player);
    bool backward = (board.at(0).at(2) == player_flag && board.at(1).at(1) == player_flag && board.at(2).at(0) == player_flag);
    return backward;
}

















} // namespace rl::games


