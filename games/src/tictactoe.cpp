#include <iostream>
#include <sstream>

#include <games/tictactoe.hpp>
#include <common/exceptions.hpp>
namespace rl::games
{
TicTacToeState::TicTacToeState(std::array<std::array<int8_t, ROWS>, COLS> board, int8_t player)
    : board_(board),
    player_{ player },
    legal_actions_{},
    is_game_over_cached_{ false },
    is_game_over_{ false },
    is_game_result_cached_{ false },
    game_result_cache_{ 0.0f }
{
}
std::unique_ptr<TicTacToeState> TicTacToeState::initialize_state()
{
    std::array<std::array<int8_t, 3>, 3> array{};
    int player = 0;
    return std::make_unique<TicTacToeState>(array, player);
}

std::unique_ptr<rl::common::IState> TicTacToeState::initialize()
{
    return TicTacToeState::initialize_state();
}

std::unique_ptr<TicTacToeState> TicTacToeState::reset_state() const
{
    return TicTacToeState::initialize_state();
}
std::unique_ptr<rl::common::IState> TicTacToeState::reset() const
{
    return reset_state();
}

std::unique_ptr<TicTacToeState> TicTacToeState::step_state(int action) const
{
    std::vector<bool> mask = actions_mask();
    if (!mask.at(action))
    {
        throw rl::common::SteppingTerminalStateException("");
    }
    int current_player = player_;
    int current_player_flag = FLAGS.at(current_player);
    int next_player = 1 - current_player;
    std::array<std::array<int8_t, ROWS>, COLS> next_board(board_);
    int action_row = action / COLS;
    int action_col = action % COLS;
    next_board.at(action_row).at(action_col) = current_player_flag;
    return std::make_unique<TicTacToeState>(next_board, next_player);
}

std::unique_ptr<rl::common::IState> TicTacToeState::step(int action) const
{
    return step_state(action);
}

std::vector<float> TicTacToeState::get_observation() const
{
    /*  observation consists of 2 binary channels/layers/boards
        the 0-index indicates the current player
        the 1-index indicates the opponent player
        each channel consists of 9 flat cells , 3 rows 3 cols
    */
    int current_player{ player_ };
    int opponent{ 1 - current_player };
    int player_flag{ FLAGS.at(current_player) };
    int opponent_flag{ FLAGS.at(opponent) };

    std::vector<float> observation;
    observation.reserve(CHANNELS * ROWS * COLS);
    for (int row{ 0 }; row < ROWS; row++)
    {
        for (int col{ 0 }; col < COLS; col++)
        {
            int cell_index = row * COLS + col;
            observation.emplace_back(board_.at(row).at(col) == player_flag ? 1.0f : 0.0f);
        }
    }
    for (int row{ 0 }; row < ROWS; row++)
    {
        for (int col{ 0 }; col < COLS; col++)
        {
            int cell_index = row * COLS + col;
            observation.emplace_back(board_.at(row).at(col) == opponent_flag ? 1.0f : 0.0f);
        }
    }
    return observation;
}

std::string TicTacToeState::to_short() const
{
    std::stringstream ss;
    for (int row{ 0 }; row < ROWS; row++)
    {
        for (int col{ 0 }; col < COLS; col++)
        {
            ss << (board_.at(row).at(col) == 1 ? 'X' : board_.at(row).at(col) == -1 ? 'O'
                : ' ');
        }
    }
    return ss.str();
}

bool TicTacToeState::is_legal_action(int action) const
{
    int action_row = action / COLS;
    int action_col = action % COLS;
    return board_.at(action_row).at(action_col) == 0;
}

std::vector<bool> TicTacToeState::actions_mask() const
{
    if (legal_actions_.size())
    {
        return legal_actions_;
    }
    std::vector<bool> legal_actions{};
    // legal_actions.reserve(ROWS * COLS);
    legal_actions_.reserve(ROWS * COLS);
    for (int row{ 0 }; row < ROWS; row++)
    {
        for (int col{ 0 }; col < COLS; col++)
        {
            // legal_actions.emplace_back(board_.at(row).at(col) == 0);
            legal_actions_.emplace_back(board_.at(row).at(col) == 0);
        }
    }
    return legal_actions_;
}

bool TicTacToeState::is_terminal() const
{
    if (is_game_over_cached_)
    {
        return is_game_over_;
    }
    int player_0 = 0;
    int player_1 = 1;
    is_game_over_ = is_winning(player_0) || is_winning(player_1) || is_full();
    is_game_over_cached_ = true;
    return is_game_over_;
}

float TicTacToeState::get_reward() const
{
    if (!is_terminal())
    {
        return 0.0f;
    }
    if (is_game_result_cached_)
    {
        return game_result_cache_;
    }
    int player{ player_ };
    int opponent{ 1 - player };
    if (is_winning(player))
    {
        game_result_cache_ = 1.0f;
    }
    else if (is_winning(opponent))
    {
        game_result_cache_ = -1.0f;
    }
    else
    {
        game_result_cache_ = 0.0f;
    }
    is_game_result_cached_ = true;
    return game_result_cache_;
}

int TicTacToeState::get_n_actions() const
{
    return ROWS * COLS;
}
std::array<int, 3> TicTacToeState::get_observation_shape() const
{
    return { CHANNELS, ROWS, COLS };
}

int TicTacToeState::player_turn() const
{
    return player_;
}
void TicTacToeState::render() const
{
    std::stringstream ss;

    int player = player_;
    int opponent = 1 - player;
    char player_rep;
    char opponent_rep;
    if (player == 0)
    {
        player_rep = 'x';
        opponent_rep = 'o';
    }
    else if (player == 1)
    {
        player_rep = 'o';
        opponent_rep = 'x';
    }

    ss << "****************************\n";
    ss << "*** Player " << player_rep << " has to move ***\n";
    ss << "****************************\n";
    ss << '\n';
    for (int row{ 0 }; row < ROWS; row++)
    {
        for (int col{ 0 }; col < COLS; col++)
        {
            int action_no = row * COLS + col;
            int8_t value = board_.at(row).at(col);
            if (value == 1)
            {
                ss << 'x';
            }
            else if (value == -1)
            {
                ss << 'o';
            }
            else
            {
                ss << action_no;
            }
        }
        ss << '\n';
    }
    ss << '\n';
    std::cout << ss.str();
}

bool TicTacToeState::is_winning(int player) const
{
    return is_horizontal_win(player) || is_vertical_win(player) || is_forward_diagonal_win(player) || is_backward_diagonal_win(player);
}

bool TicTacToeState::is_full() const
{
    for (int row{ 0 }; row < ROWS; row++)
    {
        for (int col{ 0 }; col < COLS; col++)
        {
            if (board_.at(row).at(col) == 0)
                return false;
        }
    }
    return true;
}

bool TicTacToeState::is_horizontal_win(int player) const
{
    int player_flag = FLAGS.at(player);
    bool horizontal = (board_.at(0).at(0) == player_flag && board_.at(0).at(1) == player_flag && board_.at(0).at(2) == player_flag) || ((board_.at(1).at(0) == player_flag && board_.at(1).at(1) == player_flag && board_.at(1).at(2) == player_flag)) || (board_.at(2).at(0) == player_flag && board_.at(2).at(1) == player_flag && board_.at(2).at(2) == player_flag);
    return horizontal;
}
bool TicTacToeState::is_vertical_win(int player) const
{
    int player_flag = FLAGS.at(player);
    bool vertical = (board_.at(0).at(0) == player_flag && board_.at(1).at(0) == player_flag && board_.at(2).at(0) == player_flag) || (board_.at(0).at(1) == player_flag && board_.at(1).at(1) == player_flag && board_.at(2).at(1) == player_flag) || (board_.at(0).at(2) == player_flag && board_.at(1).at(2) == player_flag && board_.at(2).at(2) == player_flag);
    return vertical;
}

bool TicTacToeState::is_forward_diagonal_win(int player) const
{
    int player_flag = FLAGS.at(player);
    bool forward = (board_.at(0).at(0) == player_flag && board_.at(1).at(1) == player_flag && board_.at(2).at(2) == player_flag);
    return forward;
}
bool TicTacToeState::is_backward_diagonal_win(int player) const
{
    int player_flag = FLAGS.at(player);
    bool backward = (board_.at(0).at(2) == player_flag && board_.at(1).at(1) == player_flag && board_.at(2).at(0) == player_flag);
    return backward;
}


TicTacToeState::~TicTacToeState() = default;
std::unique_ptr<rl::common::IState> TicTacToeState::clone() const
{
    return clone_state();
}
std::unique_ptr<TicTacToeState> TicTacToeState::clone_state() const
{
    return std::unique_ptr<TicTacToeState>(new TicTacToeState(*this));
}

void TicTacToeState::get_symmetrical_obs_and_actions(std::vector<float> const& obs, std::vector<float> const& actions_distribution, std::vector<std::vector<float>>& out_syms, std::vector<std::vector<float>>& out_actions_distribution) const
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
        float value = obs.at(tictactoe_syms::FIRST_SYM.at(i));
        first_obs.emplace_back(value);
    }

    // add first actions sym
    out_actions_distribution.push_back({});
    std::vector<float>& first_actions = out_actions_distribution.at(0);
    first_actions.reserve(N_ACTIONS);
    for (int i = 0; i < N_ACTIONS; i++)
    {
        float value = actions_distribution.at(tictactoe_syms::FIRST_ACTIONS.at(i));
        first_actions.emplace_back(value);
    }

    // add second sym
    out_syms.push_back({});
    std::vector<float>& second_obs = out_syms.at(1);
    second_obs.reserve(obs.size());
    for (int i = 0; i < CHANNELS * ROWS * COLS; i++)
    {
        float value = obs.at(tictactoe_syms::SECOND_SYM.at(i));
        second_obs.emplace_back(value);
    }

    // add second action sym
    out_actions_distribution.push_back({});
    std::vector<float>& second_actions = out_actions_distribution.at(1);
    second_actions.reserve(N_ACTIONS);
    for (int i = 0; i < N_ACTIONS; i++)
    {
        float value = actions_distribution.at(tictactoe_syms::SECOND_ACTIONS.at(i));
        second_actions.emplace_back(value);
    }

    // add third sym
    out_syms.push_back({});
    std::vector<float>& third_obs = out_syms.at(2);
    third_obs.reserve(obs.size());
    for (int i = 0; i < CHANNELS * ROWS * COLS; i++)
    {
        float value = obs.at(tictactoe_syms::THIRD_SYM.at(i));
        third_obs.emplace_back(value);
    }

    // add third action sym
    out_actions_distribution.push_back({});
    std::vector<float>& third_actions = out_actions_distribution.at(2);
    third_actions.reserve(N_ACTIONS);
    for (int i = 0; i < N_ACTIONS; i++)
    {
        float value = actions_distribution.at(tictactoe_syms::THIRD_ACTIONS.at(i));
        third_actions.emplace_back(value);
    }
}

} // namespace rl::games