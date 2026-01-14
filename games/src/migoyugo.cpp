#include <sstream>
#include <stdexcept>
#include <iomanip>
#include <iostream>
#include <games/migoyugo.hpp>
#include <common/exceptions.hpp>


namespace rl::games
{

MigoyugoState::MigoyugoState(std::array<std::array<int8_t, COLS>, ROWS> board, int player, int step, int last_action)
    : board_(board),
    current_player_(player),
    step_(step),
    last_action_(last_action)
{
}

MigoyugoState::~MigoyugoState() = default;

std::unique_ptr<MigoyugoState> MigoyugoState::initialize_state()
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
    auto state_ptr = std::make_unique<MigoyugoState>(obs, player_0, 0, -1);
    return state_ptr;
}

std::unique_ptr<rl::common::IState> MigoyugoState::initialize()
{
    return initialize_state();
}


std::unique_ptr<rl::common::IState> MigoyugoState::reset() const
{
    return reset_state();
}
std::unique_ptr<MigoyugoState> MigoyugoState::reset_state() const
{
    return initialize_state();
}


std::unique_ptr<MigoyugoState> MigoyugoState::step_state(int action) const
{
    if (is_terminal())
    {
        throw rl::common::SteppingTerminalStateException("Trying to step a terminal state");
    }
    auto am = actions_mask();
    bool action_legality = am[action];
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

    std::vector<std::pair<int, int>> board_changes = this->get_board_changes_on_action(row_id, col_id);

    if (board_changes.size() > 0)
    {
        if (player == 0)
        {
            new_board.at(row_id).at(col_id) = 2;
        }
        else {
            new_board.at(row_id).at(col_id) = -2;
        }
    }
    else {
        if (player == 0)
        {
            new_board.at(row_id).at(col_id) = 1;
        }
        else {
            new_board.at(row_id).at(col_id) = -1;
        }
    }

    for (auto& cell : board_changes)
    {
        // for each cell
        const int& cell_row = cell.first;
        const int& cell_col = cell.second;
        const int cell_value = new_board.at(cell_row).at(cell_col);
        if (cell_value <= 1 && cell_value >= -1)
        {
            new_board.at(cell_row).at(cell_col) = 0;
        }
    }

    return std::make_unique<MigoyugoState>(new_board, other, step_ + 1, action);
}

void MigoyugoState::render() const
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
std::unique_ptr<rl::common::IState> MigoyugoState::step(int action) const
{
    return step_state(action);
}

int MigoyugoState::get_n_actions()const
{
    return ROWS * COLS;
}

int MigoyugoState::player_turn() const
{
    return current_player_;
}

std::array<int, 3> MigoyugoState::get_observation_shape() const
{
    return { CHANNELS, ROWS, COLS };
}

std::vector<std::pair<int, int>> MigoyugoState::get_direction_streak(int row, int col, int row_dir, int col_dir, int player) const
{
    std::vector<std::pair<int, int>> streak{};
    row += row_dir;
    col += col_dir;

    while (is_in_board(row, col))
    {
        if (player == 0)
        {
            if (board_.at(row).at(col) <= 0) // enemy spot
            {
                break;
            }
        }
        else { // player == 1
            if (board_.at(row).at(col) >= 0) // enemy spot
            {
                break;
            }
        }
        streak.push_back(std::make_pair(row, col));
        row += row_dir;
        col += col_dir;
    }

    return streak;
}


std::vector<std::pair<int, int>> MigoyugoState::get_board_changes_on_action(int row, int col)const
{
    std::vector<std::pair<int, int>> changes;
    std::array<std::array<std::pair<int, int>, 2>, 4> directions = { {
   {{ {0, 1}, {0, -1} }},
   {{ {1, 0}, {-1, 0} }},
   {{ {1, 1}, {-1, -1} }},
   {{ {1, -1}, {-1, 1} }}
        } };

    for (auto opposites : directions)
    {
        std::vector<std::pair<int, int>> streaks{};
        for (auto pair : opposites)
        {
            auto [row_dir, col_dir] = pair;
            auto direction_streak = get_direction_streak(row, col, row_dir, col_dir, current_player_);
            for (auto st : direction_streak)
            {
                streaks.push_back(st);
            }
        }

        if (streaks.size() > 3)
        {
            changes.clear();
            return changes;
        }
        if (streaks.size() == 3)
        {
            for (auto st : streaks)
            {
                changes.push_back(st);
            }
        }
    }


    return changes;
}

bool MigoyugoState::is_in_board(int row, int col) const
{
    return row < ROWS && row >= 0 && col < COLS && col >= 0;
}


bool MigoyugoState::is_terminal() const {
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
bool MigoyugoState::has_legal_action()const {
    auto am = actions_mask();
    for (bool is_legal : am)
    {
        if (is_legal)
        {
            return true;
        }
    }
    return false;
}


bool MigoyugoState::is_opponent_won()const
{
    const int player = current_player_;
    const int opponent = 1 - player;

    const int opponent_flag = opponent == 0 ? 2 : -2;
    // check if opponent won

    std::array<std::array<int8_t, COLS>, ROWS> opponent_yugos_board{};
    for (size_t row = 0; row < ROWS; row++)
    {
        for (size_t col = 0; col < COLS; col++)
        {
            if (board_.at(row).at(col) == opponent_flag)
            {
                opponent_yugos_board.at(row).at(col) = 1;
            }
        }
    }
    bool is_opponent_won = false;
    for (size_t row = 0; row < ROWS && !is_opponent_won; row++)
    {
        for (size_t col = 0; col < COLS && !is_opponent_won; col++)
        {
            if (opponent_yugos_board.at(row).at(col) == 1)
            {
                if (check_row_winning(opponent_yugos_board, row, col))
                {
                    is_opponent_won = true;
                }
                if (!is_opponent_won && check_col_winning(opponent_yugos_board, row, col))
                {
                    is_opponent_won = true;
                }
                if (!is_opponent_won && check_forward_diagonal_winning(opponent_yugos_board, row, col))
                {
                    is_opponent_won = true;
                }
                if (!is_opponent_won && check_backward_diagonal_winning(opponent_yugos_board, row, col))
                {
                    is_opponent_won = true;
                }
            }
        }
    }
    return is_opponent_won;
}
bool MigoyugoState::check_row_winning(std::array<std::array<int8_t, COLS>, ROWS> const& opponent_yugos_board, int row, int col)const
{
    if (!is_in_board(row + 3, col))
    {
        return false;
    };

    if (
        // opponent_yugos_board.at(row).at(col) &&
        opponent_yugos_board.at(row + 1).at(col) &&
        opponent_yugos_board.at(row + 2).at(col) &&
        opponent_yugos_board.at(row + 3).at(col)
        )
    {
        return true;
    }

    return false;
}


bool MigoyugoState::check_col_winning(std::array<std::array<int8_t, COLS>, ROWS> const& opponent_yugos_board, int row, int col)const
{
    if (!is_in_board(row, col + 3))
    {
        return false;
    };

    if (
        // opponent_yugos_board.at(row).at(col) &&
        opponent_yugos_board.at(row).at(col + 1) &&
        opponent_yugos_board.at(row).at(col + 2) &&
        opponent_yugos_board.at(row).at(col + 3)
        )
    {
        return true;
    }

    return false;
}

bool MigoyugoState::check_forward_diagonal_winning(std::array<std::array<int8_t, COLS>, ROWS> const& opponent_yugos_board, int row, int col)const
{
    if (!is_in_board(row + 3, col + 3))
    {
        return false;
    };

    if (
        // opponent_yugos_board.at(row).at(col) &&
        opponent_yugos_board.at(row + 1).at(col + 1) &&
        opponent_yugos_board.at(row + 2).at(col + 2) &&
        opponent_yugos_board.at(row + 3).at(col + 3)
        )
    {
        return true;
    }
    return false;
}


bool MigoyugoState::check_backward_diagonal_winning(std::array<std::array<int8_t, COLS>, ROWS> const& opponent_yugos_board, int row, int col)const
{
    if (!is_in_board(row + 3, col - 3))
    {
        return false;
    };

    if (
        // opponent_yugos_board.at(row).at(col) &&
        opponent_yugos_board.at(row + 1).at(col - 1) &&
        opponent_yugos_board.at(row + 2).at(col - 2) &&
        opponent_yugos_board.at(row + 3).at(col - 3)
        )
    {
        return true;
    }
    return false;
}


std::vector<bool> MigoyugoState::actions_mask() const
{
    if (cached_actions_masks_.size())
    {
        return cached_actions_masks_;
    }
    std::vector<bool> result(get_n_actions(), true);

    int action = 0;
    for (size_t row = 0; row < ROWS; row++)
    {
        for (size_t col = 0; col < COLS; col++)
        {
            action = encode_action(row, col);
            if (board_.at(row).at(col) != 0)
            {

                result.at(action) = false;
                continue;
            }

            std::array<std::array<std::pair<int, int>, 2>, 4> directions = { {
            {{ {0, 1}, {0, -1} }},
            {{ {1, 0}, {-1, 0} }},
            {{ {1, 1}, {-1, -1} }},
            {{ {1, -1}, {-1, 1} }}
                    } };

            for (auto opposites : directions)
            {
                std::vector<std::pair<int, int>> streaks{};
                for (auto pair : opposites)
                {
                    auto [row_dir, col_dir] = pair;
                    auto direction_streak = get_direction_streak(row, col, row_dir, col_dir, current_player_);
                    for (auto st : direction_streak)
                    {
                        streaks.push_back(st);
                    }
                }

                if (streaks.size() > 3)
                {
                    result.at(action) = false;
                }
            }


        }
    }
    cached_actions_masks_ = result;
    return cached_actions_masks_;

}

float MigoyugoState::get_reward() const
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
    int player_1_yugos = 1;
    int cell = 0;
    for (size_t row = 0; row < ROWS; row++)
    {
        for (size_t col = 0; col < COLS; col++)
        {
            cell = board_.at(row).at(col);
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

int MigoyugoState::encode_action(int row, int col)
{
    int action = row * COLS + col;
    return action;
}


std::unique_ptr<MigoyugoState> MigoyugoState::clone_state() const
{
    return std::unique_ptr<MigoyugoState>(new MigoyugoState(*this));
}

std::unique_ptr<rl::common::IState> MigoyugoState::clone() const
{
    return clone_state();
}

std::string MigoyugoState::to_short() const
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

void MigoyugoState::get_symmetrical_obs_and_actions(std::vector<float> const& obs, std::vector<float> const& actions_distribution, std::vector<std::vector<float>>& out_syms, std::vector<std::vector<float>>& out_actions_distribution) const
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
        float value = obs.at(miguyugo_syms::FIRST_OBS_SYM.at(i));
        first_obs.emplace_back(value);
    }

    // add first actions sym
    out_actions_distribution.push_back({});
    std::vector<float>& first_actions = out_actions_distribution.at(0);
    first_actions.reserve(N_ACTIONS);
    for (int i = 0; i < N_ACTIONS; i++)
    {
        float value = actions_distribution.at(miguyugo_syms::FIRST_ACTIONS_SYM.at(i));
        first_actions.emplace_back(value);
    }

    // add second sym
    out_syms.push_back({});
    std::vector<float>& second_obs = out_syms.at(1);
    second_obs.reserve(obs.size());
    for (int i = 0; i < CHANNELS * ROWS * COLS; i++)
    {
        float value = obs.at(miguyugo_syms::SECOND_OBS_SYM.at(i));
        second_obs.emplace_back(value);
    }

    // add second action sym
    out_actions_distribution.push_back({});
    std::vector<float>& second_actions = out_actions_distribution.at(1);
    second_actions.reserve(N_ACTIONS);
    for (int i = 0; i < N_ACTIONS; i++)
    {
        float value = actions_distribution.at(miguyugo_syms::SECOND_ACTIONS_SYM.at(i));
        second_actions.emplace_back(value);
    }

    // add third sym
    out_syms.push_back({});
    std::vector<float>& third_obs = out_syms.at(2);
    third_obs.reserve(obs.size());
    for (int i = 0; i < CHANNELS * ROWS * COLS; i++)
    {
        float value = obs.at(miguyugo_syms::THIRD_OBS_SYM.at(i));
        third_obs.emplace_back(value);
    }

    // add third action sym
    out_actions_distribution.push_back({});
    std::vector<float>& third_actions = out_actions_distribution.at(2);
    third_actions.reserve(N_ACTIONS);
    for (int i = 0; i < N_ACTIONS; i++)
    {
        float value = actions_distribution.at(miguyugo_syms::THIRD_ACTIONS_SYM.at(i));
        third_actions.emplace_back(value);
    }
    ///////////

    // add horizontal sym
    out_syms.push_back({});
    std::vector<float>& horizontal_obs = out_syms.at(3);
    horizontal_obs.reserve(obs.size());
    for (int i = 0; i < CHANNELS * ROWS * COLS; i++)
    {
        float value = obs.at(miguyugo_syms::HORIZONTAL_OBS_SYM.at(i));
        horizontal_obs.emplace_back(value);
    }

    // add horizontal action sym
    out_actions_distribution.push_back({});
    std::vector<float>& horizontal_actions = out_actions_distribution.at(3);
    horizontal_actions.reserve(N_ACTIONS);
    for (int i = 0; i < N_ACTIONS; i++)
    {
        float value = actions_distribution.at(miguyugo_syms::HORIZONTAL_ACTIONS_SYM.at(i));
        horizontal_actions.emplace_back(value);
    }

    // add vertical sym
    out_syms.push_back({});
    std::vector<float>& vertical_obs = out_syms.at(4);
    vertical_obs.reserve(obs.size());
    for (int i = 0; i < CHANNELS * ROWS * COLS; i++)
    {
        float value = obs.at(miguyugo_syms::VERTICAL_OBS_SYM.at(i));
        vertical_obs.emplace_back(value);
    }

    // add vertical action sym
    out_actions_distribution.push_back({});
    std::vector<float>& vertical_actions = out_actions_distribution.at(4);
    vertical_actions.reserve(N_ACTIONS);
    for (int i = 0; i < N_ACTIONS; i++)
    {
        float value = actions_distribution.at(miguyugo_syms::VERTICAL_ACTIONS_SYM.at(i));
        vertical_actions.emplace_back(value);
    }

    // add main diagonal sym
    out_syms.push_back({});
    std::vector<float>& main_diagonal_obs = out_syms.at(5);
    main_diagonal_obs.reserve(obs.size());
    for (int i = 0; i < CHANNELS * ROWS * COLS; i++)
    {
        float value = obs.at(miguyugo_syms::MAIN_DIAGONAL_OBS_SYM.at(i));
        main_diagonal_obs.emplace_back(value);
    }

    // add main diagonal action sym
    out_actions_distribution.push_back({});
    std::vector<float>& main_diagonal_actions = out_actions_distribution.at(5);
    main_diagonal_actions.reserve(N_ACTIONS);
    for (int i = 0; i < N_ACTIONS; i++)
    {
        float value = actions_distribution.at(miguyugo_syms::MAIN_DIAGONAL_ACTIONS_SYM.at(i));
        main_diagonal_actions.emplace_back(value);
    }

    // add anti diagonal sym
    out_syms.push_back({});
    std::vector<float>& anti_diagonal_obs = out_syms.at(6);
    anti_diagonal_obs.reserve(obs.size());
    for (int i = 0; i < CHANNELS * ROWS * COLS; i++)
    {
        float value = obs.at(miguyugo_syms::ANTI_DIAGONAL_OBS_SYM.at(i));
        anti_diagonal_obs.emplace_back(value);
    }

    // add anti diagonal action sym
    out_actions_distribution.push_back({});
    std::vector<float>& anti_diagonal_actions = out_actions_distribution.at(6);
    anti_diagonal_actions.reserve(N_ACTIONS);
    for (int i = 0; i < N_ACTIONS; i++)
    {
        float value = actions_distribution.at(miguyugo_syms::ANTI_DIAGONAL_ACTIONS_SYM.at(i));
        anti_diagonal_actions.emplace_back(value);
    }
}


std::vector<float> MigoyugoState::get_observation() const
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

int MigoyugoState::get_last_action()const {
    return last_action_;
}
}
