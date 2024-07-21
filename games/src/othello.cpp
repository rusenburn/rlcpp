#include <sstream>
#include <stdexcept>
#include <iomanip>
#include <iostream>
#include <games/othello.hpp>
#include <common/exceptions.hpp>

namespace rl::games
{
    OthelloState::OthelloState(std::array<std::array<std::array<int8_t, COLS>, ROWS>, N_PLAYERS> observation, int player, int consecutive_skips)
        : observation_(observation),
          current_player_(player),
          n_consecutive_skips_(consecutive_skips),
          actions_legality_(),
          is_terminal_cached_(false),
          is_terminal_(false),
          is_result_cached_(false),
          cached_result_(false)
    {
    }

    OthelloState::~OthelloState() = default;

    std::unique_ptr<OthelloState> OthelloState::initialize_state()
    {
        std::array<std::array<std::array<int8_t, COLS>, ROWS>, N_PLAYERS> obs;
        for (int p = 0; p < N_PLAYERS; p++)
        {
            for (int row = 0; row < ROWS; row++)
            {
                for (int col = 0; col < COLS; col++)
                {
                    obs[p][row][col] = 0;
                }
            }
        }
        obs[1][3][3] = 1;
        obs[1][4][4] = 1;
        obs[0][3][4] = 1;
        obs[0][4][3] = 1;
        int player_0 = 0;
        auto state_ptr = std::make_unique<OthelloState>(obs, player_0, 0);
        return state_ptr;
    }
    std::unique_ptr<rl::common::IState> OthelloState::initialize()
    {
        return initialize_state();
    }

    std::unique_ptr<rl::common::IState> OthelloState::reset() const
    {
        return reset_state();
    }

    std::unique_ptr<OthelloState> OthelloState::reset_state() const
    {
        return initialize_state();
    }

    std::unique_ptr<OthelloState> OthelloState::step_state(int action) const
    {
        if (is_terminal())
        {
            throw rl::common::SteppingTerminalStateException("Trying to step a terminal state");
        }
        actions_legality_ = actions_mask();
        bool action_legality = actions_legality_[action];
        if (action_legality == false)
        {
            std::stringstream ss;
            ss << "Trying to perform an illegal action of " << action;
            throw rl::common::IllegalActionException(ss.str());
        }

        int player = current_player_;
        int other = 1 - player;

        std::array<std::array<std::array<int8_t, COLS>, ROWS>, 2> new_obs(observation_);

        // check if action is a skip action which is 64
        constexpr int skip_action_number = ROWS * COLS;
        if (action == skip_action_number)
        {
            // return new state with an increased number to consecutive skips by 1
            return std::make_unique<OthelloState>(new_obs, other, n_consecutive_skips_ + 1);
        }

        int skips = 0;

        // turn action into (row,col) action
        int row_id = action / COLS;
        int col_id = action % COLS;

        std::vector<std::pair<int, int>> board_changes = this->get_board_changes_on_action(row_id, col_id);
        if (board_changes.size() == 0)
        {
            throw rl::common::UnreachableCodeException("Othello No tiles to flip after picking a non skip action which indicates an implementation error.");
        }

        for (auto &cell : board_changes)
        {
            // for each cell
            const int &cell_row = cell.first;
            const int &cell_col = cell.second;

            // let our player update his cell value
            new_obs[player][cell_row][cell_col] = 1;

            // let the other player remove the cell value
            new_obs[other][cell_row][cell_col] = 0;
        }

        new_obs[player][row_id][col_id] = 1;
        return std::make_unique<OthelloState>(new_obs, other, skips);
    }

    std::unique_ptr<rl::common::IState> OthelloState::step(int action) const
    {
        return step_state(action);
    }

    bool OthelloState::is_terminal() const
    {
        if (is_terminal_cached_)
        {
            return is_terminal_;
        }
        if (n_consecutive_skips_ == 2)
            return true;
        for (int row = 0; row < ROWS; row++)
        {
            for (int col = 0; col < COLS; col++)
            {
                if (!observation_[0][row][col] && !this->observation_[1][row][col])
                {
                    is_terminal_cached_ = true;
                    is_terminal_ = false;
                    return is_terminal_;
                }
            }
        }
        is_terminal_cached_ = true;
        is_terminal_ = true;
        return is_terminal_;
    }

    float OthelloState::get_reward() const
    {
        if (is_result_cached_)
        {
            return cached_result_;
        }
        std::vector<int> wdl = {0, 0, 0};
        int player = current_player_;
        int other = 1 - player;

        int player_score = 0;
        int other_score = 0;

        for (int i = 0; i < ROWS; i++)
        {
            for (int j = 0; j < COLS; j++)
            {
                player_score += this->observation_[player][i][j];
                other_score += this->observation_[other][i][j];
            }
        }

        if (player_score > other_score)
        {
            cached_result_ = 1.0f;
        }
        else if (player_score == other_score)
        {
            cached_result_ = 0.0f;
        }
        else
        {
            cached_result_ = -1.0f;
        }
        is_result_cached_ = true;
        return cached_result_;
    }

    std::vector<bool> OthelloState::actions_mask() const
    {
        if (!actions_legality_.empty())
        {
            return actions_legality_;
        }

        int player = current_player_;
        int other = 1 - player;

        // TODO PERFORMANCE
        actions_legality_.resize(COLS * ROWS + 1, 0); // the +1 is the skip action

        bool has_legal_action = false;
        for (int row = 0; row < ROWS; row++)
        {
            for (int col = 0; col < COLS; col++)
            {
                if (!get_board_changes_on_action(row, col).empty())
                {
                    actions_legality_[row * COLS + col] = true;
                    has_legal_action = true;
                }
            }
        }

        if (!has_legal_action)
        {
            // if there is no legal actions then a skip action is a valid action
            actions_legality_[ROWS * COLS] = true;
        }
        return actions_legality_;
    }
    std::vector<float> OthelloState::get_observation() const
    {
        int player = current_player_;
        std::vector<float> true_obs;
        true_obs.reserve(N_PLAYERS * ROWS * COLS);
        if (player == 0)
        {
            for (int channel{0}; channel < N_PLAYERS; channel++)
            {
                for (int row{0}; row < ROWS; row++)
                {
                    for (int col{0}; col < COLS; col++)
                    {
                        true_obs.emplace_back(float(observation_[channel][row][col]));
                    }
                }
            }
        }
        else
        { // reverse , start from player 2 then move to player 1
            for (int channel{N_PLAYERS - 1}; channel >= 0; channel--)
            {
                for (int row{0}; row < ROWS; row++)
                {
                    for (int col{0}; col < COLS; col++)
                    {
                        true_obs.emplace_back(float(observation_[channel][row][col]));
                    }
                }
            }
        }
        return true_obs;
    }

    std::string OthelloState::to_short() const
    {
        std::stringstream ss;
        int empty_count = 0;
        for (int row{0}; row < ROWS; row++)
        {
            for (int col{0}; col < COLS; col++)
            {
                if (observation_[0][row][col] == 0 && observation_[1][row][col] == 0)
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
                    if (observation_[0][row][col])
                    {
                        ss << 'x';
                    }
                    else if (observation_[1][row][col])
                    {
                        ss << 'o';
                    }
                    else
                    {
                        throw rl::common::UnreachableCodeException("Something is wrong with the othello observasion");
                    }
                }
            }
        }
        if (empty_count)
        {
            ss << empty_count;
            empty_count = 0;
        }
        ss << "#" << current_player_;
        return ss.str();
    }

    std::array<int, 3> OthelloState::get_observation_shape() const
    {
        return {N_PLAYERS, ROWS, COLS};
    }

    int OthelloState::get_n_actions() const
    {
        return ROWS * COLS + 1;
    }

    int OthelloState::player_turn() const
    {
        return current_player_;
    }

    void OthelloState::render() const
    {
        int player_x = 0;
        int player_o = 1 - player_x;
        int obs[ROWS][COLS];
        // combine current player with other player observations into one
        // by substraction
        for (int i = 0; i < ROWS; i++)
        {
            for (int j = 0; j < COLS; j++)
            {
                int player_x_cell = this->observation_[player_x][i][j];
                int player_o_cell = this->observation_[player_o][i][j];
                obs[i][j] = player_x_cell - player_o_cell;
            }
        }

        auto legal_actions = actions_mask();
        // transform legal_actions into 2d form
        int legal_actions_2d[ROWS][COLS];
        for (int row = 0; row < ROWS; row++)
        {
            for (int col = 0; col < COLS; col++)
            {
                int action = row * COLS + col;
                legal_actions_2d[row][col] = legal_actions[action];
            }
        }

        std::cout << "\n";
        std::cout << "   0  1  2  3  4  5  6  7\n";
        for (int i = 0; i < ROWS; i++)
        {
            std::cout << std::setw(3) << std::setfill(' ') << i * 8;
            std::cout << ' ';
            for (int j = 0; j < 8; j++)
            {
                std::string v = ".";
                int current_cell = obs[i][j];
                if (current_cell == 1)
                {
                    v = 'X';
                }
                else if (current_cell == -1)
                {
                    v = 'O';
                }
                if (legal_actions_2d[i][j] == 1)
                {
                    v = std::to_string((i * COLS + j));
                }
                std::cout << std::setw(3) << std::setfill(' ') << v;
            }
            std::cout << '\n';
        }
        char v = 'X';
        if (current_player_ == 1)
        {
            v = 'O';
        }
        std::cout << "\n#Player " << v << " Turn #" << std::endl;
    }

    bool OthelloState::is_board_action(int row, int col) const
    {
        return row < ROWS && row >= 0 && col < COLS && col >= 0;
    }

    std::vector<std::pair<int, int>> OthelloState::get_board_changes_on_action(int row, int col) const
    {
        if (!this->is_board_action(row, col) || this->observation_[0][row][col] != 0 || this->observation_[1][row][col] != 0)
        {
            return std::vector<std::pair<int, int>>();
        }

        int player = current_player_;
        int other = 1 - player;

        int8_t obs[ROWS][COLS];

        for (int i = 0; i < ROWS; i++)
        {
            for (int j = 0; j < COLS; j++)
            {
                int player_cell = this->observation_[player][i][j];
                int other_cell = this->observation_[other][i][j];
                obs[i][j] = player_cell - other_cell;
            }
        }

        obs[row][col] = 1;
        std::vector<std::pair<int, int>> tiles_to_flip;

        constexpr int8_t directions[8][2] = {
            {0, 1}, {1, 1}, {1, 0}, {1, -1}, {0, -1}, {-1, -1}, {-1, 0}, {-1, 1}};

        for (auto &direction : directions)
        {
            int row_direction = direction[0];
            int col_direction = direction[1];
            int x = row;
            int y = col;
            x += row_direction;
            y += col_direction;

            // check if there is an enemy cell next to this cell
            if (is_board_action(x, y) && obs[x][y] == -1)
            {
                // move to the next cell
                x += row_direction;
                y += col_direction;

                // check if it is not on board
                if (!is_board_action(x, y))
                {
                    // continue next direction
                    continue;
                }

                // while enemy cell move direction until you leave board or cell color changes
                while (obs[x][y] == -1)
                {
                    x += row_direction;
                    y += col_direction;
                    if (!is_board_action(x, y))
                    {
                        break;
                    }
                }
                // check last visited cell is not on board
                if (!is_board_action(x, y))
                {
                    continue;
                }
                // check if last visited cell is belong to our player ( current player )
                if (obs[x][y] == 1)
                {
                    while (true)
                    {
                        // move back until you reach starting cell
                        x -= row_direction;
                        y -= col_direction;
                        // check if this this the starting point
                        if (x == row && y == col)
                        {
                            break;
                        }
                        // else add current tile to tiles_to_flip
                        std::pair<int, int> current = std::make_pair(x, y);
                        tiles_to_flip.push_back(current);
                    }
                }
            }
        }
        obs[row][col] = 0;
        return tiles_to_flip;
    }

    std::unique_ptr<OthelloState> OthelloState::clone_state() const
    {
        return std::unique_ptr<OthelloState>(new OthelloState(*this));
    }
    std::unique_ptr<rl::common::IState> OthelloState::clone() const
    {
        return clone_state();
    }

    void OthelloState::get_symmetrical_obs_and_actions(std::vector<float> const &obs, std::vector<float> const &actions_distribution, std::vector<std::vector<float>> &out_syms, std::vector<std::vector<float>> &out_actions_distribution) const
    {
        out_syms.clear();
        out_actions_distribution.clear();
        if (obs.size() != N_PLAYERS * ROWS * COLS)
        {
            std::stringstream ss;
            ss << "get_symmetrical_obs_and_actions requires an observation with size of " << N_PLAYERS * ROWS * COLS;
            ss << " but a size of " << obs.size() << " was passed.";
            throw std::runtime_error(ss.str());
        }
        constexpr int N_ACTIONS = ROWS * COLS + 1;
        constexpr int CHANNELS = N_PLAYERS;
        // add first sym
        out_syms.push_back({});
        std::vector<float> &first_obs = out_syms.at(0);
        first_obs.reserve(obs.size());
        for (int i = 0; i < N_PLAYERS * ROWS * COLS; i++)
        {
            float value = obs.at(othello_syms::FIRST_OBS_SYM.at(i));
            first_obs.emplace_back(value);
        }

        // add first actions sym
        out_actions_distribution.push_back({});
        std::vector<float> &first_actions = out_actions_distribution.at(0);
        first_actions.reserve(N_ACTIONS);
        for (int i = 0; i < N_ACTIONS; i++)
        {
            float value = actions_distribution.at(othello_syms::FIRST_ACTIONS_SYM.at(i));
            first_actions.emplace_back(value);
        }

        // add second sym
        out_syms.push_back({});
        std::vector<float> &second_obs = out_syms.at(1);
        second_obs.reserve(obs.size());
        for (int i = 0; i < CHANNELS * ROWS * COLS; i++)
        {
            float value = obs.at(othello_syms::SECOND_OBS_SYM.at(i));
            second_obs.emplace_back(value);
        }

        // add second action sym
        out_actions_distribution.push_back({});
        std::vector<float> &second_actions = out_actions_distribution.at(1);
        second_actions.reserve(N_ACTIONS);
        for (int i = 0; i < N_ACTIONS; i++)
        {
            float value = actions_distribution.at(othello_syms::SECOND_ACTIONS_SYM.at(i));
            second_actions.emplace_back(value);
        }

        // add third sym
        out_syms.push_back({});
        std::vector<float> &third_obs = out_syms.at(2);
        third_obs.reserve(obs.size());
        for (int i = 0; i < CHANNELS * ROWS * COLS; i++)
        {
            float value = obs.at(othello_syms::THIRD_OBS_SYM.at(i));
            third_obs.emplace_back(value);
        }

        // add third action sym
        out_actions_distribution.push_back({});
        std::vector<float> &third_actions = out_actions_distribution.at(2);
        third_actions.reserve(N_ACTIONS);
        for (int i = 0; i < N_ACTIONS; i++)
        {
            float value = actions_distribution.at(othello_syms::THIRD_ACTIONS_SYM.at(i));
            third_actions.emplace_back(value);
        }
    }
} // namespace rl::games
