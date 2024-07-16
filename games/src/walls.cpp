#include <iostream>
#include <sstream>

#include <games/walls.hpp>
#include <common/exceptions.hpp>

namespace rl::games
{

    WallsState::WallsState(Walls walls, Positions positions, int current_player)
        : walls_(walls),
          positions_(positions),
          current_player_{current_player},
          cached_actions_masks_{}
    {
    }

    WallsState::~WallsState() = default;

    std::unique_ptr<WallsState> WallsState::initialize_state()
    {
        Walls walls{};
        Positions pos{{{6, 3}, {0, 3}}};
        int current_player = 0;
        return std::make_unique<WallsState>(walls, pos, current_player);
    }

    std::unique_ptr<rl::common::IState> WallsState::initialize()
    {
        return initialize_state();
    }

    std::unique_ptr<WallsState> WallsState::reset_state() const
    {
        return initialize_state();
    }

    std::unique_ptr<rl::common::IState> WallsState::reset() const
    {
        return initialize();
    }

    std::unique_ptr<WallsState> WallsState::step_state(int action) const
    {
        if (is_terminal())
        {
            std::stringstream ss;
            ss << "Stepping a terminal Walls state";
            throw rl::common::SteppingTerminalStateException(ss.str());
        }
        if (actions_mask().at(action) == false)
        {
            std::stringstream ss;
            ss << "Stepping an Walls state with an illegal action " << action;
            throw rl::common::IllegalActionException(ss.str());
        }
        int player = current_player_;
        int opponent = 1 - player;
        auto [jump_row, jump_col, build_row, build_col] = get_jump_row_col_and_build_row_col_from_action(action);

        Walls new_walls{walls_};
        new_walls.at(build_row).at(build_col) = 1;
        auto opponent_position{std::get<1>(positions_)};
        Positions new_position{{opponent_position, {jump_row, jump_col}}};
        return std::make_unique<WallsState>(new_walls, new_position, opponent);
    }
    std::unique_ptr<rl::common::IState> WallsState::step(int action) const
    {
        return step_state(action);
    }

    void WallsState::render() const
    {
        std::stringstream ss;
        int player_0_row, player_0_col, player_1_row, player_1_col;
        if (current_player_ == 0)
        {
            player_0_row = positions_.at(0).at(0);
            player_0_col = positions_.at(0).at(1);
            player_1_row = positions_.at(1).at(0);
            player_1_col = positions_.at(1).at(1);
        }
        else
        {
            player_0_row = positions_.at(1).at(0);
            player_0_col = positions_.at(1).at(1);
            player_1_row = positions_.at(0).at(0);
            player_1_col = positions_.at(0).at(1);
        }

        for (int row = 0; row < ROWS; row++)
        {

            for (int col = 0; col < COLS; col++)
            {
                if (row == player_0_row && col == player_0_col)
                {
                    ss << " X ";
                }
                else if (row == player_1_row && col == player_1_col)
                {
                    ss << " O ";
                }
                else if (walls_.at(row).at(col) == 1)
                {
                    ss << " = ";
                }
                else
                {
                    ss << " . ";
                }
            }
            ss << "\n";
        }

        for (int col = 0; col < COLS; col++)
        {
            ss << "---";
        }
        ss << "\n";

        if (is_terminal())
        {
            std::string winner = current_player_ == 0 ? " O " : " X "; // if terminal then the other player has won

            ss << "Player " << winner << " Won!\n";
        }

        std::cout << ss.str() << std::endl;
    }

    bool WallsState::is_terminal() const
    {
        if (cached_is_terminal_.has_value())
        {
            return cached_is_terminal_.value();
        }

        auto actions_legality = actions_mask();
        bool has_legal_action = false;
        for (bool is_legal : actions_legality)
        {
            if (is_legal)
            {
                has_legal_action = true;
                break;
            }
        }
        cached_is_terminal_.emplace(!has_legal_action);
        return cached_is_terminal_.value();
    }

    float WallsState::get_reward() const
    {
        if (!is_terminal())
        {
            return 0.0f;
        }

        if (cached_result_.has_value())
        {
            return cached_result_.value();
        }

        auto actions_legality = actions_mask();
        bool has_legal_action = false;
        for (bool is_legal : actions_legality)
        {
            if (is_legal)
            {
                has_legal_action = true;
                break;
            }
        }

        if (has_legal_action == false)
        {
            cached_result_.emplace(-1.0f);
        }
        else
        {
            cached_result_.emplace(1.0f);
        }

        return cached_result_.value();

    } // namespace rl::games

    std::string WallsState::to_short() const
    {
        std::stringstream ss;
        int player_0_row, player_0_col, player_1_row, player_1_col;
        if (current_player_ == 0)
        {
            player_0_row = positions_.at(0).at(0);
            player_0_col = positions_.at(0).at(1);
            player_1_row = positions_.at(1).at(0);
            player_1_col = positions_.at(1).at(1);
        }
        else
        {
            player_0_row = positions_.at(1).at(0);
            player_0_col = positions_.at(1).at(1);
            player_1_row = positions_.at(0).at(0);
            player_1_col = positions_.at(0).at(1);
        }

        for (int row = 0; row < ROWS; row++)
        {

            for (int col = 0; col < COLS; col++)
            {
                if (row == player_0_row && col == player_0_col)
                {
                    ss << 'X';
                }
                else if (row == player_1_row && col == player_1_col)
                {
                    ss << 'O';
                }
                else if (walls_.at(row).at(col) == 1)
                {
                    ss << '=';
                }
                else
                {
                    ss << '.';
                }
            }
            ss << '\n';
        }

        ss << "#" << current_player_;
        return ss.str();
    }

    std::vector<float> WallsState::get_observation() const
    {
        if (cached_observation_.size())
        {
            return cached_observation_;
        }

        // Note : not threadsafe

        cached_observation_ = std::vector<float>(CHANNELS * ROWS * COLS);

        int player_row, player_col, opponent_row, opponent_col;

        // Note: positions has current player at position 0 index inside the array
        player_row = positions_.at(0).at(0);
        player_col = positions_.at(0).at(1);
        opponent_row = positions_.at(1).at(0);
        opponent_col = positions_.at(1).at(1);

        int player_index = PLAYER_CHANNEL * ROWS * COLS + player_row * COLS + player_col;
        cached_observation_[player_index] = 1;
        int opponent_index = OPPONENT_CHANNEL * ROWS * COLS + opponent_row * COLS + opponent_col;
        cached_observation_[opponent_index] = 1;

        int wall_index = WALLS_CHANNEL * ROWS * COLS;
        for (int row = 0; row < ROWS; row++)
        {
            for (int col = 0; col < COLS; col++)
            {
                cached_observation_.at(wall_index) = walls_[row][col];
                wall_index++;
            }
        }

        if (wall_index != ROWS * COLS * CHANNELS)
        {
            std::stringstream ss;
            ss << "something went wrong " << wall_index << "!= " << ROWS * COLS * CHANNELS;
            throw rl::common::UnreachableCodeException(ss.str());
        }
        return cached_observation_;
    }

    std::array<int, 3> WallsState::get_observation_shape() const
    {
        return {CHANNELS, ROWS, COLS};
    }

    int WallsState::get_n_actions() const
    {
        return N_ACTIONS;
    }

    int WallsState::player_turn() const
    {
        return current_player_;
    }

    std::vector<bool> WallsState::actions_mask() const
    {
        if (cached_actions_masks_.size())
        {
            return cached_actions_masks_;
        }

        int player = current_player_;
        int opponent = 1 - player;

        int player_row, player_col, opponent_row, opponent_col;
        // Note: positions has current player at position 0 index inside the array
        player_row = positions_.at(0).at(0);
        player_col = positions_.at(0).at(1);
        opponent_row = positions_.at(1).at(0);
        opponent_col = positions_.at(1).at(1);

        cached_actions_masks_.resize(N_ACTIONS, false);

        for (auto [row_dir, col_dir] : DIRECTIONS)
        {
            int jump_row = player_row + row_dir;
            int jump_col = player_col + col_dir;
            if (is_valid_jump(jump_row, jump_col, opponent_row, opponent_col, walls_))
            {
                for (int a = 0; a < DIRECTIONS.size(); a++)
                {
                    auto [build_row_dir, build_col_dir] = DIRECTIONS.at(a);
                    int build_row = jump_row + build_row_dir;
                    int build_col = jump_col + build_col_dir;
                    if (is_valid_build(build_row, build_col, opponent_row, opponent_col, walls_))
                    {
                        int action = encode_action(jump_row, jump_col, a);
                        action = jump_row * COLS * N_DIRECTIONS + jump_col * N_DIRECTIONS + a;
                        cached_actions_masks_[action] = true;
                    }
                }
            }
        }
        return cached_actions_masks_;
    }

    std::unique_ptr<WallsState> WallsState::clone_state() const
    {
        return std::unique_ptr<WallsState>(new WallsState(*this));
    }
    std::unique_ptr<rl::common::IState> WallsState::clone() const
    {
        return clone_state();
    }

    void WallsState::get_symmetrical_obs_and_actions(std::vector<float> const &obs, std::vector<float> const &actions_distribution, std::vector<std::vector<float>> &out_syms, std::vector<std::vector<float>> &out_actions_distribution) const
    {
        out_syms.clear();
        out_actions_distribution.clear();
    }

    void WallsState::get_valid_jumps(std::vector<std::vector<bool>> &valid_jumps_out,std::vector<std::vector<std::vector<std::pair<int,int>>>> & valid_builds)
    {
        int player = current_player_;
        int opponent = 1 - player;

        int player_row, player_col, opponent_row, opponent_col;
        // Note: positions has current player at position 0 index inside the array
        player_row = positions_.at(0).at(0);
        player_col = positions_.at(0).at(1);
        opponent_row = positions_.at(1).at(0);
        opponent_col = positions_.at(1).at(1);

        for (auto [row_dir, col_dir] : DIRECTIONS)
        {
            int jump_row = player_row + row_dir;
            int jump_col = player_col + col_dir;
            if (is_valid_jump(jump_row, jump_col, opponent_row, opponent_col, walls_))
            {
                for (int a = 0; a < DIRECTIONS.size(); a++)
                {
                    auto [build_row_dir, build_col_dir] = DIRECTIONS.at(a);
                    int build_row = jump_row + build_row_dir;
                    int build_col = jump_col + build_col_dir;
                    if (is_valid_build(build_row, build_col, opponent_row, opponent_col, walls_))
                    {
                        valid_jumps_out.at(jump_row).at(jump_col) = true;
                        valid_builds.at(jump_row).at(jump_col).push_back(std::make_pair(build_row,build_col));
                    }
                }
            }
        }
    }

    int WallsState::encode_action(int jump_row, int jump_col, int build_row, int build_col)
    {
        int row_direction = build_row - jump_row;
        int col_direction = build_col - jump_col;
        int a = 0;
        for(auto [row,col] : WallsState::DIRECTIONS)
        {
            if(row_direction == row && col_direction == col)
            {
                return encode_action(jump_row,jump_col,a);
            }
            a++;
        }
        throw rl::common::IllegalActionException("");
    }

    std::tuple<int, int, int, int> WallsState::get_jump_row_col_and_build_row_col_from_action(int action)
    {
        int direction_index = action % N_DIRECTIONS;
        int row = action / (N_DIRECTIONS * COLS);
        int col = action / (N_DIRECTIONS);
        col = col % COLS;

        auto [direction_row, direction_col] = DIRECTIONS.at(direction_index);
        int build_row = row + direction_row;
        int build_col = col + direction_col;
        return std::make_tuple(row, col, build_row, build_col);
    }

    bool WallsState::is_valid_jump(int jump_row, int jump_col, int opponent_row, int opponent_col, const Walls &walls_ref)
    {
        if (jump_row == opponent_row && jump_col == opponent_col)
        {
            return false;
        }
        if (jump_row < 0 || jump_row >= ROWS || jump_col < 0 || jump_col >= COLS)
        {
            return false;
        }
        if (walls_ref.at(jump_row).at(jump_col) == 1)
        {
            return false;
        }

        return true;
    }

    bool WallsState::is_valid_build(int build_row, int build_col, int opponent_row, int opponent_col, const Walls &walls_ref_)
    {
        return is_valid_jump(build_row, build_col, opponent_row, opponent_col, walls_ref_);
    }

    int WallsState::encode_action(int jump_row, int jump_col, int a)
    {
        int action = jump_row * COLS * N_DIRECTIONS + jump_col * N_DIRECTIONS + a;
        return action;
    }
}
