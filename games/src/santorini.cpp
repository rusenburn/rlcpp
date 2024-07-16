#include <games/santorini.hpp>
#include <common/exceptions.hpp>
#include <sstream>
#include <iostream>
#include <array>
namespace rl::games
{

    SantoriniState::SantoriniState(const Board &players, const Board &buildings, SantoriniPhase current_phase, bool is_winning_move, int turn, int current_player, std::optional<std::pair<int, int>> selection)
        : players_(players),
          buildings_(buildings),
          current_phase_{current_phase},
          is_winning_move_{is_winning_move},
          turn_{turn},
          current_player_{current_player},
          selection_(selection)
    {
    }

    SantoriniState::~SantoriniState() = default;

    std::unique_ptr<SantoriniState> SantoriniState::initialize_state()
    {
        Board players{};
        Board buildings{};

        SantoriniPhase current_phase = SantoriniPhase::placement;
        int turn = 1;
        bool is_winning_move = false;
        int starting_player = 0;
        std::optional<std::pair<int, int>> selection{};
        return std::make_unique<SantoriniState>(players, buildings, current_phase, is_winning_move, turn, starting_player, selection);
    }

    std::unique_ptr<rl::common::IState> SantoriniState::initialize()
    {
        return SantoriniState::initialize_state();
    }

    std::unique_ptr<rl::common::IState> SantoriniState::reset() const
    {
        return SantoriniState::initialize();
    }

    std::unique_ptr<SantoriniState> SantoriniState::reset_state() const
    {
        return SantoriniState::initialize_state();
    }

    std::unique_ptr<rl::common::IState> SantoriniState::step(int action) const
    {
        return step_state(action);
    }

    std::unique_ptr<SantoriniState> SantoriniState::step_state(int action) const
    {
        if (is_terminal())
        {
            std::stringstream ss;
            ss << "Stepping a terminal Santorini state";
            throw rl::common::SteppingTerminalStateException(ss.str());
        }

        if (actions_mask().at(action) == false)
        {
            std::stringstream ss;
            ss << "Stepping a santorini state with an illegal action " << action << "\n";
            auto am = actions_mask();
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

        auto [row, col] = decode_action(action);
        if (current_phase_ == SantoriniPhase::placement)
        {
            Board new_players(players_);
            if (current_player_ == 0)
            {
                new_players.at(row).at(col) = 1;
            }
            else
            {
                new_players.at(row).at(col) = -1;
            }
            int next_player = 1 - current_player_;
            int next_turn = turn_ + 1;
            SantoriniPhase next_phase = SantoriniPhase::placement;
            if (next_turn > 4)
            {
                next_phase = SantoriniPhase::selection;
            }
            std::optional<std::pair<int, int>> no_selection;
            return std::make_unique<SantoriniState>(new_players, buildings_, next_phase, false, next_turn, next_player, no_selection);
        }
        else if (current_phase_ == SantoriniPhase::selection)
        {
            std::optional<std::pair<int, int>> new_selection(std::make_pair(row, col));
            SantoriniPhase next_phase = SantoriniPhase::moving;
            return std::make_unique<SantoriniState>(players_, buildings_, next_phase, false, turn_, current_player_, new_selection);
        }
        else if (current_phase_ == SantoriniPhase::moving)
        {
            if (!selection_.has_value())
            {
                throw rl::common::UnreachableCodeException("Santorini state assertion failed , moving with no selection");
            }
            Board new_players(players_);
            auto [prev_row, prev_col] = selection_.value();
            new_players.at(prev_row).at(prev_col) = 0;
            new_players.at(row).at(col) = current_player_ == 0 ? 1 : -1;
            bool is_winning_move = false;
            if (buildings_.at(row).at(col) == 3 && buildings_.at(prev_row).at(prev_col) < 3)
            {
                is_winning_move = true;
            }
            SantoriniPhase next_phase(SantoriniPhase::building);
            std::optional<std::pair<int, int>> new_selection(std::make_pair(row, col));
            return std::make_unique<SantoriniState>(new_players, buildings_, next_phase, is_winning_move, turn_, current_player_, new_selection);
        }
        else if (current_phase_ == SantoriniPhase::building)
        {
            if (!selection_.has_value())
            {
                throw rl::common::UnreachableCodeException("Santorini state assertion failed , building with no selection");
            }
            Board new_buildings(buildings_);
            new_buildings.at(row).at(col)++;
            int next_turn = turn_ + 1;
            int next_player = 1 - current_player_;
            SantoriniPhase next_phase(SantoriniPhase::selection);
            std::optional<std::pair<int, int>> no_selection;
            return std::make_unique<SantoriniState>(players_, new_buildings, next_phase, false, next_turn, next_player, no_selection);
        }
        else
        {
            throw rl::common::UnreachableCodeException("Santorini state assertion failed");
        }
    }

    void SantoriniState::render() const
    {
        std::stringstream ss;
        int player_0_piece = 1;
        int player_1_piece = -1;
        int dome = 4;
        for (int row = 0; row < ROWS; row++)
        {
            for (int col = 0; col < COLS; col++)
            {

                int player_piece = players_.at(row).at(col);
                int building_height = buildings_.at(row).at(col);

                if (building_height == dome)
                {
                    ss << " WW ";
                }
                else
                {
                    ss << " ";
                    if (player_piece == 0)
                    {
                        ss << '.';
                    }
                    else
                    {
                        if (player_piece == player_0_piece)
                        {
                            if (selection_.has_value() && std::get<0>(selection_.value()) == row && std::get<1>(selection_.value()) == col)
                            {
                                ss << 'X';
                            }
                            else
                            {
                                ss << 'x';
                            }
                        }
                        else if (player_piece == player_1_piece)
                        {
                            if (selection_.has_value() && std::get<0>(selection_.value()) == row && std::get<1>(selection_.value()) == col)
                            {
                                ss << 'O';
                            }
                            else
                            {
                                ss << 'o';
                            }
                        }
                    }
                    ss << building_height << ' ';
                }
            }
            ss << '\n';
        }
        if (is_terminal())
        {
            ss << "Game has ended\n";
            // TODO add winner;
        }
        else
        {
            if (current_player_ == 0)
            {
                ss << "Player X ";
            }
            else
            {
                ss << "Player O ";
            }

            if (current_phase_ == SantoriniPhase::placement)
            {
                ss << "has to place a piece";
            }
            else if (current_phase_ == SantoriniPhase::selection)
            {
                ss << "has to select a piece";
            }
            else if (current_phase_ == SantoriniPhase::moving)
            {
                ss << "has to move a piece";
            }
            else if (current_phase_ == SantoriniPhase::building)
            {
                ss << "has to build";
            }

            ss << "\n";
        }

        std::cout << ss.str() << std::endl;
    }

    bool SantoriniState::is_terminal() const
    {
        if (cached_is_terminal_.has_value())
        {
            return cached_is_terminal_.value();
        }

        if (is_winning_move_)
        {
            cached_is_terminal_.emplace(true);
            return true;
        }
        auto am = actions_mask();
        for (bool is_legal : am)
        {
            if (is_legal)
            {
                return false;
            }
        }
        // No legal actions;
        return true;
    }

    float SantoriniState::get_reward() const
    {
        if (!is_terminal())
        {
            return 0.0f;
        }
        if (cached_result_.has_value())
        {
            return cached_result_.value();
        }
        if (is_winning_move_)
        {
            cached_result_.emplace(1.0f);
            return cached_result_.value();
        }

        if (has_legal_action() == false)
        {
            cached_result_.emplace(-1.0f);
            return cached_result_.value();
        }
        throw rl::common::UnreachableCodeException("Santorini state is terminal with no winner");
    }

    std::vector<float> SantoriniState::get_observation() const
    {
        if (cached_observation_.size())
        {
            return cached_observation_;
        }

        constexpr int CURRENT_PLAYER_CHANNEL = 0;
        constexpr int OPPONENT_PLAYER_CHANNEL = 1;
        constexpr int SELECTION_CHANNEL = 2;
        constexpr int ZERO_HEIGHT_CHANNEL = 3;
        constexpr int PLACEMENT_PHASE_CHANNEL = 8;
        constexpr int CHANNEL_SIZE = ROWS * COLS;

        cached_observation_.resize(CHANNELS * ROWS * COLS);

        if (selection_.has_value())
        {
            auto [row, col] = selection_.value();
            int observation_cell_id = CHANNEL_SIZE * SELECTION_CHANNEL + row * COLS + col;
            cached_observation_.at(observation_cell_id) = 1.0f;
        }
        for (int row = 0; row < ROWS; row++)
        {
            for (int col = 0; col < COLS; col++)
            {
                int player_flag = players_.at(row).at(col);
                if (current_player_ != 0)
                {
                    player_flag = -player_flag;
                }
                if (player_flag == 1) // current_player
                {
                    int channel = CURRENT_PLAYER_CHANNEL;
                    int observation_cell_id = CHANNEL_SIZE * channel + row * COLS + col;
                    cached_observation_.at(observation_cell_id) = 1.0f;
                }
                else if (player_flag == -1)
                {
                    int channel = OPPONENT_PLAYER_CHANNEL;
                    int observation_cell_id = CHANNEL_SIZE * channel + row * COLS + col;
                    cached_observation_.at(observation_cell_id) = 1.0f;
                }
            }
        }

        // buildings/height observation
        for (int row = 0; row < ROWS; row++)
        {
            for (int col = 0; col < COLS; col++)
            {
                int height = buildings_.at(row).at(col);
                int channel = ZERO_HEIGHT_CHANNEL + height;
                int observation_cell_id = CHANNEL_SIZE * channel + row * COLS + col;
                cached_observation_.at(observation_cell_id) = 1.0f;
            }
        }

        // phases observation
        int phase = 0;
        if (current_phase_ == SantoriniPhase::placement)
        {
            phase = 0;
        }
        else if (current_phase_ == SantoriniPhase::selection)
        {
            phase = 1;
        }
        else if (current_phase_ == SantoriniPhase::moving)
        {
            phase = 2;
        }
        else if (current_phase_ == SantoriniPhase::building)
        {
            phase = 3;
        }
        int phase_channel = PLACEMENT_PHASE_CHANNEL + phase;
        int channel_start = CHANNEL_SIZE * phase_channel;
        int channel_end = channel_start + CHANNEL_SIZE;
        for (int cell = channel_start; cell < channel_end; cell++)
        {
            cached_observation_.at(cell) = 1.0f;
        }

        return cached_observation_;
    }

    std::string SantoriniState::to_short() const
    {
        std::stringstream ss;
        int player_0_piece = 1;
        int player_1_piece = -1;
        int dome = 4;
        for (int row = 0; row < ROWS; row++)
        {
            for (int col = 0; col < COLS; col++)
            {
                int player_piece = players_.at(row).at(col);
                if (player_piece == 0)
                {
                    ss << ' ';
                }
                else if (player_piece == player_0_piece)
                {
                    if (selection_.has_value() && std::get<0>(selection_.value()) == row && std::get<1>(selection_.value()) == col)
                    {
                        ss << 'X';
                    }
                    else
                    {
                        ss << 'x';
                    }
                }
                else if (player_piece == player_1_piece)
                {
                    if (selection_.has_value() && std::get<0>(selection_.value()) == row && std::get<1>(selection_.value()) == col)
                    {
                        ss << 'O';
                    }
                    else
                    {
                        ss << 'o';
                    }
                }
            }
            ss << "\n";
        }

        for (int row = 0; row < ROWS; row++)
        {
            for (int col = 0; col < COLS; col++)
            {
                int building_height = buildings_.at(row).at(col);
                ss << building_height;
            }
            ss << "\n";
        }
        ss << '#';
        if (current_player_ == 0)
        {
            ss << "x";
        }
        else
        {
            ss << "o";
        }
        ss << "#";
        if (current_phase_ == SantoriniPhase::placement)
        {
            ss << "P";
        }
        else if (current_phase_ == SantoriniPhase::selection)
        {
            ss << "S";
        }
        else if (current_phase_ == SantoriniPhase::moving)
        {
            ss << "M";
        }
        else if (current_phase_ == SantoriniPhase::building)
        {
            ss << "B";
        }

        ss << '#' << turn_;
        return ss.str();
    }
    std::array<int, 3> SantoriniState::get_observation_shape() const
    {
        // our pieces , opponent pieces ,selected piece, 5 heights , 4 phases
        return {CHANNELS, ROWS, COLS};
    }

    int SantoriniState::get_n_actions() const
    {
        return 5 * 5 + 1;
    }

    int SantoriniState::player_turn() const
    {
        return current_player_;
    }

    std::vector<bool> SantoriniState::actions_mask() const
    {
        if (cached_actions_masks_.size())
        {
            return cached_actions_masks_;
        }
        if (current_phase_ == SantoriniPhase::placement)
        {
            cached_actions_masks_.resize(get_n_actions(), false);

            for (int row = 0; row < ROWS; row++)
            {
                for (int col = 0; col < COLS; col++)
                {
                    if (players_.at(row).at(col) == 0)
                    {
                        int action = encode_action(row, col);
                        cached_actions_masks_.at(action) = true;
                    }
                }
            }
            return cached_actions_masks_;
        }
        else if (current_phase_ == SantoriniPhase::selection)
        {
            cached_actions_masks_.resize(get_n_actions(), false);
            for (int row = 0; row < ROWS; row++)
            {
                for (int col = 0; col < COLS; col++)
                {
                    int player_flag = players_.at(row).at(col);
                    if (current_player_ == 0 && player_flag == 1)
                    {
                        int action = encode_action(row, col);
                        cached_actions_masks_.at(action) = true;
                    }
                    else if (current_player_ == 1 && player_flag == -1)
                    {
                        int action = encode_action(row, col);
                        cached_actions_masks_.at(action) = true;
                    }
                }
            }
            return cached_actions_masks_;
        }
        else if (current_phase_ == SantoriniPhase::moving)
        {
            auto [selection_row, selection_col] = selection_.value();
            cached_actions_masks_.resize(get_n_actions(), false);
            for (int row = 0; row < ROWS; row++)
            {
                for (int col = 0; col < COLS; col++)
                {
                    if (abs(selection_row - row) < 2 && abs(selection_col - col) < 2)
                    {
                        if (selection_row == row && selection_col == col)
                        {
                            continue;
                        }
                        int current_height = buildings_.at(selection_row).at(selection_col);
                        int target_height = buildings_.at(row).at(col);
                        if (target_height - current_height > 1)
                        {
                            continue;
                        }
                        if (players_.at(row).at(col) == 0)
                        {
                            int action = encode_action(row, col);
                            cached_actions_masks_.at(action) = true;
                        }
                    }
                }
            }
            return cached_actions_masks_;
        }
        else if (current_phase_ == SantoriniPhase::building)
        {
            auto [selection_row, selection_col] = selection_.value();
            cached_actions_masks_.resize(get_n_actions(), false);
            for (int row = 0; row < ROWS; row++)
            {
                for (int col = 0; col < COLS; col++)
                {
                    if (abs(selection_row - row) < 2 && abs(selection_col - col) < 2)
                    {
                        if (selection_row == row && selection_col == col)
                        {
                            continue;
                        }
                        int target_height = buildings_.at(row).at(col);
                        if (target_height == 4)
                        {
                            continue;
                        }
                        if (players_.at(row).at(col) == 0)
                        {
                            int action = encode_action(row, col);
                            cached_actions_masks_.at(action) = true;
                        }
                    }
                }
            }
            return cached_actions_masks_;
        }
        throw rl::common::UnreachableCodeException("Santorini state assertion error in actions mask");
    }

    std::unique_ptr<SantoriniState> SantoriniState::clone_state() const
    {
        return std::unique_ptr<SantoriniState>(new SantoriniState(*this));
    }

    std::unique_ptr<rl::common::IState> SantoriniState::clone() const
    {
        return clone_state();
    }
    void SantoriniState::get_symmetrical_obs_and_actions(std::vector<float> const &obs, std::vector<float> const &actions_distribution, std::vector<std::vector<float>> &out_syms, std::vector<std::vector<float>> &out_actions_distribution) const
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
        std::vector<float> &first_obs = out_syms.at(0);
        first_obs.reserve(obs.size());
        for (int i = 0; i < CHANNELS * ROWS * COLS; i++)
        {
            float value = obs.at(santorini_syms::FIRST_OBS_SYM.at(i));
            first_obs.emplace_back(value);
        }

        // add first actions sym
        out_actions_distribution.push_back({});
        std::vector<float> &first_actions = out_actions_distribution.at(0);
        first_actions.reserve(N_ACTIONS);
        for (int i = 0; i < N_ACTIONS; i++)
        {
            float value = actions_distribution.at(santorini_syms::FIRST_ACTIONS_SYM.at(i));
            first_actions.emplace_back(value);
        }

        // add second sym
        out_syms.push_back({});
        std::vector<float> &second_obs = out_syms.at(1);
        second_obs.reserve(obs.size());
        for (int i = 0; i < CHANNELS * ROWS * COLS; i++)
        {
            float value = obs.at(santorini_syms::SECOND_OBS_SYM.at(i));
            second_obs.emplace_back(value);
        }

        // add second action sym
        out_actions_distribution.push_back({});
        std::vector<float> &second_actions = out_actions_distribution.at(1);
        second_actions.reserve(N_ACTIONS);
        for (int i = 0; i < N_ACTIONS; i++)
        {
            float value = actions_distribution.at(santorini_syms::SECOND_ACTIONS_SYM.at(i));
            second_actions.emplace_back(value);
        }

        // add third sym
        out_syms.push_back({});
        std::vector<float> &third_obs = out_syms.at(2);
        third_obs.reserve(obs.size());
        for (int i = 0; i < CHANNELS * ROWS * COLS; i++)
        {
            float value = obs.at(santorini_syms::THIRD_OBS_SYM.at(i));
            third_obs.emplace_back(value);
        }

        // add third action sym
        out_actions_distribution.push_back({});
        std::vector<float> &third_actions = out_actions_distribution.at(2);
        third_actions.reserve(N_ACTIONS);
        for (int i = 0; i < N_ACTIONS; i++)
        {
            float value = actions_distribution.at(santorini_syms::THIRD_ACTIONS_SYM.at(i));
            third_actions.emplace_back(value);
        }
    }

    SantoriniPhase SantoriniState::get_current_phase() const
    {
        return current_phase_;
    }
    std::pair<int, int> SantoriniState::decode_action(int action)
    {
        int row = action / COLS;
        int col = action % COLS;
        return std::make_pair(row, col);
    }

    int SantoriniState::encode_action(int row, int col)
    {
        int action = row * COLS + col;
        return action;
    }

    bool SantoriniState::has_legal_action() const
    {
        auto am = actions_mask();
        for (bool is_legal_action : am)
        {
            if (is_legal_action)
            {
                return true;
            }
        }
        return false;
    }

} // namespace rl::games
