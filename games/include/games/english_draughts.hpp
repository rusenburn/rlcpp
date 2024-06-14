#ifndef RL_GAMES_ENGLISH_DRAUGHTS_HPP_
#define RL_GAMES_ENGLISH_DRAUGHTS_HPP_

#include <common/state.hpp>
#include <tuple>
#include <array>
#include <optional>

namespace rl::games
{
    class EnglishDraughtState : public rl::common::IState
    {
        struct RowColDirectionAction
        {
            int row{};
            int col{};
            int row_direction{};
            int col_direction{};
        };

    private:
        constexpr static int CHANNELS{7};
        constexpr static int ROWS{8};
        constexpr static int COLS{8};
        constexpr static int PLAYER_P_CHANNEL{0};
        constexpr static int PLAYER_K_CHANNEL{1};
        constexpr static int OPPONENT_P_CHANNEL{2};
        constexpr static int OPPONENT_K_CHANNEL{3};
        constexpr static int N_ACTIONS{ROWS * COLS / 2 * 4};
        constexpr static int MAX_NO_CAPTURE_ROUNDS{40};
        constexpr static std::array<std::array<int, 2>, 4> DIRECTIONS{
            {{-1, 1}, {-1, -1}, {1, 1}, {1, -1}}};
        constexpr static std::array<int, 2> PLAYERS_P_FLAGS{1, -1};
        constexpr static std::array<int, 2> PLAYERS_K_FLAGS{2, -2};

        using Board = std::array<std::array<int8_t, COLS>, ROWS>;

        std::array<std::array<int8_t, COLS>, ROWS> board_{};
        int n_no_capture_rounds_{};
        std::vector<int> last_jump_{};
        std::vector<bool> last_jump_actions_mask_{};
        int current_player_{};

        // cache
        mutable std::vector<bool> cached_actions_masks_;
        mutable std::optional<bool> cached_is_terminal_;
        mutable std::optional<float> cached_result_;
        mutable std::vector<float> cached_observation_;

        // private static methods
        static RowColDirectionAction get_row_col_direction_from_action_(int action);
        static bool is_king_(const Board &board, int row, int col);
        static bool is_empty_position_(const Board &board, int row, int col);
        static int encode_action_(int row, int col, int direction_id);
        static std::tuple<bool, bool> is_legal_is_capture_(const Board &board, int action, int player);
        static bool is_backward_move_(int row_direction, int col_direction, int player);
        static bool can_capture_(const Board &board, int row, int col, int row_direction, int col_direction, int player);

        // private methods
        bool is_opponent_win_() const;
        bool is_draw_() const;
        void add_no_capture_rounds_observation_(std::vector<float> &observation_out) const;
        void add_last_jump_observation_(std::vector<float> &observation_out) const;
        void add_current_player_turn_observation_(std::vector<float> &observation_out) const;

    public:
        EnglishDraughtState(std::array<std::array<int8_t, COLS>, ROWS> board, int n_no_capture_rounds, std::vector<bool> last_jump_actions_mask, std::vector<int> last_jump, int current_player);
        ~EnglishDraughtState() override;

        static std::unique_ptr<EnglishDraughtState> initialize_state();
        static std::unique_ptr<rl::common::IState> initialize();
        std::unique_ptr<EnglishDraughtState> reset_state()const;
        std::unique_ptr<rl::common::IState> reset()const override;

        std::unique_ptr<EnglishDraughtState> step_state(int action)const ;
        std::unique_ptr<rl::common::IState> step(int action)const override;

        void render()const override;
        bool is_terminal()const override;

        float get_reward()const override;

        std::vector<float> get_observation()const override;

        std::string to_short()const override;

        std::array<int,3> get_observation_shape()const override;

        int get_n_actions()const override;

        int player_turn()const override;
        std::vector<bool> actions_mask()const override;
        std::unique_ptr<EnglishDraughtState> clone_state()const ;
        std::unique_ptr<rl::common::IState> clone() const override;
    };

} // namespace rl::games

#endif