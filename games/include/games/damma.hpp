#ifndef RL_GAMES_DAMMA_HPP_
#define RL_GAMES_DAMMA_HPP_

#include <array>
#include <common/state.hpp>
#include <utility>
#include <optional>
#include <vector>
#include <memory>

/*
        -------------------------------------------------
        |  0  | 14  | 28  | 42  | 56  | 70  | 84  | 98  |
        -------------------------------------------------
        | 112 | 126 | 140 | 154 | 168 | 182 | 196 | 210 |
        -------------------------------------------------
        | 224 | 238 | 252 | 266 | 280 | 294 | 308 | 322 |
        -------------------------------------------------
        | 336 | 350 | 364 | 378 | 392 | 406 | 420 | 434 |
        -------------------------------------------------
        | 448 | 462 | 476 | 490 | 504 | 518 | 532 | 546 |
        -------------------------------------------------
        | 560 | 574 | 588 | 602 | 616 | 630 | 644 | 658 |
        -------------------------------------------------
        | 672 | 686 | 700 | 714 | 728 | 742 | 756 | 770 |
        -------------------------------------------------
        | 784 | 798 | 812 | 826 | 840 | 854 | 868 | 882 |


        -------------------------------------------------
        |     |     |     | 007 |     |     |     |     |
        -------------------------------------------------
        |     |     |     | 008 |     |     |     |     |
        -------------------------------------------------
        |     |     |     | 009 |     |     |     |     |
        -------------------------------------------------
        | 000 | 001 | 002 |  X  | 003 | 004 | 005 | 006 |
        -------------------------------------------------
        |     |     |     | 010 |     |     |     |     |
        -------------------------------------------------
        |     |     |     | 011 |     |     |     |     |
        -------------------------------------------------
        |     |     |     | 012 |     |     |     |     |
        -------------------------------------------------
        |     |     |     | 013 |     |     |     |     |
        -------------------------------------------------
        */

namespace rl::games
{
    class DammaState : public rl::common::IState
    {

    public:
        static constexpr int ROWS = 8;
        static constexpr int COLS = 8;
        static constexpr int CHANNELS = 6;
        static constexpr int N_ACTIONS = ROWS * COLS * (ROWS - 1 + COLS - 1);
        static constexpr int EMPTY_CELL = 0;
        static constexpr int PLAYER_P_CELL = 1;
        static constexpr int PLAYER_K_CELL = 2;
        static constexpr int OPPONENT_P_CELL = -1;
        static constexpr int OPPONENT_K_CELL = -2;
        static constexpr int MAX_NO_CAPTURE_ROUNDS = 40;

        using Board = std::array<std::array<int8_t, COLS>, ROWS>;
        static constexpr std::array<std::array<int, 2>, 4> DIRECTIONS{{
            {0, -1}, // LEFT
            {0, 1},  // RIGHT
            {-1, 0}, // UP
            {1, 0}   // DOWN
        }};

        DammaState(Board board,
                   int n_no_capture_rounds,
                   std::optional<std::pair<int, int>> last_jump,
                   std::vector<bool> last_jump_action_mask,
                   int current_player);
        ~DammaState() override;

        static std::unique_ptr<DammaState> initialize_state();
        static std::unique_ptr<rl::common::IState> initialize();
        std::unique_ptr<DammaState> reset_state() const;
        std::unique_ptr<rl::common::IState> reset() const override;

        std::unique_ptr<DammaState> step_state(int action) const;
        std::unique_ptr<rl::common::IState> step(int action) const override;

        void render() const override;
        bool is_terminal() const override;
        float get_reward() const override;

        std::vector<float> get_observation() const override;

        std::string to_short() const override;

        std::array<int, 3> get_observation_shape() const override;

        int get_n_actions() const override;

        int player_turn() const override;
        std::vector<bool> actions_mask() const override;
        std::unique_ptr<DammaState> clone_state() const;
        std::unique_ptr<rl::common::IState> clone() const override;
        static std::tuple<int, int, int, int> decode_action(int action);
        static int encode_action(int row, int col, int target_row, int target_col);
        static void swap_board_view(Board &board);

    private:
        Board board_{};
        int n_no_capture_rounds_{0};
        std::optional<std::pair<int, int>> last_jump_;
        std::vector<bool> last_jump_action_mask_;
        int current_player_;

        // cache
        mutable std::vector<bool> cached_actions_masks_;
        mutable std::optional<bool> cached_is_terminal_;
        mutable std::optional<float> cached_result_;
        mutable std::vector<float> cached_observation_;

        static bool assign_legal_action(Board &board, int row, int col, bool capture_only, std::vector<bool> &action_legality_no_capture_out, std::vector<bool> &action_legality_capture_out);
        bool is_opponent_win() const;
        bool is_draw() const;
        void add_no_capture_rounds_observation_(std::vector<float> &observation_out) const;
        void add_last_jump_observation_(std::vector<float> &observation_out) const;
        void add_current_player_turn_observation_(std::vector<float> &observation_out) const;
    };

} // namespace rl::games

#endif