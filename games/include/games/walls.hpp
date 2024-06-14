#ifndef RL_GAMES_WALLS_HPP_
#define RL_GAMES_WALLS_HPP_

#include <common/state.hpp>
#include <optional>
#include <array>
#include <tuple>

namespace rl::games
{
    class WallsState : public rl::common::IState
    {
    private:
        using Walls = std::array<std::array<int, 7>, 7>;
        using Positions = std::array<std::array<int, 2>, 2>;
        static constexpr int ROWS{7};
        static constexpr int COLS{7};
        static constexpr int CHANNELS{3};
        static constexpr int PLAYER_CHANNEL{0};
        static constexpr int OPPONENT_CHANNEL{1};
        static constexpr int WALLS_CHANNEL{2};
        constexpr static std::array<std::array<int, 2>, 8> DIRECTIONS{
            {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}}};
        static constexpr int N_DIRECTIONS{8};
        static constexpr int N_ACTIONS{ROWS * COLS * N_DIRECTIONS};

        int current_player_;
        Walls walls_;
        Positions positions_;

        // chache
        mutable std::vector<bool> cached_actions_masks_{};
        mutable std::optional<bool> cached_is_terminal_{};
        mutable std::optional<float> cached_result_{};
        mutable std::vector<float> cached_observation_{};

        static std::tuple<int, int, int, int> get_jump_row_col_and_build_row_col_from_action(int action);
        static bool is_valid_jump(int jump_row, int jump_col, int opponent_row, int opponent_col, const Walls &walls_ref_);
        static bool is_valid_build(int build_row, int build_col, int opponent_row, int opponent_col, const Walls &walls_ref_);
        static int encode_action(int jump_row, int jump_col, int a);

    public:
        WallsState(Walls walls, Positions positions, int current_player);
        ~WallsState() override;
        static std::unique_ptr<WallsState> initialize_state();
        static std::unique_ptr<rl::common::IState> initialize();
        std::unique_ptr<WallsState> reset_state() const;
        std::unique_ptr<rl::common::IState> reset() const override;

        std::unique_ptr<WallsState> step_state(int action) const;
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
        std::unique_ptr<WallsState> clone_state() const;
        std::unique_ptr<rl::common::IState> clone() const override;
    };
} // namespace rl::games

#endif