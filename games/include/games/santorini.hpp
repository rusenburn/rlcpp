#ifndef RL_GAMES_SANTORINI_HPP_
#define RL_GAMES_SANTORINI_HPP_

#include <common/state.hpp>
#include <memory>
#include <array>
#include <vector>
#include <optional>
namespace rl::games
{
    enum class SantoriniPhase
    {
        placement,
        selection,
        moving,
        building,
    };
    class SantoriniState : public common::IState
    {

    public:
        constexpr static int ROWS = 5;
        constexpr static int COLS = 5;
        constexpr static int CHANNELS = 12;
        using Board = std::array<std::array<int8_t, COLS>, ROWS>;

        SantoriniState(const Board &players,
                       const Board &buildings,
                       SantoriniPhase current_phase,
                       bool is_winning_move,
                       int turn,
                       int current_player,
                       std::optional<std::pair<int, int>> selection);
        ~SantoriniState() override;
        static std::unique_ptr<SantoriniState> initialize_state();
        static std::unique_ptr<rl::common::IState> initialize();
        std::unique_ptr<rl::common::IState> reset() const override;
        std::unique_ptr<SantoriniState> reset_state() const;
        std::unique_ptr<rl::common::IState> step(int action) const override;
        std::unique_ptr<SantoriniState> step_state(int action) const;
        void render() const override;
        bool is_terminal() const override;
        float get_reward() const override;
        std::vector<float> get_observation() const override;
        std::string to_short() const override;
        std::array<int, 3> get_observation_shape() const override;
        int get_n_actions() const override;
        int player_turn() const override;
        std::vector<bool> actions_mask() const override;
        std::unique_ptr<SantoriniState> clone_state() const;
        std::unique_ptr<rl::common::IState> clone() const override;
        SantoriniPhase get_current_phase() const;
        static std::pair<int, int> decode_action(int action);
        static int encode_action(int row, int col);

    private:
        Board players_, buildings_;
        SantoriniPhase current_phase_;
        bool is_winning_move_;
        int current_player_;
        int turn_;
        std::optional<std::pair<int, int>> selection_;

        bool has_legal_action() const;

        // cache
        mutable std::vector<bool> cached_actions_masks_;
        mutable std::optional<bool> cached_is_terminal_;
        mutable std::optional<float> cached_result_;
        mutable std::vector<float> cached_observation_;
    };

} // namespace rl::games

#endif
