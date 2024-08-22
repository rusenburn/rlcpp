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
    constexpr static int N_ACTIONS = ROWS * COLS + 1;
    using Board = std::array<std::array<int8_t, COLS>, ROWS>;

    SantoriniState(const Board& players,
        const Board& buildings,
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
    void get_symmetrical_obs_and_actions(std::vector<float> const& obs, std::vector<float> const& actions_distribution, std::vector<std::vector<float>>& out_syms, std::vector<std::vector<float>>& out_actions_distribution)const override;
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

namespace rl::games::santorini_syms
{
/*
If you want to convert probabilities of an actions distribution then probs_b[i] = probs_a[b[i]]
where prob_a is the probabilties of the original distribution
prob_b probabilties of the sym observation that we need to find
b is one of the actions array represented below
*/
constexpr std::array<int, 300> FIRST_OBS_SYM =
{ {20, 15, 10, 5, 0, 21, 16, 11, 6, 1, 22, 17, 12, 7, 2,
  23, 18, 13, 8, 3, 24, 19, 14, 9, 4, 45, 40, 35, 30,
  25, 46, 41, 36, 31, 26, 47, 42, 37, 32, 27, 48, 43,
  38, 33, 28, 49, 44, 39, 34, 29, 70, 65, 60, 55, 50,
  71, 66, 61, 56, 51, 72, 67, 62, 57, 52, 73, 68, 63,
  58, 53, 74, 69, 64, 59, 54, 95, 90, 85, 80, 75, 96,
  91, 86, 81, 76, 97, 92, 87, 82, 77, 98, 93, 88, 83,
  78, 99, 94, 89, 84, 79, 120, 115, 110, 105, 100, 121, 116, 111,
  106, 101, 122, 117, 112, 107, 102, 123, 118, 113, 108, 103, 124, 119, 114, 109, 104, 145, 140, 135, 130, 125, 146, 141, 136,
  131, 126, 147, 142, 137, 132, 127, 148, 143, 138, 133, 128, 149, 144, 139, 134, 129, 170, 165, 160, 155, 150, 171, 166, 161,
  156, 151, 172, 167, 162, 157, 152, 173, 168, 163, 158, 153, 174, 169, 164, 159, 154, 195, 190, 185, 180, 175, 196, 191, 186,
  181, 176, 197, 192, 187, 182, 177, 198, 193, 188, 183, 178, 199, 194, 189, 184, 179, 220, 215, 210, 205, 200, 221, 216, 211,
  206, 201, 222, 217, 212, 207, 202, 223, 218, 213, 208, 203, 224, 219, 214, 209, 204, 245, 240, 235, 230, 225, 246, 241, 236, 231,
  226, 247, 242, 237, 232, 227, 248, 243, 238, 233, 228, 249, 244, 239, 234, 229, 270, 265, 260, 255, 250, 271, 266, 261, 256,
  251, 272, 267, 262, 257, 252, 273, 268, 263, 258, 253, 274, 269, 264, 259, 254, 295, 290, 285, 280, 275, 296,
  291, 286, 281, 276, 297, 292, 287, 282, 277, 298, 293, 288,
  283, 278, 299, 294, 289, 284, 279} };
constexpr std::array<int, rl::games::SantoriniState::N_ACTIONS> FIRST_ACTIONS_SYM =
{ {20, 15, 10, 5, 0, 21, 16, 11, 6, 1, 22, 17, 12, 7, 2, 23, 18, 13, 8, 3, 24, 19, 14, 9, 4, 25} };

constexpr std::array<int, 300> SECOND_OBS_SYM =
{ {24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 49,
  48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 74, 73,
  72, 71, 70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 99, 98, 97,
  96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80, 79, 78, 77, 76, 75, 124, 123, 122, 121,
  120, 119, 118, 117, 116, 115, 114, 113, 112, 111, 110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100, 149, 148, 147, 146, 145,
  144, 143, 142, 141, 140, 139, 138, 137, 136, 135, 134, 133, 132, 131, 130, 129, 128, 127, 126, 125, 174, 173, 172, 171, 170, 169,
  168, 167, 166, 165, 164, 163, 162, 161, 160, 159, 158, 157, 156, 155, 154, 153, 152, 151, 150, 199, 198, 197, 196, 195, 194, 193,
  192, 191, 190, 189, 188, 187, 186, 185, 184, 183, 182, 181, 180, 179, 178, 177, 176, 175, 224, 223, 222, 221, 220, 219, 218, 217,
  216, 215, 214, 213, 212, 211, 210, 209, 208, 207, 206, 205, 204, 203, 202, 201, 200, 249, 248, 247, 246, 245, 244, 243, 242, 241,
  240, 239, 238, 237, 236, 235, 234, 233, 232, 231, 230, 229, 228, 227, 226, 225, 274, 273, 272, 271, 270, 269, 268, 267, 266, 265,
  264, 263, 262, 261, 260, 259, 258, 257, 256, 255, 254, 253, 252, 251, 250, 299, 298, 297, 296, 295, 294, 293, 292, 291, 290, 289,
  288, 287, 286, 285, 284, 283, 282, 281, 280, 279, 278, 277, 276, 275} };

constexpr std::array<int, rl::games::SantoriniState::N_ACTIONS> SECOND_ACTIONS_SYM =
{ {24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 25} };

constexpr std::array<int, 300> THIRD_OBS_SYM =
{ {4, 9, 14, 19, 24, 3, 8, 13, 18, 23, 2, 7, 12, 17, 22, 1, 6, 11, 16, 21, 0, 5, 10, 15, 20, 29, 34, 39, 44, 49, 28, 33, 38,
  43, 48, 27, 32, 37, 42, 47, 26, 31, 36, 41, 46, 25, 30, 35, 40, 45, 54, 59, 64, 69, 74, 53, 58, 63, 68, 73, 52, 57, 62, 67, 72, 51,
  56, 61, 66, 71, 50, 55, 60, 65, 70, 79, 84, 89, 94, 99, 78, 83, 88, 93, 98, 77, 82, 87, 92, 97, 76, 81, 86, 91, 96, 75, 80, 85, 90,
  95, 104, 109, 114, 119, 124, 103, 108, 113, 118, 123, 102, 107, 112, 117, 122, 101, 106, 111, 116, 121, 100, 105, 110, 115, 120, 129, 134, 139, 144, 149, 128, 133,
  138, 143, 148, 127, 132, 137, 142, 147, 126, 131, 136, 141, 146, 125, 130, 135, 140, 145, 154, 159, 164, 169, 174, 153, 158, 163, 168, 173, 152, 157, 162, 167, 172,
  151, 156, 161, 166, 171, 150, 155, 160, 165, 170, 179, 184, 189, 194, 199, 178, 183, 188, 193, 198, 177, 182, 187, 192, 197, 176, 181, 186, 191, 196, 175, 180, 185,
  190, 195, 204, 209, 214, 219, 224, 203, 208, 213, 218, 223, 202, 207, 212, 217, 222, 201, 206, 211, 216, 221, 200, 205, 210, 215, 220, 229, 234, 239, 244, 249, 228,
  233, 238, 243, 248, 227, 232, 237, 242, 247, 226, 231, 236, 241, 246, 225, 230, 235, 240, 245, 254, 259, 264, 269, 274, 253, 258, 263, 268, 273, 252, 257, 262, 267,
  272, 251, 256, 261, 266, 271, 250, 255, 260, 265, 270, 279, 284, 289, 294, 299, 278, 283, 288, 293, 298, 277, 282, 287, 292, 297, 276, 281, 286, 291, 296, 275, 280,
  285, 290, 295} };

constexpr std::array<int, rl::games::SantoriniState::N_ACTIONS> THIRD_ACTIONS_SYM =
{ {4, 9, 14, 19, 24, 3, 8, 13, 18, 23, 2, 7, 12, 17, 22, 1, 6, 11, 16, 21, 0, 5, 10, 15, 20, 25} };
} // namespace rl::games::santorini_rotations

#endif
