#ifndef RL_NNUE_NNUE_MCTS_HPP_
#define RL_NNUE_NNUE_MCTS_HPP_

#include <optional>
#include <memory>
#include <games/migoyugo_light.hpp>
#include "nnue_model.hpp"
#include <chrono>
namespace rl::nnue
{
class MCTSNode
{
private:
    std::unique_ptr<rl::games::MigoyugoLightState> state_ptr_;
    int n_game_actions_;
    float cpuct_;
    std::array<int16_t, 256> acc_w_{};
    std::array<int16_t, 256> acc_b_{};
    std::vector<int> actions_mask_;
    std::vector<int> priority_;
    std::vector<std::unique_ptr<MCTSNode>> children_;
    std::vector<float> probs_;
    int n_visits{ 0 };
    std::vector<float> actions_visits_;
    std::vector<float> delta_wins_;
    std::optional<bool> is_terminal_;
    std::optional<float> game_result_;
    std::pair<float, int> search(const NNUEModel& model);
    std::vector<float> get_probs(float temperature);
    int get_best_action();
    static float evaluate_nnue_simd(const std::array<int16_t, 256>& accumulator, const NNUEModel& model);
    static void apply_update(const NNUEModel& model, std::array<int16_t, 256>& aw, std::array<int16_t, 256>& ab, const rl::games::NNUEUpdate& u);
public:
    MCTSNode(std::unique_ptr<rl::games::MigoyugoLightState> state_ptr, int n_game_actions, float cpuct, const std::array<int16_t, 256>& acc_w,
        const std::array<int16_t, 256>& acc_b);
    ~MCTSNode();
    std::vector<float> search_and_get_probs(const NNUEModel& model, int n_sims, std::chrono::duration<int, std::milli> minimum_duration, float temperature);
    float get_evaluation() const;
};
class MCTS
{

private:
    float cpuct_;
    float temperature_;
    int n_game_actions_;
    static void apply_update(const NNUEModel& model, std::array<int16_t, 256>& aw, std::array<int16_t, 256>& ab, const rl::games::NNUEUpdate& u);
public:
    MCTS(int n_game_actions, float cpuct, float temperature);
    ~MCTS();
    std::vector<float> search(const rl::games::MigoyugoLightState* state_ptr, const NNUEModel& model, int minimum_no_simulations, std::chrono::duration<int, std::milli> minimum_duration);
};

} // namespace rl::searchTrees

#endif
