#ifndef RL_TRAINING_MCTS_NNUE_DATA_GENERATOR_HPP_
#define RL_TRAINING_MCTS_NNUE_DATA_GENERATOR_HPP_

#include <memory>
#include <vector>
#include <string>
#include <fstream>
#include <common/state.hpp>
#include <deeplearning/network_evaluator.hpp>
#include <players/bandits/amcts2/concurrent_amcts.hpp>

namespace rl::training {

// Value target used to label each collected position.
enum class ValueLabelMode {
    // The MCTS root's own value estimate at the time the position was
    // searched (players::ConcurrentAmcts's root evaluation, averaged over
    // n_simulations_per_move playouts). Lower variance than the final
    // outcome since it's already an average over many simulated futures -
    // closer in spirit to the smooth, easy-to-distill targets a raw
    // one-shot network pass produces, but search-refined.
    kRootValue,
    // The true final game result (like AlphaZero's z / WDL target),
    // collected once each self-play episode finishes and back-signed per
    // buffered position. Unbiased but high-variance per position, since a
    // single game's outcome depends on many subsequent moves by both
    // sides - can require far more data to average out than kRootValue.
    kFinalOutcome,
};

// Generates NNUE training examples by driving self-play with ConcurrentAmcts
// (batched tree search) instead of a single raw network pass, so both the
// move choices and the value labels come from stronger, more reliable play.
class MctsNNUEDataGenerator {
public:
    MctsNNUEDataGenerator(
        std::unique_ptr<rl::deeplearning::NetworkEvaluator> evaluator_ptr,
        int n_game_actions,
        int n_concurrent_games = 128,
        int n_simulations_per_move = 800,
        int max_async_simulations_per_tree = 8,
        float cpuct = 2.5f,
        float dirichlet_epsilon = 0.25f,
        float dirichlet_alpha = -1.0f,
        float default_visits = 1.0f,
        float default_wins = -1.0f,
        ValueLabelMode value_label_mode = ValueLabelMode::kRootValue);

    ~MctsNNUEDataGenerator();

    /**
     * @brief Generates a training set using batched MCTS self-play.
     * @param initial_state The starting state template.
     * @param total_samples Goal number of examples written to the output file
     *        (each of the 8 saved symmetric orientations of a position counts
     *        individually towards this total).
     * @param output_path Path to the binary output file.
     * @param temperature Controls exploration when sampling moves from the
     *        tree's visit distribution (1.0 = soft, 0.1 = nearly greedy).
     */
    void generate(const rl::common::IState& initial_state,
        int total_samples,
        const std::string& output_path,
        float temperature = 1.0f);

private:
    void save_sample_binary(std::ofstream& out, float score, const std::vector<float>& obs);
    int sample_action(const std::vector<float>& probs, float temp);
    std::unique_ptr<rl::common::IState> start_new_episode(const rl::common::IState& initial_state);

    std::unique_ptr<rl::players::ConcurrentAmcts> tree_ptr_;
    int n_game_actions_;
    int n_concurrent_games_;
    int n_simulations_per_move_;
    ValueLabelMode value_label_mode_;
};

} // namespace rl::training

#endif
