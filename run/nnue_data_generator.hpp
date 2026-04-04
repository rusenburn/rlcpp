#ifndef RL_TRAINING_NNUE_DATA_GENERATOR_HPP_
#define RL_TRAINING_NNUE_DATA_GENERATOR_HPP_

#include <memory>
#include <vector>
#include <string>
#include <fstream>
#include <common/state.hpp>
#include <deeplearning/network_evaluator.hpp>

namespace rl::training {

struct NNUESample {
    float score;
    std::vector<int16_t> active_features;
};

class NNUEDataGenerator {
public:
    NNUEDataGenerator(std::shared_ptr<rl::deeplearning::NetworkEvaluator> evaluator,
                      int n_concurrent_games = 1024);

    ~NNUEDataGenerator();

    /**
     * @brief Generates a training set using vectorized self-play.
     * @param initial_state The starting state template.
     * @param total_samples Goal number of samples to collect.
     * @param output_path Path to the binary output file.
     * @param temperature Controls exploration (1.0 = soft, 0.1 = nearly greedy).
     */
    void generate(const rl::common::IState& initial_state, 
                  int total_samples, 
                  const std::string& output_path,
                  float temperature = 1.0f);

private:
    void save_sample_binary(std::ofstream& out, float score, const std::vector<float>& obs);
    int sample_action(const std::vector<float>& probs, int game_idx, int n_actions, float temp);

    std::shared_ptr<rl::deeplearning::NetworkEvaluator> evaluator_ptr_;
    int n_concurrent_games_;
    int n_actions_;
};

} // namespace rl::training

#endif