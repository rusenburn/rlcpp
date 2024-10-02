#ifndef RL_DEEPLEARNING_ALPHAZERO_HPP_
#define RL_DEEPLEARNING_ALPHAZERO_HPP_

#include <memory>
#include <torch/torch.h>
#include <functional>
#include "networks/az.hpp"
#include <common/state.hpp>
#include <players/bandits/amcts2/amcts2.hpp>
#include <players/bandits/amcts2/concurrent_amcts.hpp>
#include "alphazero_config.hpp"
namespace rl::deeplearning::alphazero
{

class AlphaZero
{
private:
    // private constants
    // These are being used during evaluation, and not during data collection
    const int N_ASYNC = 8;
    const float N_VISITS = 1.0f;
    const float N_WINS = -1.0f;
    const float DIRICHLET_EPSILON = 0.25f;
    const float DIRICHLET_ALPHA = -1.0f;
    // Number of games/subtrees that run asynchronously , each has its own tree but all share the same evaluator, used during data collection
    const int N_TREES = 128;
    // Number of states to be collected per sub tree before evaluation
    const int N_SUB_TREE_ASYNC = 1;
    // Number of  games per iteration that cannot end early and must complete to end ( until terminal ) , the rest can skip when it is losing horribly
    const int N_COMPLETE_TO_END = N_TREES / 4;
    // Players that reached below this score can resign IF THEY ARE NOT COMPLETE TO END PLAYERS
    const float NO_RESIGN_THRESHOLD = -0.8f;
    // Players are not allowed to resign if the number of steps is below this number
    const int MINIMUM_STEPS = 30;
    // MCTS CPUCT
    const float CPUCT = 2.5f;

    std::unique_ptr<rl::common::IState> initial_state_ptr_;
    std::unique_ptr<rl::common::IState> test_state_ptr_;
    int n_iterations_;
    int n_episodes_;
    int n_sims_;
    float lr_;
    float critic_ceof_;
    int n_epoches_;
    int n_batches_;
    int n_testing_episodes_;
    int n_game_actions_;
    std::unique_ptr<IAlphazeroNetwork> base_network_ptr_;
    std::unique_ptr<IAlphazeroNetwork> tiny_network_ptr_;
    torch::DeviceType dev_;
    std::string load_path_;
    std::string save_name_;
    std::vector<float> all_observations_{};
    std::vector<float> all_probabilities_{};
    std::vector<float> all_wdls_{};
    std::vector<std::unique_ptr<rl::common::IState>> states_ptrs_{};
    std::vector<std::vector<float>> episode_obsevations_{};
    std::vector<std::vector<float>> episode_probs_{};
    std::vector<std::vector<float>> episode_wdls_{};
    std::vector<std::vector<int>> episode_players_{};
    std::vector<float> episode_steps_{};

    int choose_action(std::vector<float>& probs);
    void train_network(std::unique_ptr<IAlphazeroNetwork>& network, torch::optim::Optimizer& optimizer_ref, std::vector<float>& observations, std::vector<float>& probabilities, std::vector<float>& wdls);
    static torch::Tensor cross_entropy_loss_(torch::Tensor& target, torch::Tensor& prediction);
    void collect_data();
    void end_subtree(int subtree_id, int last_player, float result);
    std::unique_ptr<players::ConcurrentAmcts> AlphaZero::get_new_concurrent_tree_ptr();
    void initialize_subtrees();


public:
    AlphaZero(
        std::unique_ptr<rl::common::IState> initial_state_ptr,
        std::unique_ptr<rl::common::IState> test_state_ptr,
        int n_iterations,
        int n_episodes,
        int n_sims,
        int n_epochs,
        int n_batches,
        float lr,
        float critic_coef_,
        int n_testing_episodes,
        std::unique_ptr<IAlphazeroNetwork> network_ptr,
        std::unique_ptr<IAlphazeroNetwork> tiny_ptr,
        std::string load_path = "",
        std::string save_name = "temp.pt");
    AlphaZero(
        std::function<std::unique_ptr<rl::common::IState>()> initial_state_ptr_fn,
        std::function<std::unique_ptr<rl::common::IState>()> test_state_ptr_fn,
        std::unique_ptr<IAlphazeroNetwork> network_ptr,
        std::unique_ptr<IAlphazeroNetwork> tiny_ptr,
        AZConfig config
        );
    void train();
    ~AlphaZero();
};

} // namespace rl::DeepLearning

#endif
