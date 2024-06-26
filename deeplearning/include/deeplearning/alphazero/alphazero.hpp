#ifndef RL_DEEPLEARNING_ALPHAZERO_HPP_
#define RL_DEEPLEARNING_ALPHAZERO_HPP_

#include <memory>
#include "networks/az.hpp"
#include <common/state.hpp>
#include "alphazero_sub_tree.hpp"

namespace rl::deeplearning::alphazero
{
    class AlphaZero
    {
    private:
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
        torch::DeviceType dev_;
        std::string load_path_;
        std::string save_name_;

        int choose_action(std::vector<float> &probs);
        void train_network(std::vector<float> &observations, std::vector<float> &probabilities, std::vector<float> &wdls);
        static torch::Tensor cross_entropy_loss_(torch::Tensor &target, torch::Tensor &prediction);
        void collect_data(std::vector<float> &observations_out, std::vector<float> &probabilities_out, std::vector<float> &wdls_out,
                                 std::vector<std::unique_ptr<rl::common::IState>> &states_ptrs,
                                 std::vector<std::unique_ptr<AmctsSubTree>> &subtrees,
                                 std::vector<std::vector<float>> &env_obsevations,
                                 std::vector<std::vector<float>> &env_probs,
                                 std::vector<std::vector<float>> &env_wdls,
                                 std::vector<std::vector<int>> &env_players);
        std::unique_ptr<AmctsSubTree> get_new_subtree_ptr();
        

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
            std::string load_path = "",
            std::string save_name = "temp.pt");
        void train();
        void execute_episode(std::vector<float> &observations, std::vector<float> &probabilities, std::vector<float> &wdls, bool is_use_network);
        ~AlphaZero();
    };

} // namespace rl::DeepLearning

#endif
