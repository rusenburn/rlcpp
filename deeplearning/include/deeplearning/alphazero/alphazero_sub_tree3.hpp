#ifndef RL_DEEPLEARNING_AMCTS_SUBTREE3_HPP_
#define RL_DEEPLEARNING_AMCTS_SUBTREE3_HPP_

#include <vector>
#include <memory>
#include <common/state.hpp>
#include <optional>
#include <utility>
namespace rl::deeplearning
{
    struct AmctsInfo3
    {
        int action;
        int player;
    };
    class SubTreeNode
    {
        std::unique_ptr<rl::common::IState> state_ptr_;
        int n_game_actions_;
        float cpuct_;
        std::vector<bool> actions_mask_{};
        std::vector<std::unique_ptr<SubTreeNode>> children_;
        std::vector<float> probs_{};
        std::vector<float> dirichlet_noise_{};
        float n_visits_{0};
        float delta_wins{0};
        std::vector<float> actions_visits_{0};
        std::vector<float> delta_actions_wins{0};
        int find_best_action(bool use_dirichlet_noise);

        public:
        SubTreeNode(std::unique_ptr<rl::common::IState> state_ptr,int n_game_actions,float cpuct);
        ~SubTreeNode();
        void simulate_once(std::pair<rl::common::IState* ,std::vector<AmctsInfo3>> &rollout_info,bool use_dirichlet_noise,float default_n,float default_w);
        void backpropogate(std::vector<AmctsInfo3>& visited_path,int depth, float final_result, int final_player, std::vector<float>& probs,float default_n,float default_w);
        void expand_node();
        std::vector<float> get_probs(float temperature);
        float get_evaluation();
    };
    class AmctsSubTree3
    {
        private:
        std::unique_ptr<SubTreeNode> root_node_;
        int n_game_actions_;
        float default_n_, default_w_;
        float cpuct_;
        float temperature_; 
        std::vector<std::pair<rl::common::IState* , std::vector<AmctsInfo3>>> rollouts_{};
        void backpropogate(std::vector<AmctsInfo3>& visited_path, float final_result, int final_player , std::vector<float> &probs);
        public:
        AmctsSubTree3(int n_game_actions, float cpuct, float temperature, float default_visits, float default_wins);
        ~AmctsSubTree3();
        void set_root(const rl::common::IState* state_ptr);
        void roll();
        std::vector<const rl::common::IState*> get_rollouts();
        void evaluate_collected_states(std::tuple<std::vector<float>, std::vector<float>>& evaluations_tuple);
        std::vector<float> get_probs();
        float get_evaluation();
        void clear_rollouts();
    };
} // namespace rl::deeplearning


#endif