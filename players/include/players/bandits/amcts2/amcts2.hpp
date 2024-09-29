#ifndef RL_SEARCH_TREES_AMCTS2_HPP_
#define RL_SEARCH_TREES_AMCTS2_HPP_

#include <players/search_tree.hpp>
#include <players/bandits/amcts2/amcts2_node.hpp>
#include <players/evaluator.hpp>
namespace rl::players
{



class Amcts2 : public ISearchTree
{

public:
    Amcts2(int n_game_actions, std::unique_ptr<IEvaluator> evaluator_ptr, float cpuct, float temperature, int max_async_simulations, float dirichlet_epsilon,float dirichlet_alpha,float default_visits, float default_wins);
    ~Amcts2()override;
    std::vector<float> search(const rl::common::IState* state_ptr, int minimum_no_simulations, std::chrono::duration<int, std::milli> minimum_duration) override;

    void set_root(const rl::common::IState* state_ptr);
    void roll(float dirichlet_epsilon,float dirichlet_alpha);
    std::vector<const rl::common::IState*> get_rollouts();
    void evaluate_collected_states(std::tuple<std::vector<float>, std::vector<float>>& evaluations_tuple);
    std::vector<float> get_probs();
    float get_evaluation();
    void clear_rollout();

private:
    std::unique_ptr<Amcts2Node> root_node_;
    std::unique_ptr<IEvaluator> evaluator_ptr_;
    int n_game_actions_;
    float cpuct_;
    float temperature_;
    int max_async_simulations_;
    float dirichlet_epsilon_,dirichlet_alpha_;
    float default_n_, default_w_;
    std::vector<std::pair<rl::common::IState*, std::vector<Amcts2Info>>> rollouts_{};
    void backpropogate(std::vector<Amcts2Info>& visited_path, float final_result, int final_player, std::vector<float>& probs);
};

} // namespace rl::players




#endif