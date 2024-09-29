#ifndef RL_SEARCH_TREES_CONCURRENT_AMCTS2_HPP_
#define RL_SEARCH_TREES_CONCURRENT_AMCTS2_HPP_


#include <players/concurrent_search_tree.hpp>
#include <players/bandits/amcts2/amcts2_node.hpp>
#include <players/evaluator.hpp>
namespace rl::players
{



class ConcurrentAmcts : public IConcurrentSearchTree
{

public:
    ConcurrentAmcts(int n_game_actions, std::unique_ptr<IEvaluator> evaluator_ptr, float cpuct, float temperature, int max_async_simulations_per_tree,float dirichlet_epsilon,float dirichlet_alpha, float default_visits, float default_wins);
    ~ConcurrentAmcts()override;
    std::vector<float> search(const rl::common::IState* state_ptr, int minimum_no_simulations, std::chrono::duration<int, std::milli> minimum_duration) override;
    std::pair<std::vector<std::vector<float>>,std::vector<float>> search_multiple(const std::vector<const rl::common::IState*>& state_ptrs, int minimum_sims,std::chrono::duration<int, std::milli> minimum_duration) override;
    void evaluate_collected_states(std::unique_ptr<Amcts2Node>& root_node_ptr,std::tuple<std::vector<float>, std::vector<float>>& evaluations_tuple,std::vector<std::pair<rl::common::IState*, std::vector<Amcts2Info>>>& tree_rollouts);


private:
    std::vector<std::unique_ptr<Amcts2Node>> root_nodes_;
    std::unique_ptr<IEvaluator> evaluator_ptr_;
    int n_game_actions_;
    float cpuct_;
    float temperature_;
    int max_async_simulations_;
    float dirichlet_epsilon_,dirichlet_alpha_;
    float default_n_, default_w_;
    void backpropogate(std::unique_ptr<Amcts2Node>& root_node_ptr,std::vector<Amcts2Info>& visited_path, float final_result, int final_player, std::vector<float>& probs);
};

} // namespace rl::players


#endif