#ifndef RL_SEARCH_TREES_AMCTS2_NODE_HPP_
#define RL_SEARCH_TREES_AMCTS2_NODE_HPP_

#include <memory>
#include <vector>

#include <common/state.hpp>
namespace rl::players
{
struct Amcts2Info
{
    int action;
    int player;
};

class Amcts2Node
{
public:
    Amcts2Node(std::unique_ptr<rl::common::IState> state_ptr,int n_game_actions,float cpuct);
    ~Amcts2Node();
    void simulate_once(std::pair<rl::common::IState* ,std::vector<Amcts2Info>>& rollout_info_ref,float dirichlet_epsilon,float dirichlet_alpha,float default_n,float default_w,Amcts2Node* const root_node_ptr);
    void backpropogate(std::vector<Amcts2Info>& visited_path,int depth, float final_result, int final_player, std::vector<float>& probs,float default_n,float default_w);
    void expand_node();
    std::vector<float> get_probs(float temperature);
    float get_evaluation();
private:
    std::unique_ptr<rl::common::IState> state_ptr_;
    int n_game_actions_;
    float cpuct_;
    std::vector<bool> actions_mask_{};
    std::vector<std::unique_ptr<Amcts2Node>> children_;
    std::vector<float> probs_{};
    std::vector<float> dirichlet_noise_{};
    float n_visits_{ 0 };
    float delta_wins{ 0 };
    std::vector<float> actions_visits_{ 0 };
    std::vector<float> delta_actions_wins{ 0 };
    int find_best_action(float dirichlet_epsilon , float dirichlet_alpha);

};



} // namespace rl::players





#endif