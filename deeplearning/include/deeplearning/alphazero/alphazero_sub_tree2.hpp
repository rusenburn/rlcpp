#ifndef RL_DEEPLEARNING_AMCTS_SUBTREE2_HPP_
#define RL_DEEPLEARNING_AMCTS_SUBTREE2_HPP_

#include <map>
#include <unordered_map>
#include <set>
#include <unordered_set>
#include <string>
#include <vector>
#include <memory>
#include <common/state.hpp>
namespace rl::deeplearning
{
struct MapValues
{
public:
    std::vector<std::unique_ptr<rl::common::IState>> edges;
    float ns{};
    float ws{};
    std::vector<float> nsa{};
    std::vector<float> wsa{};
    std::vector<float> psa{};
    std::vector<bool> masks{};
    std::vector<float> dirichlet_noise{};
};


struct AmctsInfo2
{
    std::string state_short;
    int action;
    int player;
};
class AmctsSubTree2
{
private:
    int n_game_actions_, max_async_simulations_;
    float default_n_, default_w_;
    float cpuct_;
    float temperature_;
    std::unordered_set<std::string> states_;
    std::unordered_map<std::string, MapValues> values_;
    std::vector<std::tuple<const rl::common::IState*, std::vector<AmctsInfo2>>> rollouts_;

    void simulate_once(const rl::common::IState* state_ptr, std::vector<AmctsInfo2>& visited_path, bool use_dirichlet_noise = false);
    void backpropogate(std::vector<AmctsInfo2>& visited_path, float final_result, int final_player);
    void expand_state(const rl::common::IState* state_ptr, std::string& short_state);

    // TODO check if we can just simply add the visited_path reference
    void add_to_rollouts(const rl::common::IState* state_ptr, std::vector<AmctsInfo2> visited_path);
    int find_best_action(MapValues& state_value, bool use_dirichlet_noise = false);

public:
    AmctsSubTree2(int n_game_actions, float cpuct, float temperature, float default_visits, float default_wins);
    void roll(const rl::common::IState* state_ptr);
    std::vector<const rl::common::IState*> get_rollouts();
    void evaluate_collected_states(std::tuple<std::vector<float>, std::vector<float>>& evaluations_tuple);
    std::vector<float> get_probs(const rl::common::IState* state_ptr);
    float get_evaluation(const rl::common::IState* state_ptr);
    ~AmctsSubTree2();
};

}

#endif