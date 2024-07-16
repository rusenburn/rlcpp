#ifndef RL_DEEPLEARNING_AMCTS_SUBTREE_HPP_
#define RL_DEEPLEARNING_AMCTS_SUBTREE_HPP_

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
    struct AmctsInfo
    {
        std::string state_short;
        int action;
        int player;
    };
    class AmctsSubTree
    {
    private:
        int n_game_actions_, max_async_simulations_;
        float default_n_, default_w_;
        float cpuct_;
        float temperature_;
        std::unordered_set<std::string> states_;
        std::unordered_map<std::string, std::vector<std::unique_ptr<rl::common::IState>>> edges_;
        std::unordered_map<std::string, float> ns_;
        std::unordered_map<std::string, float> ws_;
        std::unordered_map<std::string, std::vector<float>> nsa_;
        std::unordered_map<std::string, std::vector<float>> wsa_;
        std::unordered_map<std::string, std::vector<float>> psa_;
        std::unordered_map<std::string, std::vector<bool>> masks_;
        std::vector<std::tuple<const rl::common::IState *, std::vector<AmctsInfo>>> rollouts_;
        
        void simulate_once(const rl::common::IState *state_ptr, std::vector<AmctsInfo> &visited_path);
        void backpropogate(std::vector<AmctsInfo> &visited_path, float final_result, int final_player);
        void expand_state(const rl::common::IState *state_ptr, std::string &short_state);

        // TODO check if we can just simply add the visited_path reference
        void add_to_rollouts(const rl::common::IState *state_ptr, std::vector<AmctsInfo> visited_path);
        int find_best_action(std::string &short_state);

    public:
        AmctsSubTree(int n_game_actions, float cpuct, float temperature, float default_visits, float default_wins);
        void roll(const rl::common::IState *state_ptr);
        std::vector<const rl::common::IState *> get_rollouts();
        void evaluate_collected_states(std::tuple<std::vector<float>,std::vector<float>>& evaluations_tuple);
        std::vector<float> get_probs(const rl::common::IState *state_ptr);
        float get_evaluation(const rl::common::IState *state_ptr);
        ~AmctsSubTree();
    };

}

#endif