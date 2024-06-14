#ifndef RL_SEARCH_TREES_AMCTS_HPP_
#define RL_SEARCH_TREES_AMCTS_HPP_

#include <map>
#include <set>
#include <string>
#include "evaluator.hpp"
#include "search_tree.hpp"

namespace rl::players
{
    struct AmctsInfo
    {
        std::string state_short;
        int action;
        int player;
    };
    class Amcts : public ISearchTree
    {
    private:
        int n_game_actions_, max_async_simulations_;
        float default_n_, default_w_;
        std::unique_ptr<IEvaluator> evaluator_ptr_;
        float cpuct_;
        float temperature_;
        std::set<std::string> states_;
        std::map<std::string, std::vector<std::unique_ptr<rl::common::IState>>> edges_;
        std::map<std::string, float> ns_;
        std::map<std::string, std::vector<float>> nsa_;
        std::map<std::string, std::vector<float>> wsa_;
        std::map<std::string, std::vector<float>> psa_;
        std::map<std::string, std::vector<bool>> masks_;
        rl::common::IState *root_ptr;
        int root_player_;
        std::vector<std::tuple<const rl::common::IState *, std::vector<AmctsInfo>>> rollouts_;
        std::vector<float> search_root(const rl::common::IState* root_ptr,int minimum_no_simulations, std::chrono::duration<int, std::milli> minimum_duration);
        void evaluate_collected_states();
        void simulate_once(const rl::common::IState *state_ptr, std::vector<AmctsInfo> &visited_path);
        void backpropogate(std::vector<AmctsInfo> &visited_path, float final_result, int final_player);
        void expand_state(const rl::common::IState *state_ptr, std::string &short_state);

        // TODO check if we can just simply add the visited_path reference
        void add_to_rollouts(const rl::common::IState *state_ptr, std::vector<AmctsInfo> visited_path);
        int find_best_action(std::string &short_state);
        std::vector<float> get_probs(const rl::common::IState* state_ptr);

    public:
        Amcts(int n_game_actions, std::unique_ptr<IEvaluator> evaluator_ptr_, float cpuct, float temperature, int max_async_simulations, float default_visits, float default_wins);
        std::vector<float> search(const rl::common::IState *state_ptr, int minimum_no_simulations, std::chrono::duration<int, std::milli> minimum_duration)override;
        ~Amcts()override;
    };

    

} // namespace rl::search_trees

#endif