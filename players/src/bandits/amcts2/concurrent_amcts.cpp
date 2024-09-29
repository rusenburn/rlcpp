#include <cassert>
#include <stdexcept>
#include <players/bandits/amcts2/concurrent_amcts.hpp>

namespace rl::players
{
ConcurrentAmcts::ConcurrentAmcts(int n_game_actions, std::unique_ptr<IEvaluator> evaluator_ptr, float cpuct, float temperature, int max_async_simulations_per_tree,float dirichlet_epsilon,float dirichlet_alpha,float default_visits, float default_wins)
    :n_game_actions_{ n_game_actions },
    evaluator_ptr_{ std::move(evaluator_ptr) },
    cpuct_{ cpuct },
    temperature_{ temperature },
    max_async_simulations_{ max_async_simulations_per_tree },
    dirichlet_epsilon_{dirichlet_epsilon},
    dirichlet_alpha_{dirichlet_alpha},
    default_n_{ default_visits },
    default_w_{ default_wins }
{
}

ConcurrentAmcts::~ConcurrentAmcts() = default;

std::vector<float> ConcurrentAmcts::search(const rl::common::IState* state_ptr, int minimum_no_simulations, std::chrono::duration<int, std::milli> minimum_duration)
{
    std::vector<const rl::common::IState*> state_ptrs = { {state_ptr} };
    auto& [probs, vs] = search_multiple(state_ptrs, minimum_no_simulations, minimum_duration);
    return probs.at(0);
}

std::pair<std::vector<std::vector<float>>, std::vector<float>> ConcurrentAmcts::search_multiple(const std::vector<const rl::common::IState*>& state_ptrs, int minimum_sims, std::chrono::duration<int, std::milli> minimum_duration)
{
    for (auto& state_ptr : state_ptrs)
    {
        if (state_ptr->is_terminal())
        {
            throw std::runtime_error("root is null ,but being searched by search tree");
        }
    }

    std::vector<bool> vec_is_state_forced(state_ptrs.size(), false);

    // TODO : check for each state it they have only 1 legal action
    int states_size = state_ptrs.size();
    int current_n_trees = states_size;

    auto probs_result = std::vector<std::vector<float>>(current_n_trees, std::vector<float>());
    auto values_result = std::vector<float>(current_n_trees, 0.0f);
    for (int i = 0; i < states_size;i++)
    {
        
        /*
        NOTE : This block is ignored for now 
        It is supposed to mark forced action states so we do not have to simulate
        and sets its probabilities as the actions mask  with the forced legal action probability as 1
        but alphazero needs the evaluation therefore it is ignored for now
        */ 

        // auto& state_ptr = state_ptrs.at(i);
        // auto legal_actions = state_ptr->actions_mask();
        // int n_legal_actions = 0;
        // for (bool m : legal_actions)
        // {
        //     n_legal_actions += m ? 1 : 0;
        // }
        // if (n_legal_actions == 1)
        // {
        //     // mark state as forced state
        //     vec_is_state_forced.at(i) = true;

        //     // add the probs
        //     std::vector<float>& res_ref = probs_result.at(i);
        //     res_ref.reserve(n_game_actions_);
        //     for (bool m : legal_actions)
        //     {
        //         res_ref.emplace_back(float(m));
        //     }
        // }
    }
    std::vector<std::vector<std::pair<rl::common::IState*, std::vector<Amcts2Info>>>> all_rollouts{};
    std::vector<std::unique_ptr<Amcts2Node>> root_nodes{};
    for (int i = 0;i < states_size;i++)
    {
        root_nodes.push_back(std::make_unique<Amcts2Node>(state_ptrs.at(i)->clone(), n_game_actions_, cpuct_));
        all_rollouts.push_back({});
    }

    auto t_start = std::chrono::high_resolution_clock::now();
    auto t_end = t_start + minimum_duration;

    int iteration{ 0 };


    while ((iteration * max_async_simulations_ <= minimum_sims) || (t_end > std::chrono::high_resolution_clock::now()))
    {
        std::vector<const rl::common::IState*> rollout_states{};
        std::vector<int> trees_idx{};
        for (int tree_id = 0;tree_id < current_n_trees;tree_id++)
        {
            if (vec_is_state_forced.at(tree_id))continue;
            auto& current_root_node = root_nodes.at(tree_id);
            auto& current_tree_rollouts = all_rollouts.at(tree_id);
            current_tree_rollouts.clear();
            for (int async_roll = 0; async_roll < max_async_simulations_;async_roll++)
            {
                // roll root node using dirichlet noise
                current_tree_rollouts.push_back(std::make_pair<rl::common::IState*, std::vector<Amcts2Info>>(nullptr, {}));
                auto& rollout_info = current_tree_rollouts.back();
                current_root_node->simulate_once(rollout_info, dirichlet_epsilon_, dirichlet_alpha_,default_n_, default_w_, current_root_node.get());

                // remove this rollout if no state is to be evaluated , Happens when the edge state is terminal 
                if (rollout_info.first == nullptr)
                {
                    current_tree_rollouts.pop_back();
                }
            }

            for (auto& a : current_tree_rollouts)
            {
                rollout_states.push_back(a.first);
                trees_idx.push_back(tree_id);
            }
        }


        std::vector<std::tuple<std::vector<float>, std::vector<float>>> trees_evaluations{};
        for (int tree_idx = 0;tree_idx < current_n_trees;tree_idx++)
        {
            trees_evaluations.push_back(std::make_tuple<std::vector<float>, std::vector<float>>(
                std::vector<float>(), std::vector<float>()));
        }
        auto& [probs, vs] = evaluator_ptr_->evaluate(rollout_states);
        int n_rollouts = rollout_states.size();
        for (int rollout_id = 0; rollout_id < n_rollouts; rollout_id++)
        {
            int probs_start = rollout_id * n_game_actions_;
            int probs_end = probs_start + n_game_actions_;
            int current_tree_id = trees_idx.at(rollout_id);
            std::vector<float>& current_probs_ref = std::get<0>(trees_evaluations.at(current_tree_id));
            std::vector<float>& current_vs_ref = std::get<1>(trees_evaluations.at(current_tree_id));
            for (int cell = probs_start; cell < probs_end; cell++)
            {
                current_probs_ref.push_back(probs.at(cell));
            }
            current_vs_ref.push_back(vs.at(rollout_id));
        }

        for (int tree_id = 0; tree_id < current_n_trees; tree_id++)
        {
            auto& current_root_node = root_nodes.at(tree_id);
            auto& current_evaluation_tuple = trees_evaluations.at(tree_id);
            // evaluate collected states
            auto& current_tree_rollout = all_rollouts.at(tree_id);
            int n_states = current_tree_rollout.size();
            evaluate_collected_states(current_root_node, current_evaluation_tuple, current_tree_rollout);
        }

        iteration++;

    } // end of simulations


    for (int tree_id = 0; tree_id < current_n_trees; tree_id++)
    {
        // check if state is forced
        if (vec_is_state_forced.at(tree_id))
        {
            auto& mask = state_ptrs.at(tree_id)->actions_mask();
        }
        else
        {
            auto& root_node = root_nodes.at(tree_id);
            probs_result.at(tree_id) = root_node->get_probs(temperature_);
            values_result.at(tree_id) = root_node->get_evaluation();
        }

    }

    return std::make_pair(std::move(probs_result), std::move(values_result));
}

void ConcurrentAmcts::evaluate_collected_states(std::unique_ptr<Amcts2Node>& root_node_ptr, std::tuple<std::vector<float>, std::vector<float>>& evaluations_tuple, std::vector<std::pair<rl::common::IState*, std::vector<Amcts2Info>>>& tree_rollouts)
{
    int n_states = static_cast<int>(tree_rollouts.size());
    std::vector<const rl::common::IState*> states_ptrs(n_states, nullptr);
    for (int i{ 0 }; i < n_states; i++)
    {
        states_ptrs.at(i) = std::get<0>(tree_rollouts.at(i));
    }
    std::vector<float>& probs = std::get<0>(evaluations_tuple);
    std::vector<float>& values = std::get<1>(evaluations_tuple);

    // backprob values
    for (int i{ 0 }; i < n_states; i++)
    {
        auto& visited_path = std::get<1>(tree_rollouts.at(i));
        auto& state_ptr = std::get<0>(tree_rollouts.at(i));
        float value = values.at(i);
        std::vector<float> state_probs(n_game_actions_, 0);
        int probs_start = i * n_game_actions_;
        for (int j = 0;j < n_game_actions_;j++)
        {
            state_probs.at(j) = probs.at(probs_start + j);
        }

        backpropogate(root_node_ptr, visited_path, value, states_ptrs.at(i)->player_turn(), state_probs);
    }

}

void ConcurrentAmcts::backpropogate(std::unique_ptr<Amcts2Node>& root_node_ptr, std::vector<Amcts2Info>& visited_path, float final_result, int final_player, std::vector<float>& probs)
{
    root_node_ptr->backpropogate(visited_path, 0, final_result, final_player, probs, default_n_, default_w_);
}
} // namespace rl::players



